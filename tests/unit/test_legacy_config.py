"""Legacy-checkpoint handling: configs written before features were removed.

Neither deserializer copes with a stale key on its own. With olmo-core installed,
``Config.from_dict`` runs through omegaconf and RAISES on any unknown key, so an old
checkpoint carrying e.g. ``attn_window_size: null`` fails to load at all. Without
olmo-core, the standalone path silently DROPS unknown keys, so a checkpoint that
genuinely used a removed feature loads as though it had been off -- quietly building a
different model than the one trained. These tests pin the three behaviours that make
old checkpoints safe:

1. a removed field left at its feature-off value is stripped (inert),
2. such a config then loads and builds cleanly, and
3. a removed field that was actually USED raises instead of loading.
"""

import copy

import pytest

from olmoearth_pretrain.model_loader import (
    REMOVED_ENCODER_FIELDS,
    REMOVED_SUPERVISION_MODALITY_FIELDS,
    patch_legacy_encoder_config,
)


def _config_dict(**encoder_overrides: object) -> dict:
    """A minimal checkpoint-shaped config dict with a register bottleneck."""
    encoder = {
        "supported_modality_names": ["sentinel2_l2a", "latlon"],
        "embedding_size": 16,
        "num_heads": 2,
        "depth": 2,
        "use_linear_patch_embed": True,
        "position_encoding": "rope",
        "use_register_bottleneck": True,
        "register_grid_size": 0,
        "register_dim": 8,
    }
    encoder.update(encoder_overrides)
    return {"model": {"encoder_config": encoder}}


def _supervised_config_dict(**modality_overrides: object) -> dict:
    """A checkpoint-shaped config dict with one register-supervision head."""
    config_dict = _config_dict()
    modality = {
        "task_type": "regression",
        "num_output_channels": 1,
        "weight": 0.01,
        "regression_loss_type": "l1",
    }
    modality.update(modality_overrides)
    config_dict["model"]["supervision_head_config"] = {
        "register_supervision": True,
        "modality_configs": {"srtm": modality},
    }
    return config_dict


def _active_value(name: str) -> object:
    """A value for ``name`` that would have turned its removed feature ON."""
    return {
        "attn_window_size": 8,
        "register_read_layers": [3, 6, 9, 12],
        "register_shared_read_kv": True,
        "register_fused_read": "uniform",
        "register_learned_read_weighting": True,
        "register_latent_every_n": 2,
        "register_output_dim": 128,
        "register_unit_norm": True,
        "register_unit_norm_scale": 4.0,
        "band_dropout_groups": {"sentinel2_l2a": [["B02"], ["B11"]]},
        "patch_embed_linear_skip": True,
        "merge_bandsets": True,
        "merge_after_layer": 3,
        "register_students": [
            {"name": "lin128", "projection_type": "linear", "dims": [128, 64]}
        ],
        "register_temporal_anchor": "year_start",
    }[name]


def _active_supervision_value(name: str) -> object:
    """A value for ``name`` that would have turned its removed head feature ON."""
    return {
        "time_conditioned": True,
        "time_harmonics": 6,
        "time_mlp_hidden_dim": 128,
    }[name]


# Every removed field must be exercised by the tests below; a new removal that is not
# listed in _active_value fails here rather than going silently untested.
def test_every_removed_field_has_an_active_example() -> None:
    """The fixture covers the whole registry."""
    for name in REMOVED_ENCODER_FIELDS:
        assert _active_value(name) is not None, name
    for name in REMOVED_SUPERVISION_MODALITY_FIELDS:
        assert _active_supervision_value(name) is not None, name


@pytest.mark.parametrize(
    "name,inert",
    [
        (name, inert)
        for name, removed in REMOVED_ENCODER_FIELDS.items()
        for inert in removed.inert
    ],
)
def test_inert_removed_field_is_stripped(name: str, inert: object) -> None:
    """(i) A removed field left at its feature-off value is dropped, not an error."""
    config_dict = _config_dict(**{name: inert})
    patched = patch_legacy_encoder_config(config_dict)
    assert name not in patched["model"]["encoder_config"]
    # The caller's dict is never mutated in place.
    assert name in config_dict["model"]["encoder_config"]


@pytest.mark.parametrize("name", sorted(REMOVED_ENCODER_FIELDS))
def test_active_removed_field_raises(name: str) -> None:
    """(iii) A removed field that was actually USED refuses to load."""
    config_dict = _config_dict(**{name: _active_value(name)})
    with pytest.raises(ValueError, match="since been removed"):
        patch_legacy_encoder_config(config_dict)


def test_error_names_every_active_feature() -> None:
    """The error lists each offending field, not just the first."""
    config_dict = _config_dict(
        register_read_layers=[3, 6], register_unit_norm=True, attn_window_size=8
    )
    with pytest.raises(ValueError) as excinfo:
        patch_legacy_encoder_config(config_dict)
    message = str(excinfo.value)
    for name in ("register_read_layers", "register_unit_norm", "attn_window_size"):
        assert name in message
    assert "multi-depth register reads (mdr)" in message


def test_config_without_legacy_fields_is_untouched() -> None:
    """A current config passes through unchanged."""
    config_dict = _config_dict()
    before = copy.deepcopy(config_dict)
    assert patch_legacy_encoder_config(config_dict) == before


def test_legacy_config_deserializes_and_builds() -> None:
    """(ii) A config carrying every inert removed field loads and builds a real model.

    Goes through the real ``EncoderConfig.from_dict``, which is the step that rejects
    unknown keys under olmo-core -- so this fails if any leftover survives the strip.
    """
    from olmoearth_pretrain.nn.flexi_vit import EncoderConfig

    inert_values = {
        name: removed.inert[0]
        for name, removed in REMOVED_ENCODER_FIELDS.items()
        if removed.inert
    }
    config_dict = _config_dict(**inert_values)
    enc = patch_legacy_encoder_config(config_dict)["model"]["encoder_config"]
    config = EncoderConfig.from_dict(enc)
    for name in REMOVED_ENCODER_FIELDS:
        assert not hasattr(config, name), f"{name} should no longer be a field"
    assert config.build() is not None


def test_missing_register_dim_restored_for_legacy_bottleneck() -> None:
    """register_dim used to default to embedding_size // 2 and is now required."""
    config_dict = _config_dict()
    del config_dict["model"]["encoder_config"]["register_dim"]
    patched = patch_legacy_encoder_config(config_dict)
    assert patched["model"]["encoder_config"]["register_dim"] == 8


@pytest.mark.parametrize(
    "name,inert",
    [
        (name, inert)
        for name, removed in REMOVED_SUPERVISION_MODALITY_FIELDS.items()
        for inert in removed.inert
    ],
)
def test_inert_removed_supervision_field_is_stripped(name: str, inert: object) -> None:
    """(i) for the supervision head: a feature-off leftover on a modality is dropped."""
    config_dict = _supervised_config_dict(**{name: inert})
    patched = patch_legacy_encoder_config(config_dict)
    modality = patched["model"]["supervision_head_config"]["modality_configs"]["srtm"]
    assert name not in modality
    # The caller's dict is never mutated in place.
    original = config_dict["model"]["supervision_head_config"]["modality_configs"]
    assert name in original["srtm"]


@pytest.mark.parametrize("name", sorted(REMOVED_SUPERVISION_MODALITY_FIELDS))
def test_active_removed_supervision_field_raises(name: str) -> None:
    """(iii) for the supervision head: a USED removed field refuses to load."""
    config_dict = _supervised_config_dict(**{name: _active_supervision_value(name)})
    with pytest.raises(ValueError, match="since been removed") as excinfo:
        patch_legacy_encoder_config(config_dict)
    assert f"modality_configs.srtm.{name}" in str(excinfo.value)


def test_legacy_supervised_config_deserializes_and_builds() -> None:
    """(ii) for the supervision head: every inert leftover stripped, the model builds.

    ``SupervisionHeadConfig.from_dict`` is the step that rejects unknown modality keys,
    so this fails if any leftover survives the strip.
    """
    from olmoearth_pretrain.nn.supervision_head import SupervisionHeadConfig

    inert_values = {
        name: removed.inert[0]
        for name, removed in REMOVED_SUPERVISION_MODALITY_FIELDS.items()
    }
    config_dict = _supervised_config_dict(**inert_values)
    head_dict = patch_legacy_encoder_config(config_dict)["model"][
        "supervision_head_config"
    ]
    config = SupervisionHeadConfig.from_dict(head_dict)
    modality = config.modality_configs["srtm"]
    for name in REMOVED_SUPERVISION_MODALITY_FIELDS:
        assert not hasattr(modality, name), f"{name} should no longer be a field"
    assert config.build(embedding_dim=8, max_patch_size=4) is not None
