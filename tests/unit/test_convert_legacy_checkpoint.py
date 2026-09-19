"""The v1.3 checkpoint converter turns the training branch's layout into the current one.

Pinned against the release checkpoint's ORIGINAL config (``tests/fixtures/perceiver_legacy``)
and its converted twin (``tests/fixtures/checkpoint_configs``), plus a state-dict round
trip on a small model.
"""

import copy
import importlib.util
import json
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).parents[2]
_spec = importlib.util.spec_from_file_location(
    "convert_legacy_checkpoint",
    ROOT / "scripts" / "official" / "v1_3" / "convert_legacy_checkpoint.py",
)
assert _spec is not None and _spec.loader is not None
convert = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(convert)

LEGACY_DIR = ROOT / "tests" / "fixtures" / "perceiver_legacy"
CURRENT_DIR = ROOT / "tests" / "fixtures" / "checkpoint_configs"
RELEASE = (
    "regbtl_v1_2_gdyn_d768_proj128lin_sup768_w1_newsamp_psuniform_stunorm_mlpgram1"
)


def _release_legacy_model() -> dict:
    return json.loads((LEGACY_DIR / f"{RELEASE}.json").read_text())["model"]


def test_release_config_converts_to_the_fixture() -> None:
    """Converting the release checkpoint's original config gives the committed fixture."""
    import olmoearth_pretrain.nn.latent_mim  # noqa: F401
    from olmoearth_pretrain.config import Config

    converted = Config.from_dict(
        convert.convert_model_config(_release_legacy_model())
    ).as_config_dict()
    expected = json.loads((CURRENT_DIR / f"{RELEASE}.json").read_text())["model"]
    assert converted == expected


def test_release_config_conversion_is_idempotent() -> None:
    """A config already in the current schema comes back unchanged."""
    current = json.loads((CURRENT_DIR / f"{RELEASE}.json").read_text())["model"]
    assert convert.convert_model_config(copy.deepcopy(current)) == current


def test_original_release_config_needs_converting() -> None:
    """The legacy fixture is only meaningful while HEAD cannot read it directly."""
    from olmoearth_pretrain.config import Config

    with pytest.raises(Exception):
        Config.from_dict(_release_legacy_model())


def test_active_removed_feature_is_refused() -> None:
    """A checkpoint that USED a removed feature cannot be converted."""
    model = _release_legacy_model()
    model["encoder_config"]["register_grid_size"] = 16
    with pytest.raises(convert.UnconvertibleCheckpoint, match="register_grid_size=16"):
        convert.convert_model_config(model)


@pytest.mark.parametrize(
    "old,new",
    [
        ("encoder.register_bottleneck.norm.weight", "encoder.perceiver.norm.weight"),
        ("encoder.register_projection.weight", "encoder.register_student.0.weight"),
        ("encoder.register_projection_norm.bias", "encoder.register_student.1.bias"),
        (
            "encoder.register_back_projections.128.0.weight",
            "register_distillation_head.back_projections.128.0.weight",
        ),
        ("encoder.norm.weight", "encoder.norm.weight"),
        ("register_bottleneck.norm.weight", "perceiver.norm.weight"),
    ],
)
def test_key_mapping_round_trips(old: str, new: str) -> None:
    """Old -> new -> old for every kind of moved parameter (and an unmoved one)."""
    assert convert.convert_key(old) == new
    assert convert.legacy_key(new) == old


def test_release_manifest_keys_map_onto_the_current_model() -> None:
    """Every parameter name in the recorded manifest is a current parameter name."""
    import olmoearth_pretrain.nn.latent_mim  # noqa: F401
    from olmoearth_pretrain.config import Config

    model_cfg = json.loads((CURRENT_DIR / f"{RELEASE}.json").read_text())["model"]
    with torch.device("meta"):
        model = Config.from_dict(model_cfg).build()
    current = set(model.state_dict())
    golden = json.loads((CURRENT_DIR / f"{RELEASE}.shapes.json").read_text())["shapes"]
    assert set(golden) == current
    # And the inverse mapping reproduces the names the checkpoint actually stores.
    legacy_names = {convert.legacy_key(k) for k in current}
    assert any(k.startswith("encoder.register_bottleneck.") for k in legacy_names)
    assert any(k.startswith("encoder.register_back_projections.") for k in legacy_names)


def test_state_dict_round_trip_on_a_small_model() -> None:
    """A state dict saved under the old names converts and strict-loads."""
    from olmoearth_pretrain.nn.flexi_vit import (
        EncoderConfig,
        PerceiverConfig,
        PredictorConfig,
    )
    from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig
    from olmoearth_pretrain.nn.register_distillation_head import (
        RegisterDistillationHeadConfig,
    )

    modalities = ["sentinel2_l2a", "latlon"]
    config = LatentMIMConfig(
        encoder_config=EncoderConfig(
            supported_modality_names=modalities,
            embedding_size=16,
            num_heads=2,
            depth=1,
            mlp_ratio=2.0,
            max_patch_size=4,
            min_patch_size=1,
            max_sequence_length=4,
            position_encoding="rope",
            perceiver_config=PerceiverConfig(
                register_dim=8, student_dims=[4, 2], student_output_norm=True
            ),
        ),
        decoder_config=PredictorConfig(
            supported_modality_names=modalities,
            encoder_embedding_size=16,
            decoder_embedding_size=16,
            num_heads=2,
            depth=1,
            mlp_ratio=2.0,
            max_sequence_length=4,
            position_encoding="rope",
            use_perceiver=True,
            register_dim=8,
        ),
        register_distillation_head_config=RegisterDistillationHeadConfig(),
    )
    model = config.build()
    legacy = {convert.legacy_key(k): v for k, v in model.state_dict().items()}
    assert any(k.startswith("encoder.register_projection_norm.") for k in legacy)
    model.load_state_dict(convert.convert_state_dict(legacy), strict=True)
    # An encoder-only dump drops the heads it cannot house.
    enc_only = {
        k[len("encoder.") :]: v for k, v in legacy.items() if k.startswith("encoder.")
    }
    enc_only["register_back_projections.4.weight"] = torch.zeros(8, 4)
    model.encoder.load_state_dict(convert.convert_state_dict(enc_only), strict=True)
