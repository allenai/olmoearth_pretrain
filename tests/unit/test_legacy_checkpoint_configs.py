"""Real checkpoint configs must still rebuild the model they were trained with.

:mod:`tests.unit.test_legacy_config` pins the mechanics of
``patch_legacy_encoder_config`` against synthetic configs. This file pins the thing that
actually breaks in practice: a ``config.json`` written by a real pretraining run, which
carries whatever set of keys that run's code happened to save. When a field is removed
from ``EncoderConfig``, olmo-core's ``Config.from_dict`` raises on the leftover key --
and the caller sees only "Failed to construct 'encoder_config' in config", with the
omegaconf cause discarded -- so these configs stop loading with no usable diagnostic.

Each fixture is a pair:

``<name>.json``
    The ``model`` subtree of a real checkpoint's config.json. That subtree is the only
    part deserialized when a checkpoint is loaded.
``<name>.shapes.json``
    Its parameter manifest, recorded ONCE at a commit where the config still
    deserializes natively -- see ``recorded_at_commit``. This is what makes the test
    non-circular: the expectation comes from the code that ran when the checkpoint was
    trained, not from HEAD.

So a fixture fails loudly in either direction: HEAD can no longer build the config at
all, or HEAD builds something structurally different from what was trained. The second
is the dangerous one, because without olmo-core the standalone deserializer drops
unknown keys silently and would load such a checkpoint as though the feature was off.

To add a run: drop its ``model`` subtree in as ``<name>.json``, then generate
``<name>.shapes.json`` from a checkout where it loads unpatched (see
``scripts/tools/record_legacy_config_shapes.py``).
"""

import copy
import json
from pathlib import Path

import pytest
import torch

from olmoearth_pretrain.model_loader import patch_legacy_encoder_config

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "legacy_configs"

CONFIGS = sorted(
    p for p in FIXTURE_DIR.glob("*.json") if not p.name.endswith(".shapes.json")
)


def _build_manifest(encoder_config: dict) -> dict[str, list[int]]:
    """Parameter name -> shape, built without allocating any storage.

    The meta device gives every parameter its real shape and no memory, which keeps a
    768-wide, 170M-parameter fixture free to check under ``pytest -n auto``.
    """
    from olmoearth_pretrain.nn.flexi_vit import EncoderConfig

    with torch.device("meta"):
        model = EncoderConfig.from_dict(encoder_config).build()
    return {name: list(t.shape) for name, t in model.state_dict().items()}


def test_fixtures_exist() -> None:
    """Guard against the glob silently matching nothing."""
    assert CONFIGS, f"no legacy config fixtures found in {FIXTURE_DIR}"


@pytest.mark.parametrize("config_path", CONFIGS, ids=lambda p: p.stem)
def test_legacy_checkpoint_config_rebuilds_identical_model(config_path: Path) -> None:
    """A real checkpoint config still builds, and builds the SAME parameter set."""
    golden_path = config_path.with_suffix(".shapes.json")
    assert golden_path.exists(), f"missing parameter manifest for {config_path.name}"
    golden = json.loads(golden_path.read_text())

    config_dict = json.loads(config_path.read_text())
    patched = patch_legacy_encoder_config(copy.deepcopy(config_dict))
    manifest = _build_manifest(patched["model"]["encoder_config"])

    expected = golden["shapes"]
    missing = sorted(set(expected) - set(manifest))
    added = sorted(set(manifest) - set(expected))
    changed = {
        k: (expected[k], manifest[k])
        for k in set(expected) & set(manifest)
        if expected[k] != manifest[k]
    }
    assert not (missing or added or changed), (
        f"{config_path.name} no longer rebuilds the model recorded at "
        f"{golden['recorded_at_commit']}:\n"
        f"  parameters lost:    {missing}\n"
        f"  parameters added:   {added}\n"
        f"  shapes changed:     {changed}"
    )


@pytest.mark.parametrize("config_path", CONFIGS, ids=lambda p: p.stem)
def test_patching_is_required(config_path: Path) -> None:
    """The fixture is only meaningful while it still needs patching.

    If HEAD grows the removed field back (or the fixture is replaced by a current
    config), the test above would pass trivially. Fail here instead, so the fixture is
    knowingly retired rather than quietly becoming a no-op.
    """
    from olmoearth_pretrain.nn.flexi_vit import EncoderConfig

    raw = json.loads(config_path.read_text())["model"]["encoder_config"]
    with pytest.raises(Exception):
        with torch.device("meta"):
            EncoderConfig.from_dict(copy.deepcopy(raw)).build()
