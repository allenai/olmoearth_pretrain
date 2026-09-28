"""Real checkpoint configs must build the model they were trained with.

Each fixture in ``tests/fixtures/checkpoint_configs`` is a pair:

``<name>.json``
    The ``model`` subtree of a real checkpoint's config.json, in the current schema
    (converted once from the training branch's layout by
    ``scripts/official/v1_3/convert_legacy_checkpoint.py``). That subtree is the only
    part deserialized when a checkpoint is loaded.

``<name>.shapes.json``
    The parameter manifest of the WHOLE model it builds -- encoder, decoder, heads --
    recorded from the code the checkpoint was trained with (see ``recorded_at_commit``)
    and mapped to the current parameter names. This is what makes the test
    non-circular: the expectation comes from the training-time code, not from HEAD.

So a fixture fails loudly in either direction: HEAD can no longer deserialize the
config, or HEAD builds something structurally different from what was trained -- the
dangerous case, because the checkpoint's weights would then load into the wrong model.

To add a run: convert its config with ``--config-only`` and drop the ``model`` subtree
in as ``<name>.json``; generate ``<name>.shapes.json`` with
``scripts/tools/record_legacy_config_shapes.py`` from a checkout that builds it natively.
"""

import json
from pathlib import Path

import pytest
import torch

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "checkpoint_configs"

CONFIGS = sorted(
    p for p in FIXTURE_DIR.glob("*.json") if not p.name.endswith(".shapes.json")
)


def _build_manifest(model_config: dict) -> dict[str, list[int]]:
    """Parameter name -> shape for the whole model, built on the meta device.

    The meta device gives every parameter its real shape and no memory, which keeps a
    768-wide, 170M-parameter fixture free to check under ``pytest -n auto``.
    """
    # Imported before entering the meta context: the model modules pull in
    # torch._dynamo at import time, which trips over the device override.
    import olmoearth_pretrain.nn.latent_mim  # noqa: F401
    from olmoearth_pretrain.config import Config

    with torch.device("meta"):
        model = Config.from_dict(model_config).build()
    return {name: list(t.shape) for name, t in model.state_dict().items()}


def test_fixtures_exist() -> None:
    """Guard against the glob silently matching nothing."""
    assert CONFIGS, f"no checkpoint config fixtures found in {FIXTURE_DIR}"


@pytest.mark.parametrize("config_path", CONFIGS, ids=lambda p: p.stem)
def test_checkpoint_config_builds_recorded_model(config_path: Path) -> None:
    """A real checkpoint config deserializes natively and builds the SAME parameter set."""
    golden_path = config_path.with_suffix(".shapes.json")
    assert golden_path.exists(), f"missing parameter manifest for {config_path.name}"
    golden = json.loads(golden_path.read_text())
    expected = golden["shapes"]

    manifest = _build_manifest(json.loads(config_path.read_text())["model"])
    missing = sorted(set(expected) - set(manifest))
    added = sorted(set(manifest) - set(expected))
    changed = {
        k: (expected[k], manifest[k])
        for k in set(expected) & set(manifest)
        if expected[k] != manifest[k]
    }
    assert not (missing or added or changed), (
        f"{config_path.name} no longer builds the model recorded at "
        f"{golden['recorded_at_commit']}:\n"
        f"  parameters lost:    {missing}\n"
        f"  parameters added:   {added}\n"
        f"  shapes changed:     {changed}"
    )
