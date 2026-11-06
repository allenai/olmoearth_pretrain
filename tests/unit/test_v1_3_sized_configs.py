"""The resized v1.3 runs must differ from the released Base config only in size fields.

``rc_ld1_pixtgt_pix512_d1024.py`` and ``rc_ld1_pixtgt_pix512_large.py`` start from the
release model config (``rc_ld1_pixtgt_pix512.py``) and replace the encoder / decoder /
register sizes. This diffs the full model and train-module configs against the release
ones so any other drift fails here, and builds each model on the meta device.
"""

import importlib.util
from pathlib import Path
from typing import Any

import pytest
import torch

from olmoearth_pretrain.internal.experiment import SubCmd

ABLATIONS = (
    Path(__file__).resolve().parents[2] / "scripts" / "official" / "v1_3" / "ablations"
)

SIZE_FIELDS = {
    "encoder_config.embedding_size",
    "encoder_config.depth",
    "encoder_config.num_heads",
    "encoder_config.perceiver_config.register_dim",
    "encoder_config.perceiver_config.attn_dim",
    "decoder_config.encoder_embedding_size",
    "decoder_config.decoder_embedding_size",
    "decoder_config.depth",
    "decoder_config.num_heads",
    "decoder_config.register_dim",
}

EXPECTED = {
    "rc_ld1_pixtgt_pix512_d1024": dict(width=1024, enc_depth=12, heads=16),
    "rc_ld1_pixtgt_pix512_large": dict(width=1024, enc_depth=24, heads=16),
}


def _load(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, ABLATIONS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _flatten(d: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(d, dict):
        out: dict[str, Any] = {}
        for k, v in d.items():
            out.update(_flatten(v, f"{prefix}{k}."))
        return out
    return {prefix.rstrip("."): d}


@pytest.fixture(scope="module")
def release() -> Any:
    return _load("rc_ld1_pixtgt_pix512")


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_only_size_fields_differ(name: str, release: Any) -> None:
    """Model and train-module configs match the release apart from the size fields."""
    sized = _load(name)
    common = release.build_common_components(
        f"{name}.py", SubCmd.dry_run, "test", "local", []
    )
    base_model = _flatten(release.build_model_config(common).as_dict())
    sized_model = _flatten(sized.build_model_config(common).as_dict())
    assert base_model.keys() == sized_model.keys()
    differ = {k for k in base_model if base_model[k] != sized_model[k]}
    assert differ <= SIZE_FIELDS, differ - SIZE_FIELDS

    exp = EXPECTED[name]
    assert sized_model["encoder_config.embedding_size"] == exp["width"]
    assert sized_model["encoder_config.depth"] == exp["enc_depth"]
    assert sized_model["encoder_config.num_heads"] == exp["heads"]
    assert sized_model["decoder_config.depth"] == base_model["decoder_config.depth"]
    assert sized_model["encoder_config.perceiver_config.register_dim"] == exp["width"]
    assert sized_model["encoder_config.perceiver_config.student_dims"] == [128, 64]

    # The train module comes from the release builders untouched.
    helper = _load("rc_ld1_pixtgt_pix512_sized")
    assert helper.build_train_module_config(common).as_dict() == (
        release.build_train_module_config(common).as_dict()
    )


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_builds_on_meta(name: str, release: Any) -> None:
    """The resized model builds, with every parameter at least as wide as Base's."""
    import olmoearth_pretrain.nn.latent_mim  # noqa: F401

    sized = _load(name)
    common = release.build_common_components(
        f"{name}.py", SubCmd.dry_run, "test", "local", []
    )
    with torch.device("meta"):
        base_model = release.build_model_config(common).build()
        sized_model = sized.build_model_config(common).build()
    n_base = sum(p.numel() for p in base_model.parameters())
    n_sized = sum(p.numel() for p in sized_model.parameters())
    assert n_sized > n_base


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_exposes_eval_job_builders(name: str, release: Any) -> None:
    """In-loop eval jobs look these names up on the script; they must be the release's.

    ``checkpoint_sweep_evals`` falls back to the default common components (a 3-group
    S2 tokenization) when ``build_common_components`` is missing, and then cannot load
    the checkpoint (the 80k eval of v1_3_d1024 failed that way, 2026-11-06).
    """
    sized = _load(name)
    for attr in ("build_common_components", "build_train_module_config"):
        assert getattr(sized, attr) is getattr(release, attr), attr
    common = sized.build_common_components(
        f"{name}.py", SubCmd.dry_run, "test", "local", []
    )
    assert (
        common.tokenization_config.as_dict()
        == release.build_common_components(
            "x.py", SubCmd.dry_run, "test", "local", []
        ).tokenization_config.as_dict()
    )
    evaluator = sized.build_trainer_config(common).callbacks["downstream_evaluator"]
    assert evaluator.beaker_eval_module_path == sized.MODULE_PATH
    release_evaluator = release.build_trainer_config(common).callbacks[
        "downstream_evaluator"
    ]
    assert evaluator.tasks.keys() == release_evaluator.tasks.keys()
