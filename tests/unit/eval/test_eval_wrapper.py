"""Tests for eval wrapper feature exit depths."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from olmoearth_pretrain.evals.datasets.configs import TaskType
from olmoearth_pretrain.evals.eval_wrapper import OlmoEarthEvalWrapper
from olmoearth_pretrain.nn.pooling import PoolingType


class DummyOlmoEarthModel(nn.Module):
    """Minimal model stub for eval wrapper tests."""

    def __init__(self, depth: int = 24) -> None:
        """Stand in for a flexi-vit-style model with a configurable depth."""
        super().__init__()
        self.blocks = [object()] * depth
        self.supported_modality_names = ["sentinel2_l2a", "sentinel1"]
        self.calls: list[dict] = []

    def forward(self, *_args: object, **kwargs: object) -> dict[str, str]:
        """Record the call and return a placeholder token bundle."""
        self.calls.append(kwargs)
        return {"tokens_and_masks": "dummy_tokens"}


def test_olmoearth_eval_wrapper_uses_fast_pass_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default eval path should preserve fast-pass behavior."""
    monkeypatch.setattr(
        "olmoearth_pretrain.evals.eval_wrapper.pool_unmasked_tokens",
        lambda embeddings, *_args, **_kwargs: embeddings,
    )
    model = DummyOlmoEarthModel()
    wrapper = OlmoEarthEvalWrapper(
        model=model,
        task_type=TaskType.CLASSIFICATION,
        patch_size=4,
        pooling_type=PoolingType.MEAN,
    )

    embeddings, labels = wrapper(object(), torch.tensor([1]))  # type: ignore[arg-type]

    assert embeddings == "dummy_tokens"
    assert torch.equal(labels, torch.tensor([1]))
    assert model.calls == [{"patch_size": 4, "token_exit_cfg": None, "fast_pass": True}]


def test_olmoearth_eval_wrapper_uses_token_exit_cfg_when_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Feature exit depth should disable fast-pass and set a uniform exit config."""
    monkeypatch.setattr(
        "olmoearth_pretrain.evals.eval_wrapper.pool_unmasked_tokens",
        lambda embeddings, *_args, **_kwargs: embeddings,
    )
    model = DummyOlmoEarthModel()
    wrapper = OlmoEarthEvalWrapper(
        model=model,
        task_type=TaskType.CLASSIFICATION,
        patch_size=4,
        pooling_type=PoolingType.MEAN,
        feature_exit_depth=12,
    )

    embeddings, labels = wrapper(object(), torch.tensor([1]))  # type: ignore[arg-type]

    assert embeddings == "dummy_tokens"
    assert torch.equal(labels, torch.tensor([1]))
    assert model.calls == [
        {
            "patch_size": 4,
            "token_exit_cfg": {"sentinel2_l2a": 12, "sentinel1": 12},
            "fast_pass": False,
        }
    ]


def test_olmoearth_eval_wrapper_rejects_invalid_feature_exit_depth() -> None:
    """Feature exit depth should be bounded by encoder depth."""
    model = DummyOlmoEarthModel(depth=4)
    wrapper = OlmoEarthEvalWrapper(
        model=model,
        task_type=TaskType.CLASSIFICATION,
        patch_size=4,
        pooling_type=PoolingType.MEAN,
        feature_exit_depth=5,
    )

    with pytest.raises(ValueError, match=r"feature_exit_depth must be in \[0, 4\]"):
        wrapper(object(), torch.tensor([1]))  # type: ignore[arg-type]
