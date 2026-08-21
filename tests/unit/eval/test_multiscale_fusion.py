"""Tests for the multi-scale (window + center-pixel) embedding fusion."""

from typing import Any, cast

import pytest
import torch

from olmoearth_pretrain.internal.all_evals import EMBEDDING_EVAL_TASKS
from olmoearth_pretrain.train.callbacks.evaluator_callback import _crop_to_center_pixel


class _FakeSample:
    """Stands in for MaskedOlmoEarthSample's dict round-trip."""

    def __init__(self, values: dict[str, torch.Tensor]) -> None:
        self.values = values

    def as_dict(self) -> dict[str, torch.Tensor]:
        return self.values

    @classmethod
    def from_dict(cls, values: dict[str, torch.Tensor]) -> "_FakeSample":
        return cls(values)


@pytest.fixture(autouse=True)
def _patch_sample(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "olmoearth_pretrain.train.callbacks.evaluator_callback.MaskedOlmoEarthSample",
        _FakeSample,
    )


def test_crop_reduces_spatial_dims_to_one() -> None:
    """Spatial tensors and their masks both collapse to a single pixel."""
    sample = _FakeSample(
        {
            "sentinel2_l2a": torch.zeros(2, 16, 16, 12, 12),
            "sentinel2_l2a_mask": torch.zeros(2, 16, 16, 12),
            "timestamps": torch.zeros(2, 12, 3),
        }
    )
    out = _crop_to_center_pixel(cast(Any, sample)).as_dict()
    assert out["sentinel2_l2a"].shape == (2, 1, 1, 12, 12)
    assert out["sentinel2_l2a_mask"].shape == (2, 1, 1, 12)


def test_timestamps_pass_through_uncropped() -> None:
    """Timestamps are [B, T, 3] with no spatial axis; cropping would truncate them."""
    timestamps = torch.arange(2 * 12 * 3, dtype=torch.float32).reshape(2, 12, 3)
    sample = _FakeSample(
        {"sentinel2_l2a": torch.zeros(2, 16, 16, 12, 12), "timestamps": timestamps}
    )
    out = _crop_to_center_pixel(cast(Any, sample)).as_dict()
    assert out["timestamps"].shape == (2, 12, 3)
    assert torch.equal(out["timestamps"], timestamps)


def test_crop_takes_the_center_pixel() -> None:
    """The kept pixel must be the labeled one, which the window is centered on."""
    image = torch.arange(16 * 16, dtype=torch.float32).reshape(1, 16, 16, 1, 1)
    out = _crop_to_center_pixel(cast(Any, _FakeSample({"x": image}))).as_dict()
    assert out["x"].item() == image[0, 8, 8, 0, 0].item()


@pytest.mark.parametrize("fusion", ["concat", "mean"])
def test_fused_tasks_registered(fusion: str) -> None:
    """Each fused task matches its single-scale sibling apart from the fusion."""
    name = (
        f"ethiopia_crops_year_aligned_ws16_ps1_sentinel1_sentinel2_landsat_"
        f"fuse{fusion}_knn"
    )
    task = EMBEDDING_EVAL_TASKS[name]
    assert task.multiscale_fusion == fusion
    assert task.window_size == 16
    assert task.patch_size == 1
    assert task.use_center_token

    baseline = EMBEDDING_EVAL_TASKS[
        "ethiopia_crops_year_aligned_ws16_ps1_sentinel1_sentinel2_landsat_knn"
    ]
    assert baseline.multiscale_fusion is None
    assert task.input_modalities == baseline.input_modalities
    assert task.eval_mode == baseline.eval_mode
