"""Unit tests for olmoearth_pretrain/evals/tile_regression.py."""

import json
from pathlib import Path

import pytest
import shapely
import torch
from rasterio.crs import CRS
from rslearn.dataset import Dataset, Window
from rslearn.utils import Projection, STGeometry
from rslearn.utils.feature import Feature
from rslearn.utils.vector_format import GeojsonVectorFormat
from upath import UPath

from olmoearth_pretrain.evals.metrics import EvalMetric
from olmoearth_pretrain.evals.tile_regression import (
    pool_tiles,
    ridge_readouts,
    tile_ridge_task_name,
    window_label_matrix,
)


def test_pool_tiles_means_consecutive_groups() -> None:
    """Each sample's tiles are averaged; its label is kept once."""
    emb = torch.arange(12, dtype=torch.float32).reshape(6, 2)  # 2 samples x 3 tiles
    labels = torch.tensor([5.0, 5.0, 5.0, 7.0, 7.0, 7.0])
    pooled, pooled_labels = pool_tiles(emb, labels, tiles_per_sample=3)
    assert torch.allclose(pooled, torch.tensor([[2.0, 3.0], [8.0, 9.0]]))
    assert torch.equal(pooled_labels, torch.tensor([5.0, 7.0]))


def test_pool_tiles_rejects_ragged_groups() -> None:
    """A tile count that doesn't divide evenly is an ordering bug, not data."""
    with pytest.raises(ValueError, match="groups of 4"):
        pool_tiles(torch.zeros(6, 2), torch.zeros(6), tiles_per_sample=4)


def _make_windows(tmp_path: Path, props: list[dict]) -> list[Window]:
    ds_path = UPath(tmp_path / "ds")
    ds_path.mkdir(parents=True)
    (ds_path / "config.json").write_text(
        json.dumps({"layers": {"label": {"type": "vector"}}})
    )
    dataset = Dataset(ds_path)
    projection = Projection(CRS.from_epsg(32614), 10, -10)
    windows = []
    for i, p in enumerate(props):
        bounds = (i * 128, 0, (i + 1) * 128, 128)
        window = Window(
            storage=dataset.storage,
            group="default",
            name=f"w{i}",
            projection=projection,
            bounds=bounds,
            time_range=None,
        )
        window.save()
        GeojsonVectorFormat().encode_vector(
            window.get_layer_dir("label"),
            [Feature(STGeometry(projection, shapely.box(*bounds), None), p)],
        )
        window.mark_layer_completed("label")
        windows.append(window)
    return windows


def test_window_label_matrix_reads_properties_with_nan_for_missing(
    tmp_path: Path,
) -> None:
    """Missing properties come back as NaN, present ones as floats, in window order."""
    windows = _make_windows(
        tmp_path,
        [{"a": 1.5, "b": 2.0}, {"a": 3.0}, {"b": -1.0}],
    )
    y = window_label_matrix(windows, "label", ["a", "b"])
    assert y.shape == (3, 2)
    assert y[0].tolist() == [1.5, 2.0]
    assert y[1, 0] == 3.0 and torch.isnan(y[1, 1])
    assert torch.isnan(y[2, 0]) and y[2, 1] == -1.0


def test_ridge_readouts_recovers_linear_signal_and_skips_nan_labels() -> None:
    """A noiseless linear target is fit near-perfectly; NaN rows are dropped per label."""
    g = torch.Generator().manual_seed(0)
    w_true = torch.randn(8, 2, generator=g)

    def split(n: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.randn(n, 8, generator=g)
        return x, x @ w_true + 3.0

    (xtr, ytr), (xva, yva), (xte, yte) = split(400), split(100), split(100)
    yva[:10, 1] = float("nan")
    results = ridge_readouts(
        {"train": xtr, "val": xva, "test": xte},
        {"train": ytr, "val": yva, "test": yte},
        properties=["p0", "p1"],
    )
    for prop in ("p0", "p1"):
        val_result, test_result = results[prop]
        assert val_result.primary_metric == EvalMetric.R2
        assert val_result.primary > 0.999
        assert test_result is not None and test_result.primary > 0.999
    assert results["p1"][0].metrics["n"] == 90.0
    assert results["p0"][0].metrics["n"] == 100.0


def test_ridge_readouts_without_test_split() -> None:
    """With no test split, only val results are returned."""
    x = torch.randn(50, 4)
    y = x[:, :1]
    results = ridge_readouts(
        {"train": x, "val": x, "test": None}, {"train": y, "val": y}, properties=["p"]
    )
    assert results["p"][1] is None


def test_tile_ridge_task_name() -> None:
    """Readouts are reported under a synthetic per-property task name."""
    assert (
        tile_ridge_task_name("us_soc", "log_income") == "us_soc_tile_ridge_log_income"
    )
