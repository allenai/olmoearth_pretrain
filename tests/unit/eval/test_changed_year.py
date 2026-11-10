"""Tests for the PASTIS changed-year test metrics."""

from pathlib import Path

import numpy as np
import pytest
import torch

from olmoearth_pretrain.evals.changed_year import (
    changed_year_metrics,
    scorer_for_dataset,
)

S, WS = 4, 2


def _tiles(a: np.ndarray) -> np.ndarray:
    """Split (N, S, S) into consecutive row-major WS x WS tiles, as the tiled loader yields them."""
    t = S // WS
    return a.reshape(-1, t, WS, t, WS).transpose(0, 1, 3, 2, 4).reshape(-1, WS, WS)


@pytest.fixture
def sidecar(tmp_path: Path) -> str:
    """Three windows: 2019 reference, a train place with half its crop changed, an unchanged test place."""
    ref = np.full((S, S), 2, np.int8)
    ref[0, 0] = -1
    trainloc = np.full((S, S), 3, np.int8)  # 2019 was class 1 on the top half
    testloc = np.full((S, S), 4, np.int8)  # 2019 was class 4 everywhere: unchanged
    l19_train = np.full((S, S), -1, np.int8)
    l19_train[:2] = 1
    l19_train[2:] = 3
    l19_test = np.full((S, S), 4, np.int8)
    path = tmp_path / "changed_year_scoring.npz"
    np.savez(
        path,
        label=np.stack([ref, trainloc, testloc]),
        label_2019=np.stack([np.full((S, S), -1, np.int8), l19_train, l19_test]),
        bench=np.array(["ref2019", "changed_trainloc", "changed_testloc"]),
        year=np.array([2019, 2024, 2016], np.int16),
    )
    return str(path)


def test_changed_year_metrics(sidecar: str) -> None:
    """Changed pixels predicted as their 2019 class count as memorised; sets are scored apart."""
    labels = np.load(sidecar)["label"]
    preds = labels.astype(np.int64).copy()
    preds[0, 0, 0] = 0
    preds[1, 0, :] = 1  # 4 of the 8 changed train-place pixels -> 2019 class
    preds[1, 1, :2] = 0  # 2 more wrong, not the 2019 class
    # shuffle window order: matching is by label content, not position
    order = [2, 0, 1]
    out = changed_year_metrics(
        torch.from_numpy(_tiles(preds[order])),
        torch.from_numpy(_tiles(labels[order].astype(np.int64))),
        num_classes=5,
        sidecar_path=sidecar,
    )
    assert out["cy_unmatched_windows"] == 0
    assert out["cy_ref2019_miou"] == pytest.approx(1.0)
    assert out["cy_trainloc_changed_acc"] == pytest.approx(2 / 8)
    assert out["cy_trainloc_changed_pred_2019"] == pytest.approx(4 / 8)
    assert out["cy_trainloc_2024_changed_pred_2019"] == pytest.approx(4 / 8)
    assert out["cy_trainloc_unchanged_acc"] == pytest.approx(1.0)
    assert out["cy_testloc_changed_pred_2019"] == 0
    assert out["cy_testloc_unchanged_acc"] == pytest.approx(1.0)
    assert out["cy_testloc_windows"] == 1


def test_changed_year_metrics_counts_unmatched(sidecar: str) -> None:
    """Windows whose label is not in the sidecar are counted, not scored."""
    other = np.full((1, S, S), 1, np.int64)
    out = changed_year_metrics(
        torch.from_numpy(_tiles(other)),
        torch.from_numpy(_tiles(other)),
        num_classes=5,
        sidecar_path=sidecar,
    )
    assert out == {"cy_unmatched_windows": 1.0}


def test_scorer_only_for_changed_year_datasets() -> None:
    """Other datasets get no extra metrics."""
    assert scorer_for_dataset("pastis") is None
