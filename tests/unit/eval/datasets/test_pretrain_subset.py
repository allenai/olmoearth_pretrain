"""Tests for pretrain-subset eval labels."""

import torch

from olmoearth_pretrain.evals.datasets.pretrain_subset import PretrainSubsetDataset


def test_split_indices_are_disjoint_and_deterministic() -> None:
    """Train/valid/test subsets should be reproducible and non-overlapping."""
    train = PretrainSubsetDataset._select_split_indices(
        total=100,
        split="train",
        seed=7,
        train_samples=10,
        valid_samples=8,
        test_samples=6,
    )
    valid = PretrainSubsetDataset._select_split_indices(
        total=100,
        split="valid",
        seed=7,
        train_samples=10,
        valid_samples=8,
        test_samples=6,
    )
    test = PretrainSubsetDataset._select_split_indices(
        total=100,
        split="test",
        seed=7,
        train_samples=10,
        valid_samples=8,
        test_samples=6,
    )
    train_again = PretrainSubsetDataset._select_split_indices(
        total=100,
        split="train",
        seed=7,
        train_samples=10,
        valid_samples=8,
        test_samples=6,
    )

    assert train == train_again
    assert len(train) == 10
    assert len(valid) == 8
    assert len(test) == 6
    assert not (set(train) & set(valid))
    assert not (set(train) & set(test))
    assert not (set(valid) & set(test))


def test_worldcover_label_maps_class_codes() -> None:
    """WorldCover raw class codes map to contiguous labels."""
    raw = torch.tensor([[[[10], [20]], [[95], [100]]]])

    label = PretrainSubsetDataset._worldcover_label(raw)

    assert torch.equal(label, torch.tensor([[0, 1], [9, 10]]))


def test_osm_label_uses_argmax_and_ignores_empty_pixels() -> None:
    """OSM multi-channel rasters become single-class segmentation labels."""
    raw = torch.zeros(2, 2, 3)
    raw[0, 0, 1] = 1
    raw[0, 1, 2] = 1

    label = PretrainSubsetDataset._osm_label(raw)

    expected = torch.tensor([[1, 2], [-1, -1]])
    assert torch.equal(label, expected)


def test_srtm_label_is_continuous() -> None:
    """SRTM labels should stay floating-point elevations."""
    raw = torch.tensor([[[[1.5], [2.0]], [[3.25], [4.0]]]])

    label = PretrainSubsetDataset._srtm_label(raw)

    assert label.dtype == torch.float32
    assert torch.equal(label, torch.tensor([[1.5, 2.0], [3.25, 4.0]]))
