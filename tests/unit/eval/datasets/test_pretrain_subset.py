"""Tests for pretrain-subset eval labels."""

import numpy as np
import torch

from olmoearth_pretrain.evals.datasets.pretrain_subset import PretrainSubsetDataset


def test_split_indices_are_disjoint_and_deterministic() -> None:
    """A single shuffled population is sliced by configured per-split counts."""
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

    shuffled = np.random.RandomState(7).permutation(100).tolist()
    assert train == shuffled[:10]
    assert valid == shuffled[10:18]
    assert test == shuffled[18:24]


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


def test_geographic_split_is_deterministic() -> None:
    """Geographic splits should not depend on Python's randomized hash seed."""
    latlons = torch.stack(
        [
            torch.linspace(-40, 40, 100),
            torch.linspace(-120, 120, 100),
        ],
        dim=1,
    ).numpy()
    candidate_positions = torch.arange(100).numpy()

    split = PretrainSubsetDataset._geographic_split_positions(
        latlons=latlons,
        candidate_positions=candidate_positions,
        split="train",
        seed=7,
        train_samples=10,
        valid_samples=8,
        test_samples=6,
    )
    split_again = PretrainSubsetDataset._geographic_split_positions(
        latlons=latlons,
        candidate_positions=candidate_positions,
        split="train",
        seed=7,
        train_samples=10,
        valid_samples=8,
        test_samples=6,
    )

    assert split.tolist() == split_again.tolist()


def test_geographic_splits_are_disjoint() -> None:
    """Geographic train/valid/test splits should assign bins once, without overlap."""
    latlons = torch.stack(
        [
            torch.linspace(-40, 40, 100),
            torch.linspace(-120, 120, 100),
        ],
        dim=1,
    ).numpy()
    candidate_positions = torch.arange(100).numpy()

    kwargs = dict(
        latlons=latlons,
        candidate_positions=candidate_positions,
        seed=7,
        train_samples=40,
        valid_samples=10,
        test_samples=10,
    )
    train = PretrainSubsetDataset._geographic_split_positions(split="train", **kwargs)
    valid = PretrainSubsetDataset._geographic_split_positions(split="valid", **kwargs)
    test = PretrainSubsetDataset._geographic_split_positions(split="test", **kwargs)

    assert not (set(train.tolist()) & set(valid.tolist()))
    assert not (set(train.tolist()) & set(test.tolist()))
    assert not (set(valid.tolist()) & set(test.tolist()))
