"""Tests for pretrain-subset eval labels."""

from types import SimpleNamespace

import numpy as np
import torch

from olmoearth_pretrain.evals.datasets.pretrain_subset import (
    OSM_POPULOUS_12_CLASS_IDS,
    OsmLabelMode,
    PretrainSplitStrategy,
    PretrainSubsetDataset,
)
from olmoearth_pretrain.evals.metrics import SEGMENTATION_IGNORE_LABEL


def test_split_strategy_enum_values() -> None:
    """Split strategies should be explicit enum values, not ad hoc strings."""
    assert PretrainSplitStrategy.RANDOM.value == "random"
    assert PretrainSplitStrategy.GEOGRAPHIC.value == "geographic"


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


def test_positions_from_split_csv_maps_h5_indices_to_dataset_positions(tmp_path) -> None:
    """Split CSVs store H5 indices, which need mapping through sample_indices."""
    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    (split_dir / "train.csv").write_text("sample_index\n30\n10\n999\n")
    dataset = SimpleNamespace(sample_indices=np.asarray([10, 20, 30]))

    positions = PretrainSubsetDataset._positions_from_split_csv(
        dataset=dataset,
        split_dir=str(split_dir),
        split="train",
    )

    assert positions.tolist() == [2, 0]


def test_split_rows_from_split_csv_returns_matched_rows(tmp_path) -> None:
    """Split CSV loading should preserve matched rows for tile-level labels."""
    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    (split_dir / "train.csv").write_text(
        "sample_index,anchor_class_id,anchor_class_name\n"
        "30,12,highway\n"
        "10,4,building\n"
        "999,1,aerodrome\n"
    )
    dataset = SimpleNamespace(sample_indices=np.asarray([10, 20, 30]))

    positions, rows = PretrainSubsetDataset._split_rows_from_split_csv(
        dataset=dataset,
        split_dir=str(split_dir),
        split="train",
    )

    assert positions.tolist() == [2, 0]
    assert rows["anchor_class_id"].tolist() == [12, 4]


def test_split_csv_rows_can_be_capped_by_split_sample_count() -> None:
    """Split-backed probes should support small train/valid/test smoke subsets."""
    positions = np.asarray([2, 0, 4, 1])
    rows = torch.arange(4)

    target_size = PretrainSubsetDataset._target_size_for_split(
        split="train",
        train_samples=2,
        valid_samples=8,
        test_samples=6,
    )

    assert positions[:target_size].tolist() == [2, 0]
    assert rows[:target_size].tolist() == [0, 1]


def test_osm_label_mode_enum_values() -> None:
    """OSM label modes should be explicit enum values."""
    assert OsmLabelMode.SEGMENTATION.value == "segmentation"
    assert OsmLabelMode.TILE_ANCHOR_CLASS.value == "tile_anchor_class"
    assert OsmLabelMode.TILE_PRESENCE.value == "tile_presence"


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


def test_osm_label_remaps_populous_classes_and_ignores_other_classes() -> None:
    """Populous OSM eval labels should be contiguous and ignore other classes."""
    raw = torch.zeros(2, 2, 30)
    raw[0, 0, 1] = 1
    raw[0, 1, 12] = 1
    raw[1, 0, 2] = 1

    label = PretrainSubsetDataset._osm_label(raw, OSM_POPULOUS_12_CLASS_IDS)

    expected = torch.tensor(
        [
            [0, 3],
            [SEGMENTATION_IGNORE_LABEL, SEGMENTATION_IGNORE_LABEL],
        ]
    )
    assert torch.equal(label, expected)


def test_srtm_label_is_continuous() -> None:
    """SRTM labels should stay floating-point elevations."""
    raw = torch.tensor([[[[1.5], [2.0]], [[3.25], [4.0]]]])

    label = PretrainSubsetDataset._srtm_label(raw)

    assert label.dtype == torch.float32
    assert torch.equal(label, torch.tensor([[1.5, 2.0], [3.25, 4.0]]))


def test_canopy_label_is_continuous() -> None:
    """Canopy height labels should stay floating-point heights."""
    raw = torch.tensor([[[[0.0], [4.0]], [[12.5], [25.0]]]])

    label = PretrainSubsetDataset._canopy_label(raw)

    assert label.dtype == torch.float32
    assert torch.equal(label, torch.tensor([[0.0, 4.0], [12.5, 25.0]]))


def test_cdl_label_ignores_no_data() -> None:
    """CDL no-data code 0 should become the segmentation ignore label."""
    raw = torch.tensor([[[[0], [1]], [[121], [255]]]])

    label = PretrainSubsetDataset._cdl_label(raw)

    expected = torch.tensor(
        [[SEGMENTATION_IGNORE_LABEL, 1], [121, 255]],
        dtype=torch.long,
    )
    assert torch.equal(label, expected)


def test_worldcereal_label_uses_primary_channel_and_ignores_empty_pixels() -> None:
    """WorldCereal labels are binary on the primary crop channel."""
    raw = torch.zeros(2, 2, 8)
    raw[0, 0, 0] = 1
    raw[0, 1, 3] = 1

    label = PretrainSubsetDataset._worldcereal_label(raw)

    expected = torch.tensor([[1, 0], [-1, -1]])
    assert torch.equal(label, expected)


def test_multi_hot_osm_presence_from_split_labels() -> None:
    """OSM tile-presence labels should become 30-way multi-hot targets."""
    label = PretrainSubsetDataset._multi_hot_osm_presence("4 12 22")

    assert label.shape == (30,)
    assert label.dtype == torch.long
    assert label.sum().item() == 3
    assert label[4] == 1
    assert label[12] == 1
    assert label[22] == 1


def test_random_split_pool_positions_are_disjoint() -> None:
    """Random split pools should be disjoint before sample caps are applied."""
    candidate_positions = np.arange(100)

    train = PretrainSubsetDataset._random_split_pool_positions(
        candidate_positions=candidate_positions,
        split="train",
        seed=7,
    )
    valid = PretrainSubsetDataset._random_split_pool_positions(
        candidate_positions=candidate_positions,
        split="valid",
        seed=7,
    )
    test = PretrainSubsetDataset._random_split_pool_positions(
        candidate_positions=candidate_positions,
        split="test",
        seed=7,
    )

    assert len(train) == 80
    assert len(valid) == 10
    assert len(test) == 10
    assert not (set(train.tolist()) & set(valid.tolist()))
    assert not (set(train.tolist()) & set(test.tolist()))
    assert not (set(valid.tolist()) & set(test.tolist()))


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
