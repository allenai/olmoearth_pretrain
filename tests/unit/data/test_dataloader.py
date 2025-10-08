"""Unit tests for dataloader module."""

from pathlib import Path

import numpy as np

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.data.dataloader import (
    OlmoEarthDataLoader,
    _IterableDatasetWrapper,
)
from olmoearth_pretrain.data.dataset import OlmoEarthDataset, collate_olmoearth_pretrain


def test_get_batch_item_params_iterator(tmp_path: Path, setup_h5py_dir: Path) -> None:
    """Test the _get_batch_item_params_iterator function."""
    # Setup test data
    """Test the OlmoEarthDataLoader class."""
    training_modalities = [
        Modality.SENTINEL2_L2A.name,
        Modality.SENTINEL1.name,
        Modality.WORLDCOVER.name,
        Modality.OPENSTREETMAP_RASTER.name,
    ]
    dataset = OlmoEarthDataset(
        h5py_dir=setup_h5py_dir,
        training_modalities=training_modalities,
        dtype=np.float32,
    )

    dataset.prepare()
    dataloader = OlmoEarthDataLoader(
        dataset=dataset,
        work_dir=tmp_path,
        global_batch_size=1,
        dp_world_size=1,
        dp_rank=0,
        fs_local_rank=0,
        seed=0,
        shuffle=True,
        num_workers=0,
        collator=collate_olmoearth_pretrain,
        target_device_type="cpu",
        token_budget=1000000,
        min_patch_size=1,
        max_patch_size=1,
        sampled_hw_p_list=[256],
    )

    dataloader.reshuffle()

    indices = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    patch_size = [1, 2, 3]
    sampled_hw_p = [4, 5, 6]
    rank_batch_size = 3

    # Set a fixed random seed for reproducibility
    dw = _IterableDatasetWrapper(dataloader)

    # Get iterator
    iterator = dw._get_batch_item_params_iterator(
        indices, patch_size, sampled_hw_p, rank_batch_size
    )

    # First batch (should all have the same patch_size and sampled_hw_p)
    first_batch = [next(iterator) for _ in range(3)]

    # Check that all items in first batch have the same patch_size and sampled_hw_p
    first_patch_size = first_batch[0][1]
    first_sampled_hw_p = first_batch[0][2]
    assert all(item[1] == first_patch_size for item in first_batch)
    assert all(item[2] == first_sampled_hw_p for item in first_batch)

    # Second batch (should have different patch_size and sampled_hw_p)
    second_batch = [next(iterator) for _ in range(3)]

    # Check that all items in second batch have the same patch_size and sampled_hw_p
    second_patch_size = second_batch[0][1]
    second_sampled_hw_p = second_batch[0][2]
    assert all(item[1] == second_patch_size for item in second_batch)
    assert all(item[2] == second_sampled_hw_p for item in second_batch)

    # Check that the patch_size or sampled_hw_p changed between batches
    assert (first_patch_size != second_patch_size) or (
        first_sampled_hw_p != second_sampled_hw_p
    )

    # Test that all indices are yielded
    remaining = list(iterator)
    assert len(remaining) == 4  # remaining 4 indices

    # Test that the indices are correct
    all_indices = [item[0] for item in first_batch + second_batch + remaining]
    assert all_indices == list(indices)

    # Test that the third batch has consistent parameters
    if len(remaining) >= 3:
        third_batch = remaining[:3]
        third_patch_size = third_batch[0][1]
        third_sampled_hw_p = third_batch[0][2]
        assert all(item[1] == third_patch_size for item in third_batch)
        assert all(item[2] == third_sampled_hw_p for item in third_batch)


def _create_test_dataloader(
    tmp_path: Path,
    seed: int = 42,
    shuffle: bool = True,
    num_dataset_repeats_per_epoch: int = 1,
    global_batch_size: int = 2,
) -> OlmoEarthDataLoader:
    """Helper function to create a test dataloader with common parameters."""

    class MockDataset(OlmoEarthDataset):
        def __init__(self, length: int) -> None:
            self.length = length

        def __len__(self) -> int:
            return self.length

    dataset = MockDataset(length=20)
    return OlmoEarthDataLoader(
        dataset=dataset,
        work_dir=tmp_path,
        global_batch_size=global_batch_size,
        dp_world_size=1,
        dp_rank=0,
        fs_local_rank=0,
        seed=seed,
        shuffle=shuffle,
        num_workers=0,
        collator=collate_olmoearth_pretrain,
        target_device_type="cpu",
        token_budget=1000000,
        min_patch_size=1,
        max_patch_size=1,
        sampled_hw_p_list=[256],
        num_dataset_repeats_per_epoch=num_dataset_repeats_per_epoch,
    )


def test_build_global_indices_same_seed_epoch_deterministic(tmp_path: Path) -> None:
    """Test that same seed and epoch produce identical indices."""
    dataloader1 = _create_test_dataloader(tmp_path, seed=42)
    dataloader1._epoch = 1
    indices1 = dataloader1._build_global_indices()

    dataloader2 = _create_test_dataloader(tmp_path, seed=42)
    dataloader2._epoch = 1
    indices2 = dataloader2._build_global_indices()

    np.testing.assert_array_equal(
        indices1, indices2, "Same seed and epoch should produce identical indices"
    )


def test_build_global_indices_different_epochs_different_shuffle(
    tmp_path: Path,
) -> None:
    """Test that different epochs produce different shuffled orders."""
    dataloader1 = _create_test_dataloader(tmp_path, seed=42)
    dataloader1._epoch = 1
    indices1 = dataloader1._build_global_indices()

    dataloader2 = _create_test_dataloader(tmp_path, seed=42)
    dataloader2._epoch = 2
    indices2 = dataloader2._build_global_indices()

    # Should have same length but different order (with high probability)
    assert len(indices1) == len(indices2), (
        "Different epochs should produce same length indices"
    )

    assert not np.array_equal(indices1, indices2), (
        "Different epochs should produce different shuffled order"
    )


def test_build_global_indices_no_shuffle_consistent(tmp_path: Path) -> None:
    """Test that no shuffle produces identical results regardless of epoch."""
    dataloader1 = _create_test_dataloader(tmp_path, seed=42, shuffle=False)
    dataloader1._epoch = 1
    indices1 = dataloader1._build_global_indices()

    dataloader2 = _create_test_dataloader(tmp_path, seed=42, shuffle=False)
    dataloader2._epoch = 5
    indices2 = dataloader2._build_global_indices()

    np.testing.assert_array_equal(
        indices1,
        indices2,
        "No shuffle should produce identical results regardless of epoch",
    )


def test_build_global_indices_different_seeds_different_results(tmp_path: Path) -> None:
    """Test that different seeds produce different results."""
    dataloader1 = _create_test_dataloader(tmp_path, seed=42)
    dataloader1._epoch = 1
    indices1 = dataloader1._build_global_indices()

    dataloader2 = _create_test_dataloader(tmp_path, seed=999)
    dataloader2._epoch = 1
    indices2 = dataloader2._build_global_indices()

    # Different seeds should produce different results (with high probability)
    assert not np.array_equal(indices1, indices2), (
        "Different seeds should produce different results"
    )


def test_build_global_indices_multiple_repeats_deterministic(tmp_path: Path) -> None:
    """Test that multiple dataset repeats are deterministic."""
    dataloader1 = _create_test_dataloader(
        tmp_path, seed=42, num_dataset_repeats_per_epoch=3
    )
    dataloader1._epoch = 1
    indices1 = dataloader1._build_global_indices()

    dataloader2 = _create_test_dataloader(
        tmp_path, seed=42, num_dataset_repeats_per_epoch=3
    )
    dataloader2._epoch = 1
    indices2 = dataloader2._build_global_indices()

    np.testing.assert_array_equal(
        indices1,
        indices2,
        "Same parameters should produce identical results with multiple repeats",
    )


def test_build_global_indices_properties_validation(tmp_path: Path) -> None:
    """Test that built indices have expected properties and constraints."""
    dataloader = _create_test_dataloader(tmp_path, seed=42)
    dataloader._epoch = 1
    indices = dataloader._build_global_indices()

    # Basic properties
    assert len(indices) > 0, "Indices should not be empty"
    assert isinstance(indices, np.ndarray), "Indices should be numpy array"
    assert indices.dtype == np.uint32, "Indices should be uint32 type"

    # All indices should be valid dataset indices
    assert np.all(indices >= 0), "All indices should be non-negative"
    assert np.all(indices < len(dataloader.dataset)), (
        "All indices should be within dataset bounds"
    )

    # Length should be divisible by global batch size (due to cropping)
    assert len(indices) % dataloader.global_batch_size == 0, (
        "Indices length should be divisible by global batch size"
    )
