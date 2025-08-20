"""Unit tests for dataloader module."""

from pathlib import Path

import numpy as np

from helios.data.constants import Modality
from helios.data.dataloader import HeliosDataLoader, _IterableDatasetWrapper
from helios.data.dataset import HeliosDataset, collate_helios


def test_get_batch_item_params_iterator(tmp_path: Path, setup_h5py_dir: Path) -> None:
    """Test the _get_batch_item_params_iterator function."""
    # Setup test data
    """Test the HeliosDataloader class."""
    training_modalities = [
        Modality.SENTINEL2_L2A.name,
        Modality.SENTINEL1.name,
        Modality.WORLDCOVER.name,
        Modality.OPENSTREETMAP_RASTER.name,
    ]
    dataset = HeliosDataset(
        h5py_dir=setup_h5py_dir,
        training_modalities=training_modalities,
        dtype=np.float32,
    )

    dataset.prepare()
    dataloader = HeliosDataLoader(
        dataset=dataset,
        work_dir=tmp_path,
        global_batch_size=1,
        dp_world_size=1,
        dp_rank=0,
        fs_local_rank=0,
        seed=0,
        shuffle=True,
        num_workers=0,
        collator=collate_helios,
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
