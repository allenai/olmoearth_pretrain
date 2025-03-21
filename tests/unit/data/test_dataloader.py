"""Unit tests for dataloader module."""

import numpy as np

from helios.data.dataloader import _get_batch_item_params_iterator


def test_get_batch_item_params_iterator() -> None:
    """Test the _get_batch_item_params_iterator function."""
    # Setup test data
    indices = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    patch_size = [1, 2, 3]
    sampled_hw_p = [4, 5, 6]
    rank_batch_size = 3

    # Set a fixed random seed for reproducibility
    np.random.seed(42)

    # Get iterator
    iterator = _get_batch_item_params_iterator(
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
