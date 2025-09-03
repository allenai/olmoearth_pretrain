"""Unit tests for the dataset module."""

from logging import getLogger
from pathlib import Path

import numpy as np
import pytest
import torch
from upath import UPath

from helios.data.constants import MISSING_VALUE, Modality
from helios.data.dataset import HeliosDataset, HeliosSample, collate_helios

logger = getLogger(__name__)


def test_collate_helios(
    samples_with_missing_modalities: list[tuple[int, HeliosSample]],
) -> None:
    """Test the collate_helios function."""
    collated_sample = collate_helios(
        samples_with_missing_modalities,
    )

    # Check that all required fields are present
    assert collated_sample[1].sentinel2_l2a is not None
    assert collated_sample[1].sentinel1 is not None
    assert collated_sample[1].worldcover is not None
    assert collated_sample[1].latlon is not None
    assert collated_sample[1].timestamps is not None

    # Check the shapes
    assert collated_sample[1].sentinel2_l2a.shape[0] == 3
    assert collated_sample[1].sentinel1.shape[0] == 3
    assert collated_sample[1].worldcover.shape[0] == 3
    assert collated_sample[1].latlon.shape[0] == 3
    assert collated_sample[1].timestamps.shape[0] == 3

    # Check the missing modality mask values
    assert torch.all(collated_sample[1].sentinel1[1] == MISSING_VALUE)
    assert torch.all(collated_sample[1].worldcover[2] == MISSING_VALUE)


class TestHeliosSample:
    """Test the HeliosSample class."""

    def test_subset_with_missing_modalities(
        self,
        samples_with_missing_modalities: list[tuple[int, HeliosSample]],
    ) -> None:
        """Test subsetting a collated sample with missing modalities."""
        sampled_hw_p = 4
        patch_size = 2
        max_tokens_per_instance = 100
        current_length = 12
        sample: HeliosSample = samples_with_missing_modalities[1][1]
        subset_sample = sample.subset_default(
            patch_size=patch_size,
            max_tokens_per_instance=max_tokens_per_instance,
            sampled_hw_p=sampled_hw_p,
            current_length=current_length,
        )

        # Check that the shapes are correct
        assert subset_sample.sentinel2_l2a is not None
        assert subset_sample.sentinel1 is not None
        assert subset_sample.worldcover is not None

        assert subset_sample.sentinel2_l2a.shape[0] == 8
        assert subset_sample.sentinel1.shape[0] == 8
        assert subset_sample.worldcover.shape[0] == 8

        # Check that the missing modality masks are preserved
        assert (subset_sample.sentinel1[1] != MISSING_VALUE).sum() == 0

    def test_get_valid_start_ts_raises_if_timesteps_are_not_cropped(self) -> None:
        """Test the get_valid_start_ts function."""
        with pytest.raises(ValueError):
            missing_timesteps = {
                "sentinel2_l2a": np.array([False] * 6 + [True] * 6),
                "sentinel1": np.array([False] * 6 + [True] * 6),
            }
            max_t = 2
            current_length = 6
            HeliosSample._get_valid_start_ts(missing_timesteps, max_t, current_length)

    def test_get_valid_start_ts_with_cropped_timesteps(self) -> None:
        """Test the get_valid_start_ts function with properly cropped timesteps."""
        missing_timesteps = {
            "sentinel2_l2a": np.array([True] * 4 + [False] * 2 + [True] * 2),
            "sentinel1": np.array([True] * 4 + [False] * 2 + [True] * 1 + [False] * 1),
        }
        max_t = 2
        current_length = 8

        # This should not raise an error since timesteps are properly cropped
        start_ts = HeliosSample._get_valid_start_ts(
            missing_timesteps, max_t, current_length
        )

        # Verify that a valid start timestamp is returned
        for t in start_ts:
            assert 0 <= t <= current_length - max_t


class TestHeliosDataset:
    """Test the HeliosDataset class."""

    @pytest.fixture
    def tmp_h5py_dir(self, tmp_path: Path) -> UPath:
        """Create a temporary h5py directory."""
        h5py_dir = tmp_path / "h5py_data"
        h5py_dir.mkdir()
        return UPath(h5py_dir)

    def test_fill_missing_timesteps(self, tmp_h5py_dir: UPath) -> None:
        """Test _fill_missing_timesteps function."""
        # Create test data
        h, w, t, c = 4, 4, 10, 2
        data = np.random.randn(h, w, t, c).astype(np.float32)
        # Only first and last timesteps present
        mask = np.array([True, False, True])
        max_sequence_length = 5

        # Create dataset instance
        dataset = HeliosDataset(
            h5py_dir=tmp_h5py_dir,
            training_modalities=["sentinel2_l2a"],
            dtype=np.float32,
            max_sequence_length=max_sequence_length,
            normalize=False,  # Disable normalization for testing
        )

        # Fill missing timesteps
        filled_data = dataset._fill_missing_timesteps(data, mask)

        # Check shape
        assert filled_data.shape == (h, w, max_sequence_length, c)

        # Check that original data is preserved at correct timesteps
        # the data is stored without any missing data for the missing timesteps
        assert np.array_equal(filled_data[..., 0, :], data[..., 0, :])
        assert np.array_equal(filled_data[..., 2, :], data[..., 1, :])

        # Check that missing timesteps are filled with MISSING_VALUE
        assert np.all(filled_data[..., 1, :] == MISSING_VALUE)
        assert np.all(filled_data[..., 3:, :] == MISSING_VALUE)

    def test_fill_missing_modality(
        self,
        tmp_h5py_dir: UPath,
        samples_with_missing_modalities: list[tuple[int, HeliosSample]],
    ) -> None:
        """Test _fill_missing_modality function."""
        sample = samples_with_missing_modalities[0][1]
        dataset = HeliosDataset(
            h5py_dir=tmp_h5py_dir,
            training_modalities=["sentinel2_l2a", "sentinel1"],
            dtype=np.float32,
            normalize=False,  # Disable normalization for testing
        )

        # Test filling a missing modality
        filled_data = dataset._fill_missing_modality(sample, "sentinel1")

        # Check shape matches expected shape
        expected_shape = sample.get_expected_shape("sentinel1")
        assert filled_data.shape == expected_shape

        # Check all values are MISSING_VALUE
        assert np.all(filled_data == MISSING_VALUE)

    def test_fill_sample_with_missing_values(self, tmp_h5py_dir: UPath) -> None:
        """Test fill_sample_with_missing_values function."""
        # Create test data
        h, w, t, c = 4, 4, 3, 2
        data = np.random.randn(h, w, t, c).astype(np.float32)
        max_sequence_length = 5

        # Create timestamps
        timestamps = np.array([[1, 1, 2023], [2, 1, 2023], [3, 1, 2023]])

        sample_dict = {
            "sentinel2_l2a": data,
            "timestamps": timestamps,
        }
        missing_timesteps_masks = {"sentinel2_l2a": np.array([True, False, True])}

        dataset = HeliosDataset(
            h5py_dir=tmp_h5py_dir,
            training_modalities=["sentinel2_l2a", "sentinel1"],
            dtype=np.float32,
            max_sequence_length=max_sequence_length,
            normalize=False,  # Disable normalization for testing
        )

        # Pad timestamps
        sample_dict, current_length = dataset._pad_timestamps(sample_dict)

        # Fill missing values
        sample, missing_modalities = dataset.fill_sample_with_missing_values(
            sample_dict, missing_timesteps_masks
        )

        # Check that sentinel1 is in missing_modalities
        assert "sentinel1" in missing_modalities

        # Check if timestamps are padded correctly
        assert sample.time == max_sequence_length

        # Check that sentinel2_l2a has been filled correctly
        assert sample.sentinel2_l2a is not None
        expected_shape = (h, w, dataset.max_sequence_length, c)
        assert sample.sentinel2_l2a.shape == expected_shape

        # Check that missing timesteps are filled with MISSING_VALUE
        assert np.all(sample.sentinel2_l2a[..., 1, :] == MISSING_VALUE)
        assert np.all(sample.sentinel2_l2a[..., 3:, :] == MISSING_VALUE)

    def test_crop_timestamps(self) -> None:
        """Test _crop_timestamps function."""
        timestamps = np.array([[i, 1, 2023] for i in range(10)])

        missing_timesteps_masks = {
            Modality.SENTINEL2_L2A.name: np.array([True] * 8 + [False] * 2),
            Modality.LANDSAT.name: np.array([False] * 3 + [True] * 6),
        }
        cropped_timestamps, cropped_missing_timesteps_masks = (
            HeliosDataset._crop_timestamps_and_masks(
                timestamps, missing_timesteps_masks
            )
        )
        assert len(cropped_timestamps) == 9
        assert len(cropped_missing_timesteps_masks[Modality.SENTINEL2_L2A.name]) == 9
        assert len(cropped_missing_timesteps_masks[Modality.LANDSAT.name]) == 9

    def test_prepare_and_filter_sample_indices_by_dataset_percentage(
        self, setup_h5py_dir_20_samples: UPath
    ) -> None:
        """Test the prepare and filter sample indices by dataset percentage."""
        same_seed = 42
        dataset1 = HeliosDataset(
            h5py_dir=setup_h5py_dir_20_samples,
            training_modalities=["sentinel2_l2a", "sentinel1"],
            dtype=np.float32,
            normalize=False,
            dataset_percentage=0.5,
            seed=same_seed,
        )
        dataset1.prepare()
        assert len(dataset1) == 10
        assert dataset1.sample_indices is not None
        assert len(dataset1.sample_indices) == 10

        dataset2 = HeliosDataset(
            h5py_dir=setup_h5py_dir_20_samples,
            training_modalities=["sentinel2_l2a", "sentinel1"],
            dtype=np.float32,
            normalize=False,
            dataset_percentage=0.5,
            seed=same_seed,
        )
        dataset2.prepare()
        assert len(dataset2) == 10
        assert dataset2.sample_indices is not None
        assert len(dataset2.sample_indices) == 10

        assert np.array_equal(dataset1.sample_indices, dataset2.sample_indices)

    def test_prepare_and_filter_sample_indices_by_dataset_percentage_different_seed(
        self, setup_h5py_dir_20_samples: UPath
    ) -> None:
        """Test the prepare and filter sample indices by dataset percentage with different seed."""
        seed1 = 43
        dataset1 = HeliosDataset(
            h5py_dir=setup_h5py_dir_20_samples,
            training_modalities=["sentinel2_l2a", "sentinel1"],
            dtype=np.float32,
            normalize=False,
            dataset_percentage=0.7,
            seed=seed1,
        )
        dataset1.prepare()
        assert len(dataset1) == 14
        assert dataset1.sample_indices is not None
        assert len(dataset1.sample_indices) == 14

        seed2 = 44
        dataset2 = HeliosDataset(
            h5py_dir=setup_h5py_dir_20_samples,
            training_modalities=["sentinel2_l2a", "sentinel1"],
            dtype=np.float32,
            normalize=False,
            dataset_percentage=0.7,
            seed=seed2,
        )
        dataset2.prepare()
        assert len(dataset2) == 14
        assert dataset2.sample_indices is not None
        assert len(dataset2.sample_indices) == 14

        assert not np.array_equal(dataset1.sample_indices, dataset2.sample_indices)
