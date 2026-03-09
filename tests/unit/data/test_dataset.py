"""Unit tests for the dataset module."""

import importlib
from logging import getLogger
from pathlib import Path

import numpy as np
import pytest
import torch
from upath import UPath

from olmoearth_pretrain.data.collate import collate_olmoearth_pretrain
from olmoearth_pretrain.data.constants import MISSING_VALUE, Modality
from olmoearth_pretrain.data.dataset import (
    OlmoEarthDataset,
    OlmoEarthSample,
    get_valid_start_ts,
    subset_sample_default,
)

logger = getLogger(__name__)


def test_collate_olmoearth_pretrain(
    samples_with_missing_modalities: list[tuple[int, OlmoEarthSample]],
) -> None:
    """Test the collate_olmoearth_pretrain function."""
    collated_sample = collate_olmoearth_pretrain(
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


class TestOlmoEarthSample:
    """Test the OlmoEarthSample class."""

    def test_subset_with_missing_modalities(
        self,
        samples_with_missing_modalities: list[tuple[int, OlmoEarthSample]],
    ) -> None:
        """Test subsetting a collated sample with missing modalities."""
        sampled_hw_p = 4
        patch_size = 2
        max_tokens_per_instance = 100
        current_length = 12
        sample: OlmoEarthSample = samples_with_missing_modalities[1][1]
        subset_sample = subset_sample_default(
            sample,
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
            get_valid_start_ts(missing_timesteps, max_t, current_length)

    def test_get_valid_start_ts_with_cropped_timesteps(self) -> None:
        """Test the get_valid_start_ts function with properly cropped timesteps."""
        missing_timesteps = {
            "sentinel2_l2a": np.array([True] * 4 + [False] * 2 + [True] * 2),
            "sentinel1": np.array([True] * 4 + [False] * 2 + [True] * 1 + [False] * 1),
        }
        max_t = 2
        current_length = 8

        # This should not raise an error since timesteps are properly cropped
        start_ts = get_valid_start_ts(missing_timesteps, max_t, current_length)

        # Verify that a valid start timestamp is returned
        for t in start_ts:
            assert 0 <= t <= current_length - max_t


class TestOlmoEarthDataset:
    """Test the OlmoEarthDataset class."""

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
        dataset = OlmoEarthDataset(
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
        samples_with_missing_modalities: list[tuple[int, OlmoEarthSample]],
    ) -> None:
        """Test _fill_missing_modality function."""
        sample = samples_with_missing_modalities[0][1]
        dataset = OlmoEarthDataset(
            h5py_dir=tmp_h5py_dir,
            training_modalities=["sentinel2_l2a", "sentinel1"],
            dtype=np.float32,
            normalize=False,  # Disable normalization for testing
        )

        # Test filling a missing modality
        filled_data = dataset._fill_missing_modality(
            "sentinel1", sample.height, sample.width, sample.time
        )

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

        dataset = OlmoEarthDataset(
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
            OlmoEarthDataset._crop_timestamps_and_masks(
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
        dataset1 = OlmoEarthDataset(
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

        dataset2 = OlmoEarthDataset(
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
        dataset1 = OlmoEarthDataset(
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
        dataset2 = OlmoEarthDataset(
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


class TestComputeNDVI:
    """Test NDVI computation in OlmoEarthDataset."""

    @pytest.fixture
    def tmp_h5py_dir(self, tmp_path: Path) -> UPath:
        """Create a temporary h5py directory."""
        h5py_dir = tmp_path / "h5py_data"
        h5py_dir.mkdir()
        return UPath(h5py_dir)

    @pytest.fixture
    def dataset(self, tmp_h5py_dir: UPath) -> OlmoEarthDataset:
        """Create a dataset instance for testing."""
        return OlmoEarthDataset(
            h5py_dir=tmp_h5py_dir,
            training_modalities=["sentinel2_l2a", "ndvi"],
            dtype=np.float32,
            normalize=False,
        )

    def test_compute_ndvi_basic(self, dataset: OlmoEarthDataset) -> None:
        """Test NDVI computation with normal reflectance values."""
        h, w, t = 4, 4, 3
        num_s2_bands = Modality.SENTINEL2_L2A.num_bands  # 12
        s2_data = np.random.uniform(0.01, 1.0, (h, w, t, num_s2_bands)).astype(
            np.float32
        )
        # B04 (Red) at index 2, B08 (NIR) at index 3
        red = s2_data[..., 2]
        nir = s2_data[..., 3]
        expected_ndvi = (nir - red) / (nir + red)

        ndvi, missing_modalities = dataset._compute_ndvi(s2_data, ["ndvi"])
        assert ndvi.shape == (h, w, t, 1)
        np.testing.assert_allclose(ndvi[..., 0], expected_ndvi, atol=1e-6)
        assert "ndvi" not in missing_modalities

    def test_compute_ndvi_missing_red(self, dataset: OlmoEarthDataset) -> None:
        """Test that NDVI is MISSING_VALUE when B04 (Red) is missing."""
        h, w, t = 4, 4, 2
        num_s2_bands = Modality.SENTINEL2_L2A.num_bands
        s2_data = np.random.uniform(0.01, 1.0, (h, w, t, num_s2_bands)).astype(
            np.float32
        )
        # Set B04 to MISSING_VALUE for the first timestep
        s2_data[:, :, 0, 2] = MISSING_VALUE

        ndvi, _ = dataset._compute_ndvi(s2_data, [])
        # First timestep should be all MISSING_VALUE
        assert np.all(ndvi[:, :, 0, 0] == MISSING_VALUE)
        # Second timestep should be computed normally
        assert np.all(ndvi[:, :, 1, 0] != MISSING_VALUE)

    def test_compute_ndvi_missing_nir(self, dataset: OlmoEarthDataset) -> None:
        """Test that NDVI is MISSING_VALUE when B08 (NIR) is missing."""
        h, w, t = 4, 4, 2
        num_s2_bands = Modality.SENTINEL2_L2A.num_bands
        s2_data = np.random.uniform(0.01, 1.0, (h, w, t, num_s2_bands)).astype(
            np.float32
        )
        # Set B08 to MISSING_VALUE for all timesteps at a specific pixel
        s2_data[1, 2, :, 3] = MISSING_VALUE

        ndvi, _ = dataset._compute_ndvi(s2_data, [])
        # That pixel should be MISSING_VALUE for all timesteps
        assert np.all(ndvi[1, 2, :, 0] == MISSING_VALUE)
        # Other pixels should be computed normally
        assert ndvi[0, 0, 0, 0] != MISSING_VALUE

    def test_compute_ndvi_both_bands_missing(self, dataset: OlmoEarthDataset) -> None:
        """Test that NDVI is MISSING_VALUE when both B04 and B08 are missing."""
        h, w, t = 2, 2, 1
        num_s2_bands = Modality.SENTINEL2_L2A.num_bands
        s2_data = np.full((h, w, t, num_s2_bands), MISSING_VALUE, dtype=np.float32)

        ndvi, _ = dataset._compute_ndvi(s2_data, [])
        assert np.all(ndvi == MISSING_VALUE)

    def test_compute_ndvi_zero_denominator(self, dataset: OlmoEarthDataset) -> None:
        """Test NDVI is 0 when NIR + Red = 0 (both bands are zero)."""
        h, w, t = 2, 2, 1
        num_s2_bands = Modality.SENTINEL2_L2A.num_bands
        s2_data = np.zeros((h, w, t, num_s2_bands), dtype=np.float32)

        ndvi, _ = dataset._compute_ndvi(s2_data, [])
        assert np.all(ndvi == 0.0)

    def test_compute_ndvi_removes_from_missing_modalities(
        self, dataset: OlmoEarthDataset
    ) -> None:
        """Test that _compute_ndvi removes 'ndvi' from missing_modalities."""
        h, w, t = 2, 2, 1
        num_s2_bands = Modality.SENTINEL2_L2A.num_bands
        s2_data = np.random.uniform(0.01, 1.0, (h, w, t, num_s2_bands)).astype(
            np.float32
        )
        _, updated_missing = dataset._compute_ndvi(s2_data, ["ndvi", "sentinel1"])
        assert "ndvi" not in updated_missing
        assert "sentinel1" in updated_missing

    def test_compute_ndvi_values_bounded(self, dataset: OlmoEarthDataset) -> None:
        """Test that computed NDVI values are in [-1, 1] for non-missing pixels."""
        h, w, t = 8, 8, 4
        num_s2_bands = Modality.SENTINEL2_L2A.num_bands
        s2_data = np.random.uniform(0.01, 1.0, (h, w, t, num_s2_bands)).astype(
            np.float32
        )

        ndvi, _ = dataset._compute_ndvi(s2_data, [])
        assert np.all(ndvi >= -1.0)
        assert np.all(ndvi <= 1.0)


def test_helios_dataset_config_deprecation_warning(tmp_path: Path) -> None:
    """Ensure the legacy HeliosDatasetConfig emits a deprecation warning."""
    module = importlib.import_module("helios.data.dataset")

    with pytest.warns(DeprecationWarning):
        module.HeliosDatasetConfig(
            h5py_dir=str(tmp_path),
            training_modalities=[Modality.SENTINEL2_L2A.name],
        )
