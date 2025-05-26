"""Unit tests for convert_to_h5py module."""

import numpy as np
import pytest
from upath import UPath

from helios.data.constants import Modality, ModalitySpec
from helios.dataset.convert_to_h5py import ConvertToH5py


@pytest.fixture
def sample_timestamps_dict() -> dict[ModalitySpec, np.ndarray]:
    """Create a sample timestamps dictionary for testing."""
    return {
        Modality.SENTINEL1: np.array([[1, 1, 2020], [2, 1, 2020], [3, 1, 2020]]),
        Modality.SENTINEL2: np.array([[1, 1, 2020], [2, 1, 2020]]),
        Modality.LANDSAT: np.array([[1, 1, 2020]]),
    }


def test_find_longest_timestamps_array(
    sample_timestamps_dict: dict[ModalitySpec, np.ndarray],
) -> None:
    """Test finding the longest timestamps array."""
    converter = ConvertToH5py(
        tile_path=UPath("dummy_path"),
        supported_modalities=list(sample_timestamps_dict.keys()),
    )

    longest_array = converter._find_longest_timestamps_array(sample_timestamps_dict)
    assert len(longest_array) == 3
    assert np.array_equal(longest_array, sample_timestamps_dict[Modality.SENTINEL1])


def test_find_longest_timestamps_array_equal_length() -> None:
    """Test finding longest timestamps array with equal length arrays."""
    timestamps_dict: dict[ModalitySpec, np.ndarray] = {
        Modality.SENTINEL1: np.array([[1, 1, 2020], [2, 1, 2020]]),
        Modality.SENTINEL2: np.array([[1, 1, 2020], [2, 1, 2020]]),
    }

    converter = ConvertToH5py(
        tile_path=UPath("dummy_path"),
        supported_modalities=list(timestamps_dict.keys()),
    )

    longest_array = converter._find_longest_timestamps_array(timestamps_dict)
    assert len(longest_array) == 2
    # Should return the first one when lengths are equal
    assert np.array_equal(longest_array, timestamps_dict[Modality.SENTINEL1])


def test_create_missing_timesteps_masks(
    sample_timestamps_dict: dict[ModalitySpec, np.ndarray],
) -> None:
    """Test creating missing timesteps masks."""
    converter = ConvertToH5py(
        tile_path=UPath("dummy_path"),
        supported_modalities=list(sample_timestamps_dict.keys()),
    )

    longest_array = sample_timestamps_dict[Modality.SENTINEL1]
    masks = converter._create_missing_timesteps_masks(
        sample_timestamps_dict, longest_array
    )

    # Check masks for each modality
    assert masks[Modality.SENTINEL1.name].all()  # All timestamps present
    assert masks[Modality.SENTINEL2.name].sum() == 2  # First two present
    assert masks[Modality.LANDSAT.name].sum() == 1  # Only first present


def test_create_missing_timesteps_masks_all_match() -> None:
    """Test creating masks when all timestamps match."""
    timestamps_dict: dict[ModalitySpec, np.ndarray] = {
        Modality.SENTINEL1: np.array([[1, 1, 2020], [2, 1, 2020]]),
        Modality.SENTINEL2: np.array([[1, 1, 2020], [2, 1, 2020]]),
    }

    converter = ConvertToH5py(
        tile_path=UPath("dummy_path"),
        supported_modalities=list(timestamps_dict.keys()),
    )

    longest_array = timestamps_dict[Modality.SENTINEL1]
    masks = converter._create_missing_timesteps_masks(timestamps_dict, longest_array)

    assert masks[Modality.SENTINEL1.name].all()  # All timestamps present
    assert masks[Modality.SENTINEL2.name].all()  # All timestamps present
