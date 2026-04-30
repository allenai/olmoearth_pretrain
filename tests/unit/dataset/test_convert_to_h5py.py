"""Unit tests for convert_to_h5py module."""

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from upath import UPath

from olmoearth_pretrain.data.constants import (
    HIGH_FREQ_NUM_TIMESTEPS,
    Modality,
    ModalitySpec,
    TimeSpan,
)
from olmoearth_pretrain.dataset.convert_to_h5py import ConvertToH5py
from olmoearth_pretrain.dataset.parse import GridTile, ModalityImage, ModalityTile
from olmoearth_pretrain.dataset.sample import SampleInformation


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


def _make_high_freq_sample(
    missing_indices: set[int],
    center_time: datetime,
) -> SampleInformation:
    start_time = datetime(2020, 1, 1, tzinfo=UTC)
    period = timedelta(days=5)
    images = [
        ModalityImage(
            start_time=start_time + period * idx,
            end_time=start_time + period * (idx + 1),
        )
        for idx in range(HIGH_FREQ_NUM_TIMESTEPS)
        if idx not in missing_indices
    ]
    grid_tile = GridTile(
        "EPSG:32610",
        Modality.SENTINEL2_L2A.tile_resolution_factor,
        0,
        0,
    )
    modality_tile = ModalityTile(
        grid_tile=grid_tile,
        images=images,
        center_time=center_time,
        band_sets={},
        modality=Modality.SENTINEL2_L2A,
    )
    return SampleInformation(
        grid_tile=grid_tile,
        time_span=TimeSpan.HIGH_FREQ,
        modalities={Modality.SENTINEL2_L2A: modality_tile},
    )


def test_high_freq_reference_timestamps_include_missing_periods() -> None:
    """High-frequency masks align to the full 72-period grid."""
    start_time = datetime(2020, 1, 1, tzinfo=UTC)
    period = timedelta(days=5)
    missing_indices = {0, 7, 17, 42, 71}
    sample = _make_high_freq_sample(
        missing_indices=missing_indices,
        center_time=start_time + period * HIGH_FREQ_NUM_TIMESTEPS / 2,
    )
    converter = ConvertToH5py(
        tile_path=UPath("dummy_path"),
        supported_modalities=[Modality.SENTINEL2_L2A],
    )

    timestamps_dict = sample.get_timestamps()
    reference_timestamps = converter._get_reference_timestamps_array(
        sample,
        timestamps_dict,
    )
    masks = converter._create_missing_timesteps_masks(
        timestamps_dict,
        reference_timestamps,
    )

    assert len(reference_timestamps) == HIGH_FREQ_NUM_TIMESTEPS
    assert masks[Modality.SENTINEL2_L2A.name].sum() == (
        HIGH_FREQ_NUM_TIMESTEPS - len(missing_indices)
    )
    for missing_idx in missing_indices:
        assert not masks[Modality.SENTINEL2_L2A.name][missing_idx]


def test_high_freq_reference_timestamps_support_legacy_tile_time() -> None:
    """Existing hfreq CSVs used window_start + 7 days as tile_time."""
    start_time = datetime(2020, 1, 1, tzinfo=UTC)
    missing_indices = {0, 5, 71}
    sample = _make_high_freq_sample(
        missing_indices=missing_indices,
        center_time=start_time + timedelta(days=7),
    )
    converter = ConvertToH5py(
        tile_path=UPath("dummy_path"),
        supported_modalities=[Modality.SENTINEL2_L2A],
    )

    timestamps_dict = sample.get_timestamps()
    reference_timestamps = converter._get_reference_timestamps_array(
        sample,
        timestamps_dict,
    )
    masks = converter._create_missing_timesteps_masks(
        timestamps_dict,
        reference_timestamps,
    )

    assert len(reference_timestamps) == HIGH_FREQ_NUM_TIMESTEPS
    assert not masks[Modality.SENTINEL2_L2A.name][0]
    assert not masks[Modality.SENTINEL2_L2A.name][5]
    assert not masks[Modality.SENTINEL2_L2A.name][71]
