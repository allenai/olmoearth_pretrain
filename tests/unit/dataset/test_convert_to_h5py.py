"""Unit tests for convert_to_h5py module."""

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest
from upath import UPath

from olmoearth_pretrain.data.constants import Modality, ModalitySpec
from olmoearth_pretrain.dataset.convert_to_h5py import ConvertToH5py


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


def _h5_contents(h5py_dir: UPath) -> dict[str, dict[str, np.ndarray]]:
    """Every dataset of every sample_*.h5 file under h5py_dir, keyed by file name."""
    contents: dict[str, dict[str, np.ndarray]] = {}
    for path in sorted(h5py_dir.glob("sample_*.h5")):
        with h5py.File(path, "r") as h5file:
            items: dict[str, np.ndarray] = {}
            h5file.visititems(
                lambda name, obj: items.__setitem__(name, obj[()])
                if isinstance(obj, h5py.Dataset)
                else None
            )
            contents[path.name] = items
    return contents


def test_staged_build_matches_single_job_build(
    tmp_path: Path,
    prepare_samples_and_supported_modalities: tuple,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Process -> prepare -> write over 2 shards writes the same h5s as stage "all"."""
    prepare_samples, supported_modalities = prepare_samples_and_supported_modalities
    samples = prepare_samples(tmp_path) * 5
    modalities = [m for m in supported_modalities if m != Modality.LATLON]
    monkeypatch.setattr(ConvertToH5py, "_get_samples", lambda self: list(samples))

    def build(folder: str, **kwargs: Any) -> ConvertToH5py:
        converter = ConvertToH5py(
            tile_path=UPath(tmp_path),
            supported_modalities=modalities,
            multiprocessed_sample_processing=False,
            multiprocessed_h5_creation=False,
            **kwargs,
        )
        converter.h5py_folder = folder
        return converter

    single = build("single")
    single.run()
    assert single.h5py_dir is not None

    work_dir = UPath(tmp_path / "work")
    for stage, shards in (("process", range(2)), ("prepare", [0]), ("write", range(2))):
        for shard_index in shards:
            build(
                "staged",
                stage=stage,
                num_shards=2,
                shard_index=shard_index,
                work_dir=work_dir,
            ).run()
    staged_dir = UPath(str(single.h5py_dir).replace("/single", "/staged"))

    single_contents = _h5_contents(single.h5py_dir)
    staged_contents = _h5_contents(staged_dir)
    assert len(single_contents) > 0
    assert single_contents.keys() == staged_contents.keys()
    for name, items in single_contents.items():
        assert items.keys() == staged_contents[name].keys()
        for key, value in items.items():
            np.testing.assert_array_equal(value, staged_contents[name][key])
    assert (single.h5py_dir / single.sample_metadata_fname).read_text() == (
        staged_dir / single.sample_metadata_fname
    ).read_text()
