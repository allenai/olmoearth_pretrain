"""Test the HeliosDataset class."""

import logging
from pathlib import Path

from helios.data.constants import Modality
from helios.data.dataset import GetItemArgs, HeliosDataset, HeliosSample

logger = logging.getLogger(__name__)


def test_helios_dataset(
    setup_h5py_dir: Path,
) -> None:
    """Test the HeliosDataset class."""
    training_modalities = [
        Modality.SENTINEL2_L2A.name,
        Modality.SENTINEL1.name,
        Modality.WORLDCOVER.name,
        Modality.OPENSTREETMAP_RASTER.name,
    ]
    dataset = HeliosDataset(
        h5py_dir=setup_h5py_dir,
        dtype="float32",
        training_modalities=training_modalities,
    )

    dataset.prepare()

    assert len(dataset) == 1
    args = GetItemArgs(
        idx=0,
        patch_size=1,
        sampled_hw_p=256,
    )
    patch_size, item = dataset[args]
    assert patch_size == 1
    assert isinstance(item, HeliosSample)
    assert item.sentinel2_l2a.shape == (256, 256, 12, 12)  # type: ignore
    assert item.sentinel1.shape == (256, 256, 12, 2)  # type: ignore
    assert item.worldcover.shape == (256, 256, 1, 1)  # type: ignore
    assert item.openstreetmap_raster.shape == (256, 256, 1, 30)  # type: ignore
    assert item.latlon.shape == (2,)  # type: ignore
    assert item.timestamps.shape == (12, 3)  # type: ignore


class TestHeliosDataset:
    """Test the HeliosDataset class."""

    def test_load_sample_correct_band_order(
        self,
        setup_h5py_dir: Path,
        set_random_seeds: None,  # calls the fixture
    ) -> None:
        """Test the load_sample method."""
        training_modalities = [
            Modality.SENTINEL2_L2A.name,
            Modality.SENTINEL1.name,
            Modality.WORLDCOVER.name,
            Modality.OPENSTREETMAP_RASTER.name,
        ]
        dataset = HeliosDataset(
            h5py_dir=setup_h5py_dir,
            dtype="float32",
            training_modalities=training_modalities,
            normalize=False,
        )
        args = GetItemArgs(
            idx=0,
            patch_size=1,
            sampled_hw_p=256,
        )
        _, helios_sample = dataset[args]
        image = helios_sample.sentinel2_l2a
        assert image is not None
        sentinel2_bandset_indices = Modality.SENTINEL2_L2A.bandsets_as_indices()
        # checking that sample data is loaded in the order corresponding to the bandset indices
        # These are manually extracted values from each band and dependent on the seed (call with conftest.py)
        expected_values = [
            [135, 10, 36, 92],
            [135, 31, 130, 28, 10, 88],
            [135, 37],
        ]
        data_matches_expected = []
        for bandset_index, expected_value_lst in zip(
            sentinel2_bandset_indices, expected_values
        ):
            loaded_data = image[..., bandset_index]
            for idx in range(len(expected_value_lst)):
                print(loaded_data[0, 0, 0, idx])
                data_matches_expected.append(
                    loaded_data[0, 0, 0, idx] == expected_value_lst[idx]
                )
        assert all(data_matches_expected)

        # Now check that different bandset indices change the values
        fake_bandset_indices = [[1, 2, 3, 9], [4, 5, 6, 8, 10, 11], [0, 9]]
        data_matches = []
        for fake_bandset_index, expected_value_lst in zip(
            fake_bandset_indices, expected_values
        ):
            loaded_data = image[..., fake_bandset_index]
            for idx in range(len(fake_bandset_index)):
                data_matches.append(
                    loaded_data[0, 0, 0, idx] == expected_value_lst[idx]
                )
        assert not all(data_matches)
