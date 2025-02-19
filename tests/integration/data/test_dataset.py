"""Test the HeliosDataset class."""

import logging
from pathlib import Path

from helios.data.constants import Modality
from helios.data.dataset import HeliosDataset, HeliosSample
from helios.dataset.parse import ModalityTile
from helios.dataset.sample import SampleInformation

logger = logging.getLogger(__name__)


def test_helios_dataset(
    tmp_path: Path, prepare_samples_and_supported_modalities: tuple
) -> None:
    """Test the HeliosDataset class."""
    prepare_samples, supported_modalities = prepare_samples_and_supported_modalities
    samples = prepare_samples(tmp_path)
    dataset = HeliosDataset(
        samples=samples,
        tile_path=tmp_path,
        supported_modalities=supported_modalities,
    )
    dataset.prepare()

    assert len(dataset) == 1
    assert isinstance(dataset[0], HeliosSample)
    assert dataset[0].sentinel2.shape == (256, 256, 12, 13)  # type: ignore
    assert dataset[0].sentinel1.shape == (256, 256, 12, 2)  # type: ignore
    assert dataset[0].worldcover.shape == (256, 256, 1, 1)  # type: ignore
    assert dataset[0].latlon.shape == (2,)  # type: ignore
    assert dataset[0].timestamps.shape == (12, 3)  # type: ignore


class TestHeliosDataset:
    """Test the HeliosDataset class."""

    def test_load_sample_correct_band_order(
        self,
        tmp_path: Path,
        prepare_samples_and_supported_modalities: tuple,
        set_random_seeds: None,  # calls the fixture
    ) -> None:
        """Test the load_sample method."""
        prepare_samples, _ = prepare_samples_and_supported_modalities
        samples = prepare_samples(tmp_path)
        logger.info(f"samples: {len(samples)}")
        sample: SampleInformation = samples[0]
        sample_modality: ModalityTile = sample.modalities[Modality.SENTINEL2]
        image = HeliosDataset.load_sample(sample_modality, sample)
        sentinel2_bandset_indices = Modality.SENTINEL2.bandsets_as_indices()
        # checking that sample data is loaded in the order corresponding to the bandset indices
        # These are manually extracted values from each band and dependent on the seed (call with conftest.py)
        expected_values = [
            [61, 161, 95, 142],
            [176, 214, 252, 194, 68, 88],
            [183, 94, 223],
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
        fake_bandset_indices = [[1, 2, 3, 9], [4, 5, 6, 8, 11, 12], [0, 9, 10]]
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
