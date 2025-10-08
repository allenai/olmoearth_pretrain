"""Test the OlmoEarthDataset class."""

import logging
from pathlib import Path

import numpy as np
from numpy.random import default_rng

from olmoearth_pretrain.data.constants import MISSING_VALUE, Modality
from olmoearth_pretrain.data.dataset import (
    GetItemArgs,
    OlmoEarthDataset,
    OlmoEarthSample,
)

logger = logging.getLogger(__name__)


def test_helios_dataset(
    setup_h5py_dir: Path,
) -> None:
    """Test the OlmoEarthDataset class."""
    training_modalities = [
        Modality.SENTINEL2_L2A.name,
        Modality.SENTINEL1.name,
        Modality.WORLDCOVER.name,
        Modality.OPENSTREETMAP_RASTER.name,
    ]
    dataset = OlmoEarthDataset(
        h5py_dir=setup_h5py_dir,
        dtype=np.float32,
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
    assert isinstance(item, OlmoEarthSample)
    assert item.sentinel2_l2a.shape == (256, 256, 12, 12)  # type: ignore
    assert item.sentinel1.shape == (256, 256, 12, 2)  # type: ignore
    assert item.worldcover.shape == (256, 256, 1, 1)  # type: ignore
    assert item.openstreetmap_raster.shape == (256, 256, 1, 30)  # type: ignore
    assert item.timestamps.shape == (12, 3)  # type: ignore


class TestOlmoEarthDataset:
    """Test the OlmoEarthDataset class."""

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
        dataset = OlmoEarthDataset(
            h5py_dir=setup_h5py_dir,
            dtype=np.float32,
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

    def test_subsetting_and_filling_missing_timesteps(
        self,
        setup_h5py_dir: Path,
        set_random_seeds: None,  # calls the fixture
    ) -> None:
        """Test the subsetting and filling missing timesteps method."""
        training_modalities = [
            Modality.SENTINEL2_L2A.name,
            Modality.SENTINEL1.name,
            Modality.WORLDCOVER.name,
            # Modality.OPENSTREETMAP_RASTER.name,
            # Modality.SRTM.name,
            # Modality.LANDSAT.name,
        ]
        dataset = OlmoEarthDataset(
            h5py_dir=setup_h5py_dir,
            dtype=np.float32,
            training_modalities=training_modalities,
            normalize=False,
        )
        args = GetItemArgs(idx=0, patch_size=1, sampled_hw_p=11, token_budget=1500)
        sample_dict = {}
        rng = default_rng(42)
        num_s2_timesteps = 10
        num_s1_timesteps = 4
        num_landsat_timesteps = 12
        logger.info(f"Training modalities: {dataset.training_modalities}")
        sample_present_modalities = [
            Modality.SENTINEL2_L2A.name,
            Modality.SENTINEL1.name,
        ]
        if Modality.SENTINEL2_L2A.name in sample_present_modalities:
            mock_sentinel2_l2a = rng.random(
                (256, 256, num_s2_timesteps, 12), dtype=np.float32
            )
            sample_dict["sentinel2_l2a"] = mock_sentinel2_l2a
        if Modality.SENTINEL1.name in sample_present_modalities:
            mock_sentinel1 = rng.random(
                (256, 256, num_s1_timesteps, 2), dtype=np.float32
            )
            sample_dict[Modality.SENTINEL1.name] = mock_sentinel1
        if Modality.WORLDCOVER.name in sample_present_modalities:
            mock_worldcover = rng.random((256, 256, 1, 1), dtype=np.float32)
            sample_dict["worldcover"] = mock_worldcover
        if Modality.LATLON.name in sample_present_modalities:
            mock_latlon = rng.random((2,), dtype=np.float32)
            sample_dict["latlon"] = mock_latlon
        if Modality.OPENSTREETMAP_RASTER.name in sample_present_modalities:
            mock_openstreetmap_raster = rng.random((256, 256, 1, 30), dtype=np.float32)
            sample_dict["openstreetmap_raster"] = mock_openstreetmap_raster
        if Modality.SRTM.name in sample_present_modalities:
            mock_srtm = rng.random((256, 256, 1, 1), dtype=np.float32)
            sample_dict["srtm"] = mock_srtm
        if Modality.LANDSAT.name in sample_present_modalities:
            mock_landsat = rng.random(
                (256, 256, num_landsat_timesteps, Modality.LANDSAT.num_bands),
                dtype=np.float32,
            )
            sample_dict["landsat"] = mock_landsat

        largest_num_timesteps = max(
            num_s2_timesteps, num_s1_timesteps, num_landsat_timesteps
        )
        days = rng.integers(0, 25, (largest_num_timesteps, 1))
        months = rng.integers(0, 12, (largest_num_timesteps, 1))
        years = rng.integers(2018, 2020, (largest_num_timesteps, 1))
        timestamps = np.concatenate([days, months, years], axis=1)  # shape: (12, 3)
        sample_dict["timestamps"] = timestamps
        missing_timesteps_masks = {
            "sentinel1": np.array(
                [
                    False,
                    True,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    True,
                    True,
                    True,
                ]
            ),
            "sentinel2_l2a": np.array(
                [
                    True,
                    True,
                    False,
                    True,
                    True,
                    False,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                ]
            ),
        }
        timestamps, missing_timesteps_masks = dataset._crop_timestamps_and_masks(
            sample_dict["timestamps"], missing_timesteps_masks
        )
        sample_dict["timestamps"] = timestamps
        sample_dict, current_length = dataset._pad_timestamps(sample_dict)
        # current length is not zero indexed
        # fill sample currently takes like .08 seconds which may bottleneck smaller models
        sample, missing_modalities = dataset.fill_sample_with_missing_values(
            sample_dict, missing_timesteps_masks
        )
        logger.warning(
            f"missing_timesteps_masks: {missing_timesteps_masks} missing_modalities: {missing_modalities}"
        )
        # Everything is filled to 12 here always so we never run into the too long issue before
        # We just pick the lowest that is correct and then repad to the correct length
        subset_sample = sample.subset_default(
            patch_size=args.patch_size,
            max_tokens_per_instance=args.token_budget,
            sampled_hw_p=args.sampled_hw_p,
            current_length=current_length,
            missing_timesteps_masks=missing_timesteps_masks,
        )
        data = [
            getattr(subset_sample, modality) for modality in dataset.training_modalities
        ]
        data = np.concatenate([d.flatten() for d in data])
        # want to make sure we subset to timesteps with actual data
        assert (data != MISSING_VALUE).sum() > 0  # type: ignore
