"""Test the HeliosDataset class."""

import calendar
import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from helios.data.constants import BandSet, Modality
from helios.data.dataset import HeliosDataset, HeliosSample
from helios.dataset.parse import GridTile, ModalityImage, ModalityTile, TimeSpan
from helios.dataset.sample import SampleInformation

logger = logging.getLogger(__name__)


def create_geotiff(
    file_path: Path,
    width: int,
    height: int,
    resolution: float,
    crs: str,
    num_bands: int,
) -> None:
    """Create a GeoTIFF file with specified resolution and size."""
    transform = from_origin(0, 0, resolution, resolution)
    data = np.random.randint(0, 255, (num_bands, height, width), dtype=np.uint8)
    with rasterio.open(
        file_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=num_bands,
        dtype=np.uint8,
        crs=crs,
        transform=transform,
    ) as dst:
        for band in range(1, num_bands + 1):
            dst.write(data[band - 1], band)


def prepare_samples(data_path: Path) -> list[SampleInformation]:
    """Prepare the dataset."""
    # Create three S2 tiles corresponding to its bandsets & resolutions
    crs = "EPSG:32610"
    sentinel2_10m_path = data_path / "s2_10m.tif"
    sentinel2_20m_path = data_path / "s2_20m.tif"
    sentinel2_40m_path = data_path / "s2_40m.tif"
    sentinel1_10m_path = data_path / "s1_10m.tif"
    worldcover_path = data_path / "worldcover.tif"
    create_geotiff(sentinel2_10m_path, 256, 256, 10, crs, 4 * 12)
    create_geotiff(sentinel2_20m_path, 128, 128, 20, crs, 6 * 12)
    create_geotiff(sentinel2_40m_path, 64, 64, 40, crs, 3 * 12)
    # Create one S1 tile
    create_geotiff(sentinel1_10m_path, 256, 256, 10, crs, 2 * 12)
    # Create one WorldCover tile
    create_geotiff(worldcover_path, 256, 256, 10, crs, 1 * 1)

    images = []
    # Create a list of ModalityImage objects for the year 2020
    start_date = datetime(2020, 1, 1)
    while start_date.year == 2020:
        last_day = calendar.monthrange(start_date.year, start_date.month)[1]
        end_date = datetime(start_date.year, start_date.month, last_day)
        images.append(ModalityImage(start_date, end_date))
        start_date = end_date + timedelta(days=1)

    samples = [
        SampleInformation(
            grid_tile=GridTile(crs=crs, resolution_factor=16, col=165, row=-1968),
            time_span=TimeSpan.YEAR,
            modalities={
                Modality.SENTINEL2: ModalityTile(
                    grid_tile=GridTile(
                        crs=crs, resolution_factor=16, col=165, row=-1968
                    ),
                    images=images,
                    center_time=datetime(2020, 6, 30),
                    band_sets={
                        BandSet(["B02", "B03", "B04", "B08"], 16): data_path
                        / "s2_10m.tif",
                        BandSet(
                            ["B05", "B06", "B07", "B8A", "B11", "B12"], 32
                        ): data_path / "s2_20m.tif",
                        BandSet(["B01", "B09", "B10"], 64): data_path / "s2_40m.tif",
                    },
                ),
                Modality.SENTINEL1: ModalityTile(
                    grid_tile=GridTile(
                        crs=crs, resolution_factor=16, col=165, row=-1968
                    ),
                    images=images,
                    center_time=datetime(2020, 6, 30),
                    band_sets={
                        BandSet(["VV", "VH"], 16): data_path / "s1_10m.tif",
                    },
                ),
                Modality.WORLDCOVER: ModalityTile(
                    grid_tile=GridTile(
                        crs=crs, resolution_factor=16, col=165, row=-1968
                    ),
                    images=images,
                    center_time=datetime(2020, 6, 30),
                    band_sets={BandSet(["B1"], 16): data_path / "worldcover.tif"},
                ),
                Modality.LATLON: ModalityTile(
                    grid_tile=GridTile(
                        crs=crs, resolution_factor=16, col=165, row=-1968
                    ),
                    images=[],
                    center_time=datetime(2020, 6, 30),
                    band_sets={BandSet(["lat", "lon"], 16): data_path / "latlon.tif"},
                ),
            },
        )
    ]
    logger.info(f"num samples: {len(samples)}")
    return samples


def test_helios_dataset(tmp_path: Path) -> None:
    """Test the HeliosDataset class."""
    samples = prepare_samples(tmp_path)
    dataset = HeliosDataset(*samples, path=tmp_path)
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

    def test_load_sample_correct_band_order(self, tmp_path: Path) -> None:
        """Test the load_sample method."""
        samples = prepare_samples(tmp_path)
        sample: SampleInformation = samples[0]
        sample_modality: ModalityTile = sample.modalities[Modality.SENTINEL2]
        image = HeliosDataset.load_sample(sample_modality, sample, dtype=np.uint16)
        sentinel2_bandset_indices = Modality.SENTINEL2.bandsets_as_indices()
        # checking that sample data is loaded in the order corresponding to the bandset indices
        # These are manually extracted values from each band
        expected_values = [
            [171, 147, 241, 226],
            [246, 79, 39, 18, 74, 129],
            [103, 33, 68],
        ]
        for bandset_index, expected_value_lst in zip(
            sentinel2_bandset_indices, expected_values
        ):
            loaded_data = image[..., bandset_index]
            for idx in range(len(expected_value_lst)):
                assert loaded_data[0, 0, 0, idx] == expected_value_lst[idx]

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
