"""Test the HeliosDataset class."""

import calendar
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from helios.data.constants import BandSet, Modality
from helios.data.dataset import HeliosDataset, HeliosSample
from helios.dataset.parse import GridTile, ModalityImage, ModalityTile, TimeSpan
from helios.dataset.sample import SampleInformation


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


def prepare_dataset(data_path: Path) -> HeliosDataset:
    """Prepare the dataset."""
    # Create three S2 tiles corresponding to its bandsets & resolutions
    crs = "EPSG:32610"
    create_geotiff(data_path / "s2_10m.tif", 256, 256, 10, crs, 4 * 12)
    create_geotiff(data_path / "s2_20m.tif", 128, 128, 20, crs, 6 * 12)
    create_geotiff(data_path / "s2_40m.tif", 64, 64, 40, crs, 3 * 12)

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
                Modality.S2: ModalityTile(
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
                )
            },
        )
    ]
    dataset = HeliosDataset(*samples, path=data_path)
    return dataset


def test_helios_dataset(tmp_path: Path) -> None:
    """Test the HeliosDataset class."""
    dataset = prepare_dataset(tmp_path)
    dataset.prepare()

    assert len(dataset) == 1
    assert isinstance(dataset[0], HeliosSample)
    assert dataset[0].s2.shape == (13, 12, 256, 256)  # type: ignore
    assert dataset[0].latlon.shape == (2,)  # type: ignore
    assert dataset[0].timestamps.shape == (12, 3)  # type: ignore
