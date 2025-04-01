"""Conftest for the tests."""

import calendar
import random
import sys
import types
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
import rasterio
import torch
from rasterio.transform import from_origin

from helios.data.constants import BandSet, Modality, ModalitySpec
from helios.dataset.parse import GridTile, ModalityImage, ModalityTile, TimeSpan
from helios.dataset.sample import SampleInformation

# Avoid triton imports from olmo-core during tests
sys.modules["triton"] = types.SimpleNamespace(
    runtime=types.SimpleNamespace(autotuner=object(), driver=object())  # type: ignore
)


@pytest.fixture(scope="session", autouse=True)
def set_random_seeds() -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)


# TODO: not sure this needs to be shared across tests
@pytest.fixture
def supported_modalities() -> list[ModalitySpec]:
    """Create a list of supported modalities for testing."""
    return [Modality.SENTINEL2_L2A, Modality.LATLON]


# TODO: add some create mock data factory functions for all the contracts and different steps
def create_geotiff(
    file_path: Path,
    width: int,
    height: int,
    resolution: float,
    crs: str,
    num_bands: int,
) -> None:
    """Create a GeoTIFF file with specified resolution and size."""
    generator = np.random.default_rng(42)
    transform = from_origin(0, 0, resolution, resolution)
    data = generator.integers(0, 255, (num_bands, height, width), dtype=np.uint8)
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


@pytest.fixture
def prepare_samples_and_supported_modalities() -> (
    tuple[Callable[[Path], list[SampleInformation]], list[ModalitySpec]]
):
    """Function to create samples in a directory.

    and also returns what modalities are supported in these samples
    """

    def prepare_samples_func(data_path: Path) -> list[SampleInformation]:
        """Prepare the dataset."""
        # Create three S2 tiles corresponding to its bandsets & resolutions
        crs = "EPSG:32610"
        sentinel2_l2a_10m_path = data_path / "s2_l2a_10m.tif"
        sentinel2_l2a_20m_path = data_path / "s2_l2a_20m.tif"
        sentinel2_l2a_40m_path = data_path / "s2_l2a_40m.tif"
        sentinel1_10m_path = data_path / "s1_10m.tif"
        worldcover_path = data_path / "worldcover.tif"
        openstreetmap_path = data_path / "openstreetmap.tif"
        create_geotiff(sentinel2_l2a_10m_path, 256, 256, 10, crs, 4 * 12)
        create_geotiff(sentinel2_l2a_20m_path, 128, 128, 20, crs, 6 * 12)
        create_geotiff(sentinel2_l2a_40m_path, 64, 64, 40, crs, 2 * 12)
        # Create one S1 tile
        create_geotiff(sentinel1_10m_path, 256, 256, 10, crs, 2 * 12)
        # Create one WorldCover tile
        create_geotiff(worldcover_path, 256, 256, 10, crs, 1 * 1)
        # Create one OpenStreetMap tile
        create_geotiff(openstreetmap_path, 1024, 1024, 2.5, crs, 1 * 30)

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
                    Modality.SENTINEL2_L2A: ModalityTile(
                        grid_tile=GridTile(
                            crs=crs, resolution_factor=16, col=165, row=-1968
                        ),
                        images=images,
                        center_time=datetime(2020, 6, 30),
                        band_sets={
                            BandSet(["B02", "B03", "B04", "B08"], 16): data_path
                            / "s2_l2a_10m.tif",
                            BandSet(
                                ["B05", "B06", "B07", "B8A", "B11", "B12"], 32
                            ): data_path / "s2_l2a_20m.tif",
                            BandSet(["B01", "B09"], 64): data_path / "s2_l2a_40m.tif",
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
                    Modality.OPENSTREETMAP_RASTER: ModalityTile(
                        grid_tile=GridTile(
                            crs=crs, resolution_factor=16, col=165, row=-1968
                        ),
                        images=images,
                        center_time=datetime(2020, 6, 30),
                        band_sets={
                            BandSet(
                                [
                                    "aerialway_pylon",
                                    "aerodrome",
                                    "airstrip",
                                    "amenity_fuel",
                                    "building",
                                    "chimney",
                                    "communications_tower",
                                    "crane",
                                    "flagpole",
                                    "fountain",
                                    "generator_wind",
                                    "helipad",
                                    "highway",
                                    "leisure",
                                    "lighthouse",
                                    "obelisk",
                                    "observatory",
                                    "parking",
                                    "petroleum_well",
                                    "power_plant",
                                    "power_substation",
                                    "power_tower",
                                    "river",
                                    "runway",
                                    "satellite_dish",
                                    "silo",
                                    "storage_tank",
                                    "taxiway",
                                    "water_tower",
                                    "works",
                                ],
                                4,
                            ): data_path / "openstreetmap.tif",
                        },
                    ),
                },
            )
        ]
        return samples

    return (
        prepare_samples_func,
        [
            Modality.SENTINEL2_L2A,
            Modality.SENTINEL1,
            Modality.WORLDCOVER,
            Modality.LATLON,  # We want to include latlon even though it is not a read in modality
            Modality.OPENSTREETMAP_RASTER,
        ],
    )
