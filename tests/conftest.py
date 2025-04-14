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

from helios.data.constants import MISSING_VALUE, BandSet, Modality, ModalitySpec
from helios.data.dataset import HeliosSample
from helios.dataset.convert_to_h5py import ConvertToH5py
from helios.dataset.parse import GridTile, ModalityImage, ModalityTile, TimeSpan
from helios.dataset.sample import SampleInformation
from helios.train.masking import MaskValue

# Avoid triton imports from olmo-core during tests
sys.modules["triton"] = types.SimpleNamespace(
    runtime=types.SimpleNamespace(autotuner=object(), driver=object())  # type: ignore
)


@pytest.fixture(autouse=True)
def set_random_seeds() -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(42)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
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


@pytest.fixture
def setup_h5py_dir(
    tmp_path: Path, prepare_samples_and_supported_modalities: tuple
) -> Path:
    """Setup the h5py directory."""
    prepare_samples, supported_modalities = prepare_samples_and_supported_modalities
    prepared_samples = prepare_samples(tmp_path)
    convert_to_h5py = ConvertToH5py(
        tile_path=tmp_path,
        supported_modalities=[m for m in supported_modalities if m != Modality.LATLON],
        multiprocessed_h5_creation=False,
    )
    convert_to_h5py.prepare_h5_dataset(prepared_samples)
    supported_modalities = [
        m.name for m in supported_modalities if m != Modality.LATLON
    ]
    return tmp_path / "h5py_data" / "_".join(sorted(supported_modalities)) / "1"


@pytest.fixture
def masked_sample_dict(
    modality_band_set_len_and_total_bands: dict[str, tuple[int, int]],
) -> dict[str, torch.Tensor]:
    """Get a masked sample dictionary."""
    sentinel2_l2a_num_bands = modality_band_set_len_and_total_bands["sentinel2_l2a"][1]
    latlon_num_bands = modality_band_set_len_and_total_bands["latlon"][1]
    B, H, W, T, C = (
        4,
        4,
        4,
        2,
        sentinel2_l2a_num_bands,
    )
    # Create dummy sentinel2_l2a data: shape (B, H, W, T, C)
    sentinel2_l2a = torch.randn(B, H, W, T, C, requires_grad=True)
    # Here we assume 0 (ONLINE_ENCODER) means the token is visible.
    sentinel2_l2a_mask = torch.full(
        (B, H, W, T, C), fill_value=MaskValue.ONLINE_ENCODER.value, dtype=torch.long
    )
    # Dummy latitude-longitude data.
    latlon = torch.randn(B, latlon_num_bands, requires_grad=True)
    latlon_mask = torch.full(
        (B, latlon_num_bands), fill_value=MaskValue.DECODER.value, dtype=torch.float32
    )
    worldcover = torch.randn(B, H, W, 1, 1, requires_grad=True)
    worldcover_mask = torch.full(
        (B, H, W, 1, 1), fill_value=MaskValue.DECODER.value, dtype=torch.float32
    )
    # Generate valid timestamps:
    # - days: range 1..31,
    # - months: range 1..13,
    # - years: e.g. 2018-2019.
    days = torch.randint(0, 25, (B, T, 1), dtype=torch.long)
    months = torch.randint(0, 12, (B, T, 1), dtype=torch.long)
    years = torch.randint(2018, 2020, (B, T, 1), dtype=torch.long)
    timestamps = torch.cat([days, months, years], dim=-1)  # Shape: (B, T, 3)

    masked_sample_dict = {
        "sentinel2_l2a": sentinel2_l2a,
        "sentinel2_l2a_mask": sentinel2_l2a_mask,
        "latlon": latlon,
        "latlon_mask": latlon_mask,
        "worldcover": worldcover,
        "worldcover_mask": worldcover_mask,
        "timestamps": timestamps,
    }
    return masked_sample_dict


@pytest.fixture
def samples_with_missing_modalities() -> list[tuple[int, HeliosSample]]:
    """Samples with missing modalities."""
    s2_H, s2_W, s2_T, s2_C = 16, 16, 12, 13
    s1_H, s1_W, s1_T, s1_C = 16, 16, 12, 2
    wc_H, wc_W, wc_T, wc_C = 16, 16, 1, 10

    example_s2_data = np.random.randn(s2_H, s2_W, s2_T, s2_C)
    example_s1_data = np.random.randn(s1_H, s1_W, s1_T, s1_C)
    example_wc_data = np.random.randn(wc_H, wc_W, wc_T, wc_C)
    example_latlon_data = np.random.randn(2)
    timestamps = np.array(
        [
            [15, 7, 2023],
            [15, 8, 2023],
            [15, 9, 2023],
            [15, 10, 2023],
            [15, 11, 2023],
            [15, 11, 2023],
            [15, 1, 2024],
            [15, 2, 2024],
            [15, 3, 2024],
            [15, 4, 2024],
            [15, 5, 2024],
            [15, 6, 2024],
        ],
        dtype=np.int32,
    )
    missing_s1_data = np.random.randn(s1_H, s1_W, s1_T, s1_C)
    missing_s1_data[:] = MISSING_VALUE
    missing_wc_data = np.random.randn(wc_H, wc_W, wc_T, wc_C)
    missing_wc_data[:] = MISSING_VALUE
    example_s2_data = example_s2_data.astype(np.float32)
    example_s1_data = example_s1_data.astype(np.float32)
    example_wc_data = example_wc_data.astype(np.float32)
    example_latlon_data = example_latlon_data.astype(np.float32)
    missing_s1_data = missing_s1_data.astype(np.float32)
    missing_wc_data = missing_wc_data.astype(np.float32)

    sample1 = HeliosSample(
        sentinel2_l2a=example_s2_data,
        sentinel1=example_s1_data,
        worldcover=example_wc_data,
        latlon=example_latlon_data,
        timestamps=timestamps,
    )

    sample2 = HeliosSample(
        sentinel2_l2a=example_s2_data,
        sentinel1=missing_s1_data,
        worldcover=example_wc_data,
        latlon=example_latlon_data,
        timestamps=timestamps,
    )

    sample_3 = HeliosSample(
        sentinel2_l2a=example_s2_data,
        sentinel1=example_s1_data,
        worldcover=missing_wc_data,
        latlon=example_latlon_data,
        timestamps=timestamps,
    )

    batch = [(1, sample1), (1, sample2), (1, sample_3)]
    return batch


@pytest.fixture
def samples_without_missing_modalities(
    set_random_seeds: None,
) -> list[tuple[int, HeliosSample]]:
    """Samples without missing modalities."""
    s2_H, s2_W, s2_T, s2_C = 16, 16, 12, 13
    s1_H, s1_W, s1_T, s1_C = 16, 16, 12, 2
    wc_H, wc_W, wc_T, wc_C = 16, 16, 1, 10
    example_s2_data = np.random.randn(s2_H, s2_W, s2_T, s2_C).astype(np.float32)
    example_s1_data = np.random.randn(s1_H, s1_W, s1_T, s1_C).astype(np.float32)
    example_wc_data = np.random.randn(wc_H, wc_W, wc_T, wc_C).astype(np.float32)
    example_latlon_data = np.random.randn(2).astype(np.float32)
    timestamps = np.array(
        [
            [15, 7, 2023],
            [15, 8, 2023],
            [15, 9, 2023],
            [15, 10, 2023],
            [15, 11, 2023],
            [15, 11, 2023],
            [15, 1, 2024],
            [15, 2, 2024],
            [15, 3, 2024],
            [15, 4, 2024],
            [15, 5, 2024],
            [15, 6, 2024],
        ],
        dtype=np.int32,
    )

    sample1 = HeliosSample(
        sentinel2_l2a=example_s2_data,
        sentinel1=example_s1_data,
        worldcover=example_wc_data,
        latlon=example_latlon_data,
        timestamps=timestamps,
    )

    sample2 = HeliosSample(
        sentinel2_l2a=example_s2_data,
        sentinel1=example_s1_data,
        worldcover=example_wc_data,
        latlon=example_latlon_data,
        timestamps=timestamps,
    )

    sample_3 = HeliosSample(
        sentinel2_l2a=example_s2_data,
        sentinel1=example_s1_data,
        worldcover=example_wc_data,
        latlon=example_latlon_data,
        timestamps=timestamps,
    )

    batch = [(1, sample1), (1, sample2), (1, sample_3)]
    return batch
