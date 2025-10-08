"""Unit tests for OlmoEarthSample."""

import calendar
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path

import pytest
import torch

from olmoearth_pretrain.data.constants import BandSet, Modality, ModalitySpec
from olmoearth_pretrain.data.dataset import OlmoEarthSample
from olmoearth_pretrain.dataset.parse import (
    GridTile,
    ModalityImage,
    ModalityTile,
    TimeSpan,
)
from olmoearth_pretrain.dataset.sample import image_tiles_to_samples

CRS = "EPSG:32610"


def test_all_attrs_have_bands() -> None:
    """Test all attributes are described in attribute_to_bands."""
    for attribute_name in OlmoEarthSample._fields:
        _ = OlmoEarthSample.num_bands(attribute_name)


@pytest.fixture
def create_image_tiles(tmp_path: Path) -> Callable:
    """Create a set of fake image tiles for testing."""

    def _create_image_tiles(
        data_path: Path,
    ) -> dict[ModalitySpec, dict[TimeSpan, list[ModalityTile]]]:
        """Create image tiles for the given data path."""
        image_tiles: dict[ModalitySpec, dict[TimeSpan, list[ModalityTile]]] = {}
        images = []
        crs = "EPSG:32610"
        # Create a list of ModalityImage objects for the year 2020
        start_date = datetime(2020, 1, 1)
        while start_date.year == 2020:
            last_day = calendar.monthrange(start_date.year, start_date.month)[1]
            end_date = datetime(start_date.year, start_date.month, last_day)
            images.append(ModalityImage(start_date, end_date))
            start_date = end_date + timedelta(days=1)
        image_tiles[Modality.SENTINEL2] = {
            TimeSpan.YEAR: [
                ModalityTile(
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
                    modality=Modality.SENTINEL2,
                ),
            ]
        }
        image_tiles[Modality.SENTINEL1] = {
            TimeSpan.YEAR: [
                ModalityTile(
                    grid_tile=GridTile(
                        crs=crs, resolution_factor=16, col=165, row=-1968
                    ),
                    images=images,
                    center_time=datetime(2020, 6, 30),
                    band_sets={
                        BandSet(["VV", "VH"], 16): data_path / "s1_10m.tif",
                    },
                    modality=Modality.SENTINEL1,
                ),
            ]
        }
        return image_tiles

    return _create_image_tiles


def test_image_tiles_to_samples(tmp_path: Path, create_image_tiles: Callable) -> None:
    """Test image_tiles_to_samples with a simple case."""
    image_tiles = create_image_tiles(tmp_path)
    samples = image_tiles_to_samples(image_tiles)

    assert len(samples) == 1
    assert samples[0].grid_tile == GridTile(
        crs=CRS, resolution_factor=16, col=165, row=-1968
    )
    assert samples[0].time_span == TimeSpan.YEAR
    assert samples[0].modalities.keys() == {Modality.SENTINEL2, Modality.SENTINEL1}


def test_image_tiles_to_samples_only_sentinel2(
    tmp_path: Path, create_image_tiles: Callable
) -> None:
    """Test image_tiles_to_samples with only Sentinel-2."""
    image_tiles = create_image_tiles(tmp_path)
    supported_modalities = [Modality.SENTINEL2]
    samples = image_tiles_to_samples(image_tiles, supported_modalities)

    assert len(samples) == 1
    assert samples[0].grid_tile == GridTile(
        crs=CRS, resolution_factor=16, col=165, row=-1968
    )
    assert samples[0].time_span == TimeSpan.YEAR
    assert samples[0].modalities.keys() == {Modality.SENTINEL2}


def test_image_tiles_to_samples_only_sentinel1(
    tmp_path: Path, create_image_tiles: Callable
) -> None:
    """Test image_tiles_to_samples with only Sentinel-1."""
    image_tiles = create_image_tiles(tmp_path)
    supported_modalities = [Modality.SENTINEL1]
    samples = image_tiles_to_samples(image_tiles, supported_modalities)

    assert len(samples) == 1
    assert samples[0].grid_tile == GridTile(
        crs=CRS, resolution_factor=16, col=165, row=-1968
    )
    assert samples[0].time_span == TimeSpan.YEAR
    assert samples[0].modalities.keys() == {Modality.SENTINEL1}


def test_supporting_latlon(tmp_path: Path, create_image_tiles: Callable) -> None:
    """Test that latlon is supported."""
    image_tiles = create_image_tiles(tmp_path)
    supported_modalities = [Modality.LATLON, Modality.SENTINEL2]
    # Latlon should not change anything
    samples = image_tiles_to_samples(image_tiles, supported_modalities)
    assert len(samples) == 1
    assert samples[0].modalities.keys() == {Modality.SENTINEL2}


def test_default_subsetting() -> None:
    """Test subsetting works."""
    (
        h,
        w,
        t,
    ) = (
        16,
        16,
        100,
    )
    sample = OlmoEarthSample(
        sentinel2_l2a=torch.ones((h, w, t, OlmoEarthSample.num_bands("sentinel2_l2a"))),
        timestamps=torch.ones((t, OlmoEarthSample.num_bands("timestamps"))),
    )
    subsetted_sample = sample.subset_default(
        patch_size=4, max_tokens_per_instance=100, sampled_hw_p=4, current_length=12
    )

    # 16 / 4 = 4 tokens along the height and width dimension
    # total s2 tokens = t * 4 * 4 * 3 (band sets) = 48
    # so a token budget of floor(100 / 48) = 2
    assert subsetted_sample.time == 2


def test_cutmix_subsetting() -> None:
    """Test cutmix subsetting works."""
    (
        h,
        w,
        t,
    ) = (
        16,
        16,
        100,
    )
    sample = OlmoEarthSample(
        sentinel2_l2a=torch.ones((h, w, t, OlmoEarthSample.num_bands("sentinel2_l2a"))),
        timestamps=torch.ones((t, OlmoEarthSample.num_bands("timestamps"))),
    )
    default_subsetted_sample = sample.subset_default(
        patch_size=4, max_tokens_per_instance=100, sampled_hw_p=4, current_length=12
    )
    cutmix_subsetted_sample = sample.subset_cutmix(
        patch_size=4, max_tokens_per_instance=100, sampled_hw_p=4, current_length=12
    )
    print(default_subsetted_sample)
    print(cutmix_subsetted_sample)
    # CutMix is mixing up patches, and should has the exact shape as the default subsetting
    assert cutmix_subsetted_sample.shape(
        "sentinel2_l2a"
    ) == default_subsetted_sample.shape("sentinel2_l2a")
    assert cutmix_subsetted_sample.time == default_subsetted_sample.time


def test_subsetting_worldcover_too() -> None:
    """Test subsetting works."""
    (
        h,
        w,
        t,
    ) = (
        16,
        16,
        100,
    )
    sample = OlmoEarthSample(
        sentinel2_l2a=torch.ones((h, w, t, OlmoEarthSample.num_bands("sentinel2_l2a"))),
        worldcover=torch.ones((h, w, OlmoEarthSample.num_bands("worldcover"))),
        timestamps=torch.ones((t, OlmoEarthSample.num_bands("timestamps"))),
    )
    subsetted_sample = sample.subset_default(
        patch_size=4,
        max_tokens_per_instance=100,
        sampled_hw_p=4,
        current_length=12,
    )

    # 16 / 4 = 4 tokens along the height and width dimension
    # total s2 tokens = t * 4 * 4 * 3 (band sets) = 48
    # total worldcover tokens = 4 * 4 * 1 (band set) = 16
    # so a token budget of floor((100 - 16) / 48 = 1)

    assert subsetted_sample.time == 1
