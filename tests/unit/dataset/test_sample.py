"""Unit tests for OlmoEarthSample."""

import calendar
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path

import pytest
import torch
from pyproj import Transformer

from olmoearth_pretrain.data.constants import BandSet, Modality, ModalitySpec
from olmoearth_pretrain.data.dataset import (
    OlmoEarthSample,
    _get_max_t_within_token_budget,
    subset_sample_cutmix,
    subset_sample_default,
)
from olmoearth_pretrain.dataset.parse import (
    GridTile,
    ModalityImage,
    ModalityTile,
    TimeSpan,
)
from olmoearth_pretrain.dataset.sample import SampleInformation, image_tiles_to_samples
from olmoearth_pretrain.nn.tokenization import (
    ModalityTokenization,
    TokenizationConfig,
)

CRS = "EPSG:32610"


def test_all_attrs_have_bands() -> None:
    """Test all attributes are described in attribute_to_bands."""
    for f_name in OlmoEarthSample._fields:
        _ = OlmoEarthSample.num_bands(f_name)


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


def test_image_tiles_to_samples_joins_by_example_id(
    tmp_path: Path, create_image_tiles: Callable
) -> None:
    """Centered examples at the same pixel join modalities only by their identity."""
    image_tiles = create_image_tiles(tmp_path)
    for modality_tiles in image_tiles.values():
        original = modality_tiles[TimeSpan.YEAR][0]
        modality_tiles[TimeSpan.YEAR] = [
            ModalityTile(
                grid_tile=GridTile(
                    crs=original.grid_tile.crs,
                    resolution_factor=original.grid_tile.resolution_factor,
                    col=original.grid_tile.col,
                    row=original.grid_tile.row,
                    example_id=example_id,
                ),
                images=original.images,
                center_time=original.center_time,
                band_sets=original.band_sets,
                modality=original.modality,
            )
            for example_id in ("sample_a", "sample_b")
        ]

    samples = image_tiles_to_samples(image_tiles)

    assert len(samples) == 2
    assert {sample.grid_tile.example_id for sample in samples} == {
        "sample_a",
        "sample_b",
    }
    assert all(
        sample.modalities.keys() == {Modality.SENTINEL2, Modality.SENTINEL1}
        for sample in samples
    )


def test_centered_sample_latlon_uses_absolute_pixel_coordinates() -> None:
    """Centered-window col/row are absolute pixels rather than grid-tile indices."""
    sample = SampleInformation(
        grid_tile=GridTile(
            crs=CRS,
            resolution_factor=16,
            col=50000,
            row=-450000,
            example_id="sample_a",
        ),
        time_span=TimeSpan.YEAR,
        modalities={},
    )
    latlon = sample.get_latlon(image_tile_size=128, pixel_coord_windows=True)
    transformer = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(500005, 4499995)

    assert latlon == pytest.approx([lat, lon])


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
    subsetted_sample = subset_sample_default(
        sample,
        patch_size=4,
        max_tokens_per_instance=100,
        sampled_hw_p=4,
        current_length=12,
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
    default_subsetted_sample = subset_sample_default(
        sample,
        patch_size=4,
        max_tokens_per_instance=100,
        sampled_hw_p=4,
        current_length=12,
    )
    cutmix_subsetted_sample = subset_sample_cutmix(
        sample,
        patch_size=4,
        max_tokens_per_instance=100,
        sampled_hw_p=4,
        current_length=12,
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
    subsetted_sample = subset_sample_default(
        sample,
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


def test_supervision_modalities_can_be_excluded_from_token_budget() -> None:
    """Loaded label layers do not reduce the imagery timestep budget."""
    h, w, t = 16, 16, 100
    sample = OlmoEarthSample(
        sentinel2_l2a=torch.ones((h, w, t, OlmoEarthSample.num_bands("sentinel2_l2a"))),
        open_set=torch.ones((h, w, 1, OlmoEarthSample.num_bands("open_set"))),
        open_set_regression=torch.ones(
            (h, w, 1, OlmoEarthSample.num_bands("open_set_regression"))
        ),
        timestamps=torch.ones((t, OlmoEarthSample.num_bands("timestamps"))),
    )

    subsetted_sample = subset_sample_default(
        sample,
        patch_size=4,
        max_tokens_per_instance=100,
        sampled_hw_p=4,
        current_length=12,
        token_budget_excluded_modalities={"open_set", "open_set_regression"},
    )

    assert subsetted_sample.time == 2
    assert subsetted_sample.open_set is not None
    assert subsetted_sample.open_set_regression is not None


def test_subsetting_with_tokenization_config() -> None:
    """Test subsetting respects TokenizationConfig band set overrides."""
    h, w, t = 16, 16, 100
    sample = OlmoEarthSample(
        sentinel2_l2a=torch.ones((h, w, t, OlmoEarthSample.num_bands("sentinel2_l2a"))),
        timestamps=torch.ones((t, OlmoEarthSample.num_bands("timestamps"))),
    )

    # Default: sentinel2_l2a has 3 band sets
    # tokens_per_timestep = 4*4*3 = 48, max_t = floor(100/48) = 2
    default_subsetted = subset_sample_default(
        sample,
        patch_size=4,
        max_tokens_per_instance=100,
        sampled_hw_p=4,
        current_length=12,
    )
    assert default_subsetted.time == 2

    # Override: put all S2 bands into a single token (1 band set)
    s2_bands = Modality.SENTINEL2_L2A.band_order
    config = TokenizationConfig(
        overrides={
            Modality.SENTINEL2_L2A.name: ModalityTokenization(
                band_groups=[list(s2_bands)]
            )
        }
    )
    # tokens_per_timestep = 4*4*1 = 16, max_t = floor(100/16) = 6
    config_subsetted = subset_sample_default(
        sample,
        patch_size=4,
        max_tokens_per_instance=100,
        sampled_hw_p=4,
        current_length=12,
        tokenization_config=config,
    )
    assert config_subsetted.time == 6


def test_get_max_t_within_token_budget_with_tokenization_config() -> None:
    """Test _get_max_t_within_token_budget directly with TokenizationConfig."""
    h, w, t = 16, 16, 100
    sample = OlmoEarthSample(
        sentinel2_l2a=torch.ones((h, w, t, OlmoEarthSample.num_bands("sentinel2_l2a"))),
        timestamps=torch.ones((t, OlmoEarthSample.num_bands("timestamps"))),
    )

    # Default: 3 band sets -> tokens_per_timestep = 4*4*3 = 48
    max_t_default = _get_max_t_within_token_budget(
        sample, h_w_p=4, max_tokens_per_instance=100
    )
    assert max_t_default == 2

    # Single band set override -> tokens_per_timestep = 4*4*1 = 16
    s2_bands = Modality.SENTINEL2_L2A.band_order
    config = TokenizationConfig(
        overrides={
            Modality.SENTINEL2_L2A.name: ModalityTokenization(
                band_groups=[list(s2_bands)]
            )
        }
    )
    max_t_config = _get_max_t_within_token_budget(
        sample, h_w_p=4, max_tokens_per_instance=100, tokenization_config=config
    )
    assert max_t_config == 6
