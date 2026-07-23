"""Unit tests for the embedding materializer (fetchers, providers, orchestration)."""

import importlib.util
import json
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine
from rasterio.crs import CRS
from rslearn.dataset import Dataset, Window
from rslearn.utils.geometry import PixelBounds, Projection
from upath import UPath

from olmoearth_pretrain.data.constants import Modality, ModalitySpec
from olmoearth_pretrain.evals.embedding_materializer.fetchers import (
    EmbeddingFetcher,
    SourceTile,
    TesseraFetcher,
    mosaic_tiles_to_bounds,
    year_time_range,
)
from olmoearth_pretrain.evals.embedding_materializer.materialize import (
    materialize_product,
    write_manifest,
)
from olmoearth_pretrain.evals.embedding_materializer.providers import (
    RslearnWindowProvider,
    get_target_year,
)

PROJECTION = Projection(CRS.from_epsg(32610), 10, -10)
WINDOW_SIZE = 16
TIME_RANGE = (
    datetime(2019, 6, 1, tzinfo=UTC),
    datetime(2020, 6, 1, tzinfo=UTC),
)


class FakeFetcher(EmbeddingFetcher):
    """Deterministic fetcher returning arange arrays, or None for gap bounds."""

    def __init__(self, gap_bounds: set[PixelBounds] | None = None) -> None:
        """Initialize a FakeFetcher.

        Args:
            gap_bounds: window bounds for which fetch returns None (no
                coverage).
        """
        self.gap_bounds = gap_bounds or set()
        self.calls: list[tuple[PixelBounds, int]] = []

    @property
    def modality(self) -> ModalitySpec:
        """The GSE modality (64 bands)."""
        return Modality.GSE

    @property
    def product_version(self) -> str:
        """Fake product version."""
        return "fake-v1"

    @property
    def nodata_value(self) -> float:
        """Fake nodata value."""
        return -1.0

    def fetch(
        self, bounds: PixelBounds, projection: Projection, year: int
    ) -> np.ndarray | None:
        """Return a deterministic (C, H, W) array, or None for gap bounds."""
        self.calls.append((tuple(bounds), year))
        if tuple(bounds) in self.gap_bounds:
            return None
        num_bands = len(self.modality.band_order)
        height = bounds[3] - bounds[1]
        width = bounds[2] - bounds[0]
        return (
            np.arange(num_bands * height * width, dtype=np.float32).reshape(
                num_bands, height, width
            )
            + year
        )


def make_dataset(tmp_path: Path) -> tuple[UPath, list[Window]]:
    """Create a tiny rslearn dataset on disk with three windows.

    Args:
        tmp_path: pytest tmp_path fixture value.

    Returns:
        tuple of (dataset path, list of windows). The third window ("w3") is
        placed at distinct bounds so tests can designate it a coverage gap.
    """
    ds_path = UPath(tmp_path) / "dataset"
    ds_path.mkdir(parents=True)
    with (ds_path / "config.json").open("w") as f:
        json.dump({"layers": {}}, f)

    dataset = Dataset(ds_path)
    windows = []
    for idx, name in enumerate(["w1", "w2", "w3"]):
        offset = idx * WINDOW_SIZE
        window = Window(
            storage=dataset.storage,
            group="default",
            name=name,
            projection=PROJECTION,
            bounds=(offset, offset, offset + WINDOW_SIZE, offset + WINDOW_SIZE),
            time_range=TIME_RANGE,
        )
        window.save()
        windows.append(window)
    return ds_path, windows


GAP_BOUNDS = (32, 32, 48, 48)  # bounds of window w3


def test_materialize_writes_layers(tmp_path: Path) -> None:
    """Layers are written as GeoTIFFs with the right bands and marked complete."""
    ds_path, windows = make_dataset(tmp_path)
    fetcher = FakeFetcher(gap_bounds={GAP_BOUNDS})
    manifest = materialize_product(ds_path, fetcher, product_name="fake")

    bands = Modality.GSE.band_order
    for window in windows[:2]:
        assert window.is_layer_completed("gse")
        raster_path = window.get_raster_dir("gse", bands) / "geotiff.tif"
        assert raster_path.exists()
        with rasterio.open(str(raster_path)) as src:
            assert src.count == len(bands)
            assert src.dtypes[0] == "float32"
            assert src.nodata == -1.0
            data = src.read()
        # Deterministic content round-trips (midpoint year of TIME_RANGE = 2019).
        expected = fetcher.fetch(window.bounds, window.projection, 2019)
        np.testing.assert_array_equal(data, expected)

    # The gap window has no layer.
    assert not windows[2].is_layer_completed("gse")
    assert manifest["num_windows_written"] == 2
    assert manifest["coverage_gaps"] == ["default/w3"]


def test_idempotent_rerun_skips_existing(tmp_path: Path) -> None:
    """A re-run without overwrite skips windows whose layer already exists."""
    ds_path, _ = make_dataset(tmp_path)
    fetcher = FakeFetcher()
    materialize_product(ds_path, fetcher, product_name="fake")
    assert len(fetcher.calls) == 3

    rerun_fetcher = FakeFetcher()
    manifest = materialize_product(ds_path, rerun_fetcher, product_name="fake")
    assert len(rerun_fetcher.calls) == 0
    assert manifest["num_windows_written"] == 0
    assert manifest["num_windows_skipped_existing"] == 3


def test_overwrite_rewrites(tmp_path: Path) -> None:
    """With overwrite=True, existing layers are fetched and written again."""
    ds_path, _ = make_dataset(tmp_path)
    materialize_product(ds_path, FakeFetcher(), product_name="fake")

    # workers=2 also exercises the threaded fetch/write path.
    fetcher = FakeFetcher()
    manifest = materialize_product(
        ds_path, fetcher, product_name="fake", overwrite=True, workers=2
    )
    assert len(fetcher.calls) == 3
    assert manifest["num_windows_written"] == 3
    assert manifest["num_windows_skipped_existing"] == 0


def test_year_policy(tmp_path: Path) -> None:
    """The fixed --year overrides the per-window time-range midpoint year."""
    ds_path, _ = make_dataset(tmp_path)

    fetcher = FakeFetcher()
    manifest = materialize_product(ds_path, fetcher, product_name="fake")
    # Midpoint of 2019-06-01..2020-06-01 is 2019-12-01.
    assert all(year == 2019 for _, year in fetcher.calls)
    assert manifest["year_policy"] == "window_time_range_midpoint"

    fetcher = FakeFetcher()
    manifest = materialize_product(
        ds_path, fetcher, product_name="fake", year=2021, overwrite=True
    )
    assert all(year == 2021 for _, year in fetcher.calls)
    assert manifest["year_policy"] == "fixed:2021"


def test_get_target_year_no_time_range() -> None:
    """Windows without a time range yield None unless a year override is set."""
    window = Window(
        storage=None,
        group="default",
        name="no_time",
        projection=PROJECTION,
        bounds=(0, 0, WINDOW_SIZE, WINDOW_SIZE),
        time_range=None,
    )
    assert get_target_year(window) is None
    assert get_target_year(window, year_override=2020) == 2020


def test_manifest_contents_and_write(tmp_path: Path) -> None:
    """The manifest records product metadata, tallies, gaps, and CLI args."""
    ds_path, _ = make_dataset(tmp_path)
    fetcher = FakeFetcher(gap_bounds={GAP_BOUNDS})
    cli_args = {"dataset_path": str(ds_path), "products": "fake"}
    manifest = materialize_product(
        ds_path, fetcher, product_name="fake", cli_args=cli_args
    )

    assert manifest["product"] == "fake"
    assert manifest["product_version"] == "fake-v1"
    assert manifest["modality"] == "gse"
    assert manifest["year_policy"] == "window_time_range_midpoint"
    assert manifest["num_windows_written"] == 2
    assert manifest["num_windows_skipped_existing"] == 0
    assert manifest["num_coverage_gaps"] == 1
    assert manifest["coverage_gaps"] == ["default/w3"]
    assert manifest["cli_args"] == cli_args

    manifest_path = write_manifest(ds_path, "fake", manifest)
    assert manifest_path.name == "embedding_materializer_manifest_fake.json"
    with manifest_path.open() as f:
        assert json.load(f) == manifest


def test_write_embedding_validates_shape(tmp_path: Path) -> None:
    """Arrays whose shape mismatches the modality/window are rejected."""
    ds_path, windows = make_dataset(tmp_path)
    provider = RslearnWindowProvider(ds_path)
    bad_array = np.zeros((3, WINDOW_SIZE, WINDOW_SIZE), dtype=np.float32)
    with pytest.raises(ValueError, match="expected array of shape"):
        provider.write_embedding(windows[0], Modality.GSE, bad_array, nodata_value=-1.0)


def test_mosaic_tiles_to_bounds() -> None:
    """Tiles are placed on the requested grid; uncovered pixels stay NaN."""
    num_bands = 2
    bounds = (0, 0, 8, 8)
    # A tile in the same CRS/resolution covering the left half of the bounds.
    tile_array = np.ones((num_bands, 8, 4), dtype=np.float32)
    tile_array[1] = 5.0
    tile = SourceTile(
        array=tile_array,
        crs=PROJECTION.crs,
        transform=Affine(10, 0, 0, 0, -10, 0),
    )
    mosaic = mosaic_tiles_to_bounds([tile], PROJECTION, bounds, num_bands)
    assert mosaic is not None
    assert mosaic.shape == (num_bands, 8, 8)
    np.testing.assert_array_equal(mosaic[0, :, :4], 1.0)
    np.testing.assert_array_equal(mosaic[1, :, :4], 5.0)
    assert np.isnan(mosaic[:, :, 4:]).all()


def test_mosaic_tiles_to_bounds_first_valid_composite() -> None:
    """When tiles overlap, the first tile providing a pixel wins."""
    num_bands = 1
    bounds = (0, 0, 4, 4)
    transform = Affine(10, 0, 0, 0, -10, 0)
    first = SourceTile(
        array=np.full((1, 4, 4), 1.0, dtype=np.float32),
        crs=PROJECTION.crs,
        transform=transform,
    )
    second = SourceTile(
        array=np.full((1, 4, 4), 2.0, dtype=np.float32),
        crs=PROJECTION.crs,
        transform=transform,
    )
    mosaic = mosaic_tiles_to_bounds([first, second], PROJECTION, bounds, num_bands)
    assert mosaic is not None
    np.testing.assert_array_equal(mosaic, 1.0)


def test_mosaic_tiles_to_bounds_no_coverage() -> None:
    """A tile entirely outside the requested bounds yields None."""
    num_bands = 1
    bounds = (0, 0, 4, 4)
    # Tile located far away from the requested bounds.
    tile = SourceTile(
        array=np.ones((1, 4, 4), dtype=np.float32),
        crs=PROJECTION.crs,
        transform=Affine(10, 0, 100000, 0, -10, 100000),
    )
    assert mosaic_tiles_to_bounds([tile], PROJECTION, bounds, num_bands) is None
    assert mosaic_tiles_to_bounds([], PROJECTION, bounds, num_bands) is None


def test_year_time_range() -> None:
    """year_time_range spans the full calendar year in UTC."""
    start, end = year_time_range(2019)
    assert start == datetime(2019, 1, 1, tzinfo=UTC)
    assert end == datetime(2019, 12, 31, 23, 59, 59, tzinfo=UTC)


@pytest.mark.skipif(
    importlib.util.find_spec("geotessera") is not None,
    reason="geotessera is installed; the lazy-import error path does not apply",
)
def test_tessera_fetcher_requires_geotessera() -> None:
    """Constructing TesseraFetcher without geotessera raises a helpful error."""
    with pytest.raises(ImportError, match="pip install geotessera"):
        TesseraFetcher()


class FakeTesseraClient:
    """Minimal stand-in for geotessera.GeoTessera used to test TesseraFetcher."""

    def __init__(self, tiles: list[tuple[np.ndarray, CRS, Affine]]) -> None:
        """Initialize with (hwc array, crs, transform) tiles to serve."""
        self.tiles = tiles
        self.registry = self

    def load_blocks_for_region(
        self, bounds: tuple[float, float, float, float], year: int
    ) -> list[int]:
        """Return one opaque block descriptor per tile (empty list if none)."""
        return list(range(len(self.tiles)))

    def fetch_embeddings(
        self, tiles_to_fetch: list[int]
    ) -> Iterator[tuple[int, float, float, np.ndarray, CRS, Affine]]:
        """Yield (year, lon, lat, hwc array, crs, transform) per tile."""
        for idx in tiles_to_fetch:
            hwc, crs, transform = self.tiles[idx]
            yield (2024, 0.0, 0.0, hwc, crs, transform)


def test_tessera_fetcher_with_fake_client() -> None:
    """TesseraFetcher mosaics geotessera tiles onto the requested grid."""
    num_bands = len(Modality.TESSERA.band_order)
    hwc = np.ones((WINDOW_SIZE, WINDOW_SIZE, num_bands), dtype=np.float32)
    client = FakeTesseraClient(
        tiles=[(hwc, PROJECTION.crs, Affine(10, 0, 0, 0, -10, 0))]
    )
    fetcher = TesseraFetcher(client=client)
    array = fetcher.fetch((0, 0, WINDOW_SIZE, WINDOW_SIZE), PROJECTION, 2024)
    assert array is not None
    assert array.shape == (num_bands, WINDOW_SIZE, WINDOW_SIZE)
    np.testing.assert_array_equal(array, 1.0)

    # No blocks in the region -> coverage gap.
    empty_client = FakeTesseraClient(tiles=[])
    fetcher = TesseraFetcher(client=empty_client)
    assert fetcher.fetch((0, 0, WINDOW_SIZE, WINDOW_SIZE), PROJECTION, 2024) is None
