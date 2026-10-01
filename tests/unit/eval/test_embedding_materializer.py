"""Unit tests for the embedding materializer (rslearn layer + manifest)."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import rasterio
import shapely
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rslearn.data_sources.aws_google_satellite_embedding_v1 import (
    GoogleSatelliteEmbeddingV1,
)
from rslearn.data_sources.data_source import Item
from rslearn.dataset import Dataset, Window
from rslearn.dataset import manage as rslearn_manage
from rslearn.utils.geometry import (
    WGS84_PROJECTION,
    PixelBounds,
    Projection,
    STGeometry,
)
from rslearn.utils.raster_array import RasterArray, RasterMetadata
from upath import UPath

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.evals.embedding_materializer.materialize import (
    AEF,
    CONFIG_BACKUP_NAME,
    EmbeddingProduct,
    get_target_year,
    materialize_product,
    request_time_offset,
    write_manifest,
)

PROJECTION = Projection(CRS.from_epsg(32610), 10, -10)
WINDOW_SIZE = 16
# Starts in 2019 but is centred on 2020-03, so reading the midpoint year (2020)
# differs from rslearn's default of the earliest matching item (2019).
TIME_RANGE = (
    datetime(2019, 9, 1, tzinfo=UTC),
    datetime(2020, 9, 1, tzinfo=UTC),
)
# Pixel origin of the windows: 500 km easting, 4000 km northing in UTM 10N.
ORIGIN = (50_000, -400_000)
# w3 sits 100 km away from the product's footprint, so it is a coverage gap.
GAP_OFFSET = 10_000


def window_bounds(idx: int) -> PixelBounds:
    """Return the pixel bounds of the idx-th test window."""
    offset = idx * WINDOW_SIZE + (GAP_OFFSET if idx == 2 else 0)
    x0, y0 = ORIGIN[0] + offset, ORIGIN[1] + offset
    return (x0, y0, x0 + WINDOW_SIZE, y0 + WINDOW_SIZE)


def expected_raster(bounds: PixelBounds, year: int) -> np.ndarray:
    """The deterministic (C, H, W) array FakeAEFSource returns for a read."""
    num_bands = len(Modality.GSE.band_order)
    height, width = bounds[3] - bounds[1], bounds[2] - bounds[0]
    values = np.arange(num_bands * height * width, dtype=np.float32) / 1e6 + year
    return values.reshape(num_bands, height, width)


class FakeAEFSource(GoogleSatelliteEmbeddingV1):
    """AEF data source with an in-memory index and synthetic rasters.

    The index has 2019 and 2020 items covering w1 and w2 only. Reads for
    bounds listed in ``fail_bounds`` raise OSError while their failure budget
    lasts. State lives on the class because rslearn instantiates the source
    from its class path.
    """

    fail_bounds: dict[PixelBounds, int | None] = {}
    attempts: dict[PixelBounds, int] = {}

    def _read_index_csv(self) -> pd.DataFrame:
        """Return a two-year index whose footprint covers w1 and w2."""
        corners = [window_bounds(0), window_bounds(1)]
        footprint = STGeometry(
            PROJECTION,
            shapely.box(
                corners[0][0] - 64,
                corners[0][1] - 64,
                corners[1][2] + 64,
                corners[1][3] + 64,
            ),
            None,
        ).to_projection(WGS84_PROJECTION)
        return pd.DataFrame(
            [
                {"WKT": footprint.shp.wkt, "year": year, "path": f"s3://x/{year}.tiff"}
                for year in (2019, 2020)
            ]
        )

    def read_raster(
        self,
        layer_name: str,
        item: Item,
        bands: list[str],
        projection: Projection,
        bounds: PixelBounds,
        resampling: Resampling = Resampling.bilinear,
    ) -> RasterArray:
        """Return expected_raster for the item's year, failing on demand."""
        key = tuple(bounds)
        if key in self.fail_bounds:
            self.attempts[key] = self.attempts.get(key, 0) + 1
            budget = self.fail_bounds[key]
            if budget is None or self.attempts[key] <= budget:
                raise OSError("simulated transient read failure")
        return RasterArray(
            chw_array=expected_raster(bounds, item.geometry.time_range[0].year),
            time_range=item.geometry.time_range,
            metadata=RasterMetadata(nodata_value=-1.0),
        )


FAKE_AEF = EmbeddingProduct(
    name="fake",
    modality=Modality.GSE,
    product_version="fake-v1",
    data_source_class_path=f"{__name__}.FakeAEFSource",
    nodata_value=-1.0,
)


@pytest.fixture(autouse=True)
def reset_fake_source(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear FakeAEFSource's failure state and make rslearn retries instant."""
    FakeAEFSource.fail_bounds = {}
    FakeAEFSource.attempts = {}
    monkeypatch.setattr(rslearn_manage.time, "sleep", lambda _: None)


def make_dataset(
    tmp_path: Path, time_ranges: list[tuple[datetime, datetime] | None] | None = None
) -> tuple[UPath, list[Window]]:
    """Create a tiny rslearn dataset on disk with three windows.

    Args:
        tmp_path: pytest tmp_path fixture value.
        time_ranges: per-window time ranges; defaults to TIME_RANGE for all.

    Returns:
        tuple of (dataset path, list of windows). The third window ("w3") lies
        outside the fake product's footprint.
    """
    ds_path = UPath(tmp_path) / "dataset"
    ds_path.mkdir(parents=True)
    with (ds_path / "config.json").open("w") as f:
        json.dump({"layers": {}}, f)

    if time_ranges is None:
        time_ranges = [TIME_RANGE] * 3
    dataset = Dataset(ds_path)
    windows = []
    for idx, (name, time_range) in enumerate(zip(["w1", "w2", "w3"], time_ranges)):
        window = Window(
            storage=dataset.storage,
            group="default",
            name=name,
            projection=PROJECTION,
            bounds=window_bounds(idx),
            time_range=time_range,
        )
        window.save()
        windows.append(window)
    return ds_path, windows


def read_layer(window: Window) -> tuple[np.ndarray, Any]:
    """Read a window's materialized gse raster and its nodata value."""
    path = window.get_raster_dir("gse", Modality.GSE.band_order) / "geotiff.tif"
    with rasterio.open(str(path)) as src:
        return src.read(), src.nodata


def test_materialize_writes_midpoint_year(tmp_path: Path) -> None:
    """Covered windows get the midpoint year's raster; the gap window gets none."""
    ds_path, windows = make_dataset(tmp_path)
    manifest = materialize_product(ds_path, FAKE_AEF)

    for window in windows[:2]:
        assert window.is_layer_completed("gse")
        data, nodata = read_layer(window)
        assert data.dtype == np.float32
        assert nodata == -1.0
        np.testing.assert_allclose(data, expected_raster(window.bounds, 2020))

    assert not windows[2].is_layer_completed("gse")
    assert manifest["num_windows_written"] == 2
    assert manifest["coverage_gaps"] == ["default/w3"]
    assert manifest["num_windows_failed"] == 0


def test_layer_is_declared_in_config(tmp_path: Path) -> None:
    """The layer is written to config.json with a direct-read data source."""
    ds_path, _ = make_dataset(tmp_path)
    materialize_product(ds_path, FAKE_AEF)

    with (ds_path / "config.json").open() as f:
        layer = json.load(f)["layers"]["gse"]
    assert layer["band_sets"][0]["bands"] == list(Modality.GSE.band_order)
    assert layer["data_source"]["class_path"] == FAKE_AEF.data_source_class_path
    assert layer["data_source"]["ingest"] is False
    assert (ds_path / CONFIG_BACKUP_NAME).exists()
    # rslearn itself can load the config.
    assert "gse" in Dataset(ds_path).layers


def test_idempotent_rerun_skips_existing(tmp_path: Path) -> None:
    """A re-run without overwrite reads nothing and reports every window as done."""
    ds_path, _ = make_dataset(tmp_path)
    materialize_product(ds_path, FAKE_AEF)

    FakeAEFSource.fail_bounds = {window_bounds(0): None, window_bounds(1): None}
    manifest = materialize_product(ds_path, FAKE_AEF)
    assert FakeAEFSource.attempts == {}
    assert manifest["num_windows_written"] == 0
    assert manifest["num_windows_skipped_existing"] == 2
    assert manifest["num_coverage_gaps"] == 1


def test_overwrite_rewrites(tmp_path: Path) -> None:
    """With overwrite=True, existing layers are materialized again."""
    ds_path, _ = make_dataset(tmp_path)
    materialize_product(ds_path, FAKE_AEF)

    # workers=2 also exercises the threaded path.
    manifest = materialize_product(ds_path, FAKE_AEF, overwrite=True, workers=2)
    assert manifest["num_windows_written"] == 2
    assert manifest["num_windows_skipped_existing"] == 0


def test_transient_read_error_is_retried(tmp_path: Path) -> None:
    """A read that fails transiently is retried and the window still written."""
    ds_path, windows = make_dataset(tmp_path)
    FakeAEFSource.fail_bounds = {window_bounds(1): 2}
    manifest = materialize_product(ds_path, FAKE_AEF)

    assert FakeAEFSource.attempts[window_bounds(1)] == 3
    assert manifest["num_windows_written"] == 2
    assert manifest["num_windows_failed"] == 0
    assert windows[1].is_layer_completed("gse")


def test_persistent_read_error_recorded_not_fatal(tmp_path: Path) -> None:
    """A window that keeps failing is recorded as failed; a re-run picks it up."""
    ds_path, windows = make_dataset(tmp_path)
    FakeAEFSource.fail_bounds = {window_bounds(1): None}
    manifest = materialize_product(ds_path, FAKE_AEF, workers=2)

    assert manifest["num_windows_written"] == 1
    assert manifest["windows_failed"] == ["default/w2"]
    assert not windows[1].is_layer_completed("gse")

    FakeAEFSource.fail_bounds = {}
    manifest = materialize_product(ds_path, FAKE_AEF)
    assert manifest["num_windows_written"] == 1
    assert manifest["num_windows_skipped_existing"] == 1
    assert windows[1].is_layer_completed("gse")


def test_windows_without_time_range_are_skipped(tmp_path: Path) -> None:
    """Windows with no time range are reported, not materialized."""
    ds_path, windows = make_dataset(tmp_path, [TIME_RANGE, None, TIME_RANGE])
    manifest = materialize_product(ds_path, FAKE_AEF)

    assert manifest["windows_without_year"] == ["default/w2"]
    assert manifest["num_windows_written"] == 1
    assert not windows[1].is_layer_completed("gse")


def test_request_time_offset_handles_leap_years(tmp_path: Path) -> None:
    """Calendar-year windows of 365 and 366 days share one offset."""
    years = [2019, 2020, 2021]
    _, windows = make_dataset(
        tmp_path,
        [
            (datetime(y, 1, 1, tzinfo=UTC), datetime(y + 1, 1, 1, tzinfo=UTC))
            for y in years
        ],
    )
    offset = request_time_offset(windows)
    assert [(w.time_range[0] + offset).year for w in windows] == years
    assert [get_target_year(w) for w in windows] == years


def test_request_time_offset_rejects_mixed_lengths(tmp_path: Path) -> None:
    """One offset cannot serve windows whose midpoints fall in different years."""
    start = datetime(2019, 9, 1, tzinfo=UTC)
    _, windows = make_dataset(
        tmp_path,
        [
            (start, start + timedelta(days=60)),
            (start, start + timedelta(days=60)),
            (start, start + timedelta(days=366)),
        ],
    )
    with pytest.raises(ValueError, match="different year"):
        request_time_offset(windows)


def test_manifest_contents_and_write(tmp_path: Path) -> None:
    """The manifest records product metadata, tallies, gaps, and CLI args."""
    ds_path, _ = make_dataset(tmp_path)
    cli_args = {"dataset_path": str(ds_path), "products": "fake"}
    manifest = materialize_product(ds_path, FAKE_AEF, cli_args=cli_args)

    assert manifest["product"] == "fake"
    assert manifest["product_version"] == "fake-v1"
    assert manifest["modality"] == "gse"
    assert manifest["year_policy"] == "window_time_range_midpoint"
    assert manifest["num_coverage_gaps"] == 1
    assert manifest["cli_args"] == cli_args

    manifest_path = write_manifest(ds_path, "fake", manifest)
    assert manifest_path.name == "embedding_materializer_manifest_fake.json"
    with manifest_path.open() as f:
        assert json.load(f) == manifest


def test_aef_product_points_at_rslearn_source() -> None:
    """The real AEF product uses rslearn's AWS Google Satellite Embedding source."""
    module, _, name = AEF.data_source_class_path.rpartition(".")
    assert (
        getattr(__import__(module, fromlist=[name]), name) is GoogleSatelliteEmbeddingV1
    )
