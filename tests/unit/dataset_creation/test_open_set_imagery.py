"""Tests for open-set imagery conversion window grouping and the change boundary."""

import csv
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rslearn.utils.geometry import Projection
from upath import UPath

from olmoearth_pretrain.data.constants import Modality, TimeSpan
from olmoearth_pretrain.dataset.utils import get_modality_fname
from olmoearth_pretrain.dataset_creation.rslearn_to_olmoearth.open_set_imagery import (
    convert_change_boundary,
    group_windows_by_example,
    is_paired_group,
)
from olmoearth_pretrain.dataset_creation.util import (
    get_modality_temp_meta_fname,
    get_window_metadata,
)


@dataclass
class _FakeWindow:
    """Duck-typed stand-in for an rslearn Window (only .options is used)."""

    name: str
    options: dict = field(default_factory=dict)
    projection: Projection | None = None
    bounds: tuple[int, int, int, int] | None = None
    time_range: tuple[datetime, datetime] | None = None


def _paired_windows() -> list[_FakeWindow]:
    projection = Projection(CRS.from_epsg(32610), 10, -10)
    # Bounds not aligned to the 128 px coarse pixel, as for centered windows.
    bounds = (1000 - 64, 2000 - 64, 1000 + 64, 2000 + 64)
    boundary = datetime(2021, 4, 1, tzinfo=UTC)
    options = {
        "crs": "EPSG:32610",
        "resolution": 10,
        "col": 1000,
        "row": 2000,
        "time": boundary.isoformat(),
        "example_id": "ds_p1",
    }
    pre = _FakeWindow(
        "ds_p1_pre",
        {**options, "paired_part": "pre"},
        projection,
        bounds,
        (datetime(2020, 10, 3, tzinfo=UTC), boundary),
    )
    post = _FakeWindow(
        "ds_p1_post",
        {**options, "paired_part": "post"},
        projection,
        bounds,
        (boundary, datetime(2021, 9, 28, tzinfo=UTC)),
    )
    return [post, pre]


def test_convert_change_boundary_writes_static_vector(tmp_path: Path) -> None:
    """Paired examples get a 1x1, 3-band [day, month-1, year] raster + meta CSV."""
    windows = _paired_windows()
    olmoearth_path = UPath(tmp_path)
    convert_change_boundary(windows, olmoearth_path)  # type: ignore[arg-type]

    modality = Modality.OPEN_SET_CHANGE_BOUNDARY
    band_set = modality.band_sets[0]
    fname = get_modality_fname(
        olmoearth_path,
        modality,
        TimeSpan.STATIC,
        get_window_metadata(windows[0]),  # type: ignore[arg-type]
        band_set.get_resolution(),
        "tif",
    )
    with rasterio.open(fname) as src:
        array = src.read()
    assert array.shape == (3, 1, 1)
    # timestamps convention: 0-based month.
    np.testing.assert_array_equal(array.reshape(-1), [1, 3, 2021])

    meta_fname = get_modality_temp_meta_fname(
        olmoearth_path, modality, TimeSpan.STATIC, "ds_p1"
    )
    with meta_fname.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["example_id"] == "ds_p1"
    assert rows[0]["start_time"] == "2020-10-03T00:00:00+00:00"
    assert rows[0]["end_time"] == "2021-09-28T00:00:00+00:00"


def test_convert_change_boundary_skips_unpaired(tmp_path: Path) -> None:
    """Regular (single-window) examples get no boundary raster."""
    single = _FakeWindow("a", {"example_id": "a"})
    assert not is_paired_group([single])  # type: ignore[list-item]
    convert_change_boundary([single], UPath(tmp_path))  # type: ignore[list-item]
    assert list(Path(tmp_path).iterdir()) == []


def test_group_windows_by_example() -> None:
    """Paired windows group by example_id; regular windows stay singletons."""
    single = _FakeWindow("a", {"example_id": "a"})
    pre = _FakeWindow("b_pre", {"example_id": "b", "paired_part": "pre"})
    post = _FakeWindow("b_post", {"example_id": "b", "paired_part": "post"})
    grid = _FakeWindow("EPSG:32610_10.0_5_6", {})

    groups = group_windows_by_example([single, pre, grid, post])  # type: ignore[list-item]

    by_size = sorted(groups, key=len)
    assert [len(g) for g in by_size] == [1, 1, 2]
    assert {g[0].name for g in by_size[:2]} == {"a", "EPSG:32610_10.0_5_6"}
    assert {w.name for w in by_size[2]} == {"b_pre", "b_post"}
