"""I/O helpers: disk checks, UTM/pixel math, and writing label GeoTIFFs + metadata.

All label patches are single-band, 10 m/pixel, in a local UTM projection, north-up.
Classification -> uint8 (class ids from 0; 255 = nodata/ignore). Regression -> source
dtype with a recorded nodata sentinel (default -99999).
"""

import json
import math
import shutil
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import shapely
from rslearn.const import WGS84_PROJECTION
from rslearn.utils.geometry import Projection, STGeometry
from rslearn.utils.get_utm_ups_crs import get_utm_ups_projection
from rslearn.utils.raster_array import RasterArray
from rslearn.utils.raster_format import GeotiffRasterFormat
from upath import UPath

from .manifest import OUTPUT_ROOT

RESOLUTION = 10
MIN_FREE_BYTES = 5_000_000_000_000  # 5 TB
CLASS_NODATA = 255
REGRESSION_NODATA = -99999
MAX_TILE = 64


def check_disk(path: UPath = OUTPUT_ROOT) -> int:
    """Raise if < 5 TB free on the weka volume; return free bytes otherwise."""
    free = shutil.disk_usage(str(path)).free
    if free < MIN_FREE_BYTES:
        raise RuntimeError(
            f"Only {free / 1e12:.2f} TB free on {path}; need >= 5 TB. Stopping."
        )
    return free


def dataset_dir(slug: str) -> UPath:
    """Return the output directory for a dataset's processed windows."""
    return OUTPUT_ROOT / "datasets" / slug


def locations_dir(slug: str) -> UPath:
    """Return the directory holding a dataset's per-location label tiles."""
    return dataset_dir(slug) / "locations"


def raw_dir(slug: str) -> UPath:
    """Return the directory for a dataset's raw (undownloaded) source files."""
    return OUTPUT_ROOT / "raw" / slug


def utm_projection_for_lonlat(lon: float, lat: float) -> Projection:
    """UTM/UPS projection at 10 m/pixel for a lon/lat."""
    return get_utm_ups_projection(lon, lat, RESOLUTION, -RESOLUTION)


def lonlat_to_utm_pixel(
    lon: float, lat: float, projection: Projection | None = None
) -> tuple[Projection, int, int]:
    """Return (projection, col, row): the 10 m pixel containing lon/lat."""
    if projection is None:
        projection = utm_projection_for_lonlat(lon, lat)
    geom = STGeometry(WGS84_PROJECTION, shapely.Point(lon, lat), None).to_projection(
        projection
    )
    return projection, int(math.floor(geom.shp.x)), int(math.floor(geom.shp.y))


def centered_bounds(
    col: int, row: int, width: int, height: int
) -> tuple[int, int, int, int]:
    """Pixel bounds for a width x height tile centered on pixel (col, row).

    width/height must be <= MAX_TILE.
    """
    if width > MAX_TILE or height > MAX_TILE:
        raise ValueError(f"tile {width}x{height} exceeds {MAX_TILE}")
    x_min = col - width // 2
    y_min = row - height // 2
    return (x_min, y_min, x_min + width, y_min + height)


def year_range(year: int) -> tuple[datetime, datetime]:
    """A one-year UTC time range [Jan 1 year, Jan 1 year+1)."""
    return (
        datetime(year, 1, 1, tzinfo=UTC),
        datetime(year + 1, 1, 1, tzinfo=UTC),
    )


def centered_time_range(
    center: datetime, half_window_days: int = 15
) -> tuple[datetime, datetime]:
    """A short UTC window [center - half_window_days, center + half_window_days].

    For rapidly-varying condition labels (e.g. live fuel moisture content, snow
    presence) whose value is only valid for a short period around the measurement
    date, rather than a static year. Default +/-15 days => a ~1-month window. The
    span stays well under the 360-day pretraining cap.
    """
    if center.tzinfo is None:
        center = center.replace(tzinfo=UTC)
    return (
        center - timedelta(days=half_window_days),
        center + timedelta(days=half_window_days),
    )


# Six months, used as the default half-span for change pre/post windows.
SIX_MONTHS_DAYS = 183


def pre_post_time_ranges(
    change_time: datetime,
    pre_months_days: int = SIX_MONTHS_DAYS,
    post_months_days: int = SIX_MONTHS_DAYS,
    gap_days: int = 0,
    pre_offset_days: int = 0,
) -> tuple[tuple[datetime, datetime], tuple[datetime, datetime]]:
    """Two ~6-month observation windows around a change event.

    Change/event labels are a mask of *where* a change occurred; pretraining pairs each
    with a "before" and an "after" image stack and probes on their difference. Instead of
    one ~1-year window centered on ``change_time`` (the old scheme), we emit two independent
    six-month windows so the pre period need not be adjacent to the post period:

        post = [change_time + gap_days, change_time + gap_days + post_months_days)
        pre  = [change_time - pre_offset_days - gap_days - pre_months_days,
                change_time - pre_offset_days - gap_days)

    With the defaults (``gap_days=0``, ``pre_offset_days=0``) the two windows are adjacent
    and split exactly at ``change_time`` (total span ~1 year, matching the old scheme).
    ``gap_days`` inserts a guard gap on both sides of ``change_time`` to absorb change-date
    imprecision. ``pre_offset_days`` additionally pushes the pre window earlier -- use it
    when ``change_time`` is a *post*-event acquisition date (the event itself is somewhat
    earlier) so the pre window ends before the true event. Each window stays <= 183 days,
    well under the 360-day pretraining cap. Returns ``(pre_range, post_range)``.
    """
    if change_time.tzinfo is None:
        change_time = change_time.replace(tzinfo=UTC)
    post_start = change_time + timedelta(days=gap_days)
    post = (post_start, post_start + timedelta(days=post_months_days))
    pre_end = change_time - timedelta(days=pre_offset_days + gap_days)
    pre = (pre_end - timedelta(days=pre_months_days), pre_end)
    return pre, post


def write_label_geotiff(
    slug: str,
    sample_id: str,
    array: np.ndarray,
    projection: Projection,
    bounds: tuple[int, int, int, int],
    nodata: float | None = None,
) -> None:
    """Write a single-band label patch atomically to locations/{sample_id}.tif.

    ``array`` is (H, W) or (1, H, W); its dtype is written as-is.
    """
    arr = np.asarray(array)
    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]
    assert arr.shape[0] == 1, "label patch must be single-band"
    opts = {"nodata": nodata} if nodata is not None else {}
    fmt = GeotiffRasterFormat(geotiff_options=opts)
    d = locations_dir(slug)
    d.mkdir(parents=True, exist_ok=True)
    tmp_name = f"{sample_id}.tif.tmp"
    fmt.encode_raster(d, projection, bounds, RasterArray(chw_array=arr), fname=tmp_name)
    (d / tmp_name).rename(d / f"{sample_id}.tif")


def _iso_range(
    tr: tuple[datetime, datetime] | list[str] | None,
) -> list[str] | None:
    """Normalize a (datetime, datetime) tuple or [iso, iso] list to [iso, iso]."""
    if tr is None:
        return None
    if isinstance(tr[0], str):
        return list(tr)  # type: ignore[arg-type]
    return [t.isoformat() for t in tr]  # type: ignore[union-attr]


def write_sample_json(
    slug: str,
    sample_id: str,
    projection: Projection,
    bounds: tuple[int, int, int, int],
    time_range: tuple[datetime, datetime] | None,
    change_time: datetime | None = None,
    source_id: str | None = None,
    classes_present: list[int] | None = None,
    pre_time_range: tuple[datetime, datetime] | None = None,
    post_time_range: tuple[datetime, datetime] | None = None,
) -> None:
    """Write the per-sample sidecar JSON.

    For change/event datasets pass ``pre_time_range``/``post_time_range`` (the two
    six-month windows from ``pre_post_time_ranges``); ``time_range`` is then forced to
    ``null`` because the two windows may be far apart (up to several years) and no single
    <=360-day window can represent them -- pretraining reads pre/post directly. Non-change
    datasets pass ``time_range`` and leave pre/post as None.
    """
    d = locations_dir(slug)
    d.mkdir(parents=True, exist_ok=True)
    has_prepost = pre_time_range is not None or post_time_range is not None
    obj: dict[str, Any] = {
        "crs": projection.crs.to_string(),
        "pixel_bounds": list(bounds),
        "time_range": None if has_prepost else _iso_range(time_range),
        "change_time": change_time.isoformat() if change_time else None,
        "pre_time_range": _iso_range(pre_time_range),
        "post_time_range": _iso_range(post_time_range),
    }
    if source_id is not None:
        obj["source_id"] = source_id
    if classes_present is not None:
        obj["classes_present"] = classes_present
    # Write atomically (.tmp then rename) so an interrupted write never leaves a
    # truncated/empty JSON that a later idempotent re-run would skip past.
    tmp = d / f"{sample_id}.json.tmp"
    with tmp.open("w") as f:
        json.dump(obj, f)
    tmp.rename(d / f"{sample_id}.json")


def write_dataset_metadata(slug: str, metadata: dict[str, Any]) -> None:
    """Write datasets/{slug}/metadata.json."""
    d = dataset_dir(slug)
    d.mkdir(parents=True, exist_ok=True)
    with (d / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)


def write_points_table(slug: str, task_type: str, points: list[dict[str, Any]]) -> None:
    """Write the dataset-wide point table as GeoJSON: datasets/{slug}/points.geojson (spec §2a).

    Each input point dict should have: id, lon, lat, label (class id or value), time_range
    (tuple[datetime, datetime] or [iso, iso]), and optionally change_time, source_id. Points
    are WGS84 lon/lat (GeoJSON's native CRS). Output is a FeatureCollection with one Point
    Feature per location; id/label/time_range/change_time/source_id live in ``properties``.
    dataset/task_type/count are FeatureCollection-level foreign members. Pure sparse-point
    datasets use this INSTEAD of per-point GeoTIFFs.

    Change/event point datasets additionally pass ``pre_time_range``/``post_time_range``
    (the two six-month windows from ``pre_post_time_ranges``); ``time_range`` is then
    written as ``null`` (see write_sample_json).

    Any extra keys in a point dict beyond the reserved set (id/lon/lat/label/time_range/
    change_time/source_id/pre_time_range/post_time_range) are copied verbatim into that
    feature's ``properties`` as auxiliary fields (e.g. a raw regression value alongside a
    classification ``label``).
    """
    reserved = {
        "id",
        "lon",
        "lat",
        "label",
        "time_range",
        "change_time",
        "source_id",
        "pre_time_range",
        "post_time_range",
    }
    d = dataset_dir(slug)
    d.mkdir(parents=True, exist_ok=True)
    features = []
    for p in points:
        tr = p.get("time_range")
        if tr and not isinstance(tr[0], str):
            tr = [t.isoformat() for t in tr]
        ct = p.get("change_time")
        if ct is not None and not isinstance(ct, str):
            ct = ct.isoformat()
        pre = _iso_range(p.get("pre_time_range"))
        post = _iso_range(p.get("post_time_range"))
        # Change points carry pre/post; a single <=360-day time_range can't represent two
        # possibly-far-apart windows, so it is null when pre/post are present.
        props = {
            "id": p["id"],
            "label": p["label"],
            "time_range": None if (pre or post) else tr,
            "change_time": ct,
            "pre_time_range": pre,
            "post_time_range": post,
            "source_id": p.get("source_id"),
        }
        for k, v in p.items():
            if k not in reserved:
                props[k] = v
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [p["lon"], p["lat"]]},
                "properties": props,
            }
        )
    obj = {
        "type": "FeatureCollection",
        "dataset": slug,
        "task_type": task_type,
        "count": len(features),
        "features": features,
    }
    tmp = d / "points.geojson.tmp"
    with tmp.open("w") as f:
        json.dump(obj, f)
    tmp.rename(d / "points.geojson")


def pixel_center_lonlat(
    crs: str, bounds: list[int] | tuple[int, ...]
) -> tuple[float, float]:
    """Return the WGS84 lon/lat of the center of a pixel-bounds box in the given CRS."""
    from rasterio.crs import CRS

    proj = Projection(CRS.from_string(crs), RESOLUTION, -RESOLUTION)
    cx = (bounds[0] + bounds[2]) / 2.0
    cy = (bounds[1] + bounds[3]) / 2.0
    geom = STGeometry(proj, shapely.Point(cx, cy), None).to_projection(WGS84_PROJECTION)
    return float(geom.shp.x), float(geom.shp.y)
