r"""Validate the acquired Landsat sun-elevation table.

Two independent layers of validation on the output of
``acquire_landsat_sun_elevation.py``:

Part A -- physical checks over *every* row (cheap, no network):
  * coverage: how many (window, month) rows resolved / are empty.
  * range: sun elevation in a sane band; flags <=0 (sun below horizon for a
    daytime optical scene) or >90.
  * platform sanity: LC08/LC09 only; flags LC09 in a pre-2021-11 month
    (Landsat-9 was not operational before then).
  * solar geometry: from the window centre lat/lon + month, computes the
    solar-noon maximum elevation ``90 - |lat - decl|`` and asserts the reported
    ``SUN_ELEVATION`` cannot exceed it (+tolerance). Also compares against the
    expected elevation at Landsat's ~10:30 local overpass and reports the
    residual distribution. A physically-impossible angle means we attached the
    wrong scene's / month's / location's elevation.

Part B -- definitive pixel-diff over a random sample (needs AWS creds):
  For each sampled window it re-selects the scene exactly as the acquisition did
  (same geometry, same default MOSAIC/WITHIN query, same cloud_cover sort), reads
  that scene's B8 straight from S3, warps it onto the *stored* DN tile's own grid,
  and correlates the two. If the re-selected scene is the one that actually
  produced the materialized tile, the correlation is ~1.0 (and most DN match
  exactly); a different scene correlates poorly. Partial scene coverage of the
  window flags a genuinely multi-scene (mosaicked) tile, where a single sun
  elevation is an approximation. This is the check that directly answers "did we
  apply the sun elevation from the correct image to this window?".

Usage:
    python -m olmoearth_pretrain.internal.validate_sun_elevation \
        --sun_elevation_csv /weka/dfive-default/yawenz/landsat_refl_work/sun_elevation.csv \
        --metadata_cache_dir /weka/dfive-default/yawenz/landsat_refl_work/landsat_metadata_cache \
        --sample 150 --output_report /weka/dfive-default/yawenz/landsat_refl_work/validation_report.json
"""

import argparse
import csv
import json
import logging
import math
import random
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import shapely.geometry

logger = logging.getLogger(__name__)

csv.field_size_limit(sys.maxsize)

TILE_SIZE = 256
MONTH_DURATION = timedelta(days=30)
# Landsat-9 became operational late 2021; anything earlier tagged LC09 is wrong.
LC09_FIRST_MONTH = datetime(2021, 11, 1)
# Approx local solar time of the Landsat descending-node overpass.
DEFAULT_OVERPASS_HOUR = 10.5


# --------------------------------------------------------------------------- #
# Part A: physical checks
# --------------------------------------------------------------------------- #
def solar_declination_deg(day_of_year: float) -> float:
    """Cooper's approximation of the solar declination (degrees)."""
    return 23.44 * math.sin(math.radians(360.0 * (day_of_year + 284) / 365.0))


def solar_elevation_deg(
    lat_deg: float, decl_deg: float, hour_angle_deg: float
) -> float:
    """Solar elevation (degrees) for a latitude, declination and hour angle."""
    lat = math.radians(lat_deg)
    decl = math.radians(decl_deg)
    h = math.radians(hour_angle_deg)
    sin_elev = math.sin(lat) * math.sin(decl) + math.cos(lat) * math.cos(
        decl
    ) * math.cos(h)
    return math.degrees(math.asin(max(-1.0, min(1.0, sin_elev))))


def compute_window_latlons(
    windows: set[tuple[str, int, int]],
) -> dict[tuple[str, int, int], tuple[float, float]]:
    """Batch window-centre (lat, lon) for all unique windows, one transformer per CRS.

    Uses the same pixel->CRS convention as the acquisition's projection
    (``x_resolution=10``, ``y_resolution=-10``): the centre pixel of window
    ``(col, row)`` is ``((col+0.5)*256, (row+0.5)*256)`` and maps to CRS metres by
    multiplying by the (signed) resolution. Batching per CRS makes this ~1000x
    faster than reprojecting each window through rslearn individually.
    """
    from pyproj import Transformer

    by_crs: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for crs, col, row in windows:
        by_crs[crs].append((col, row))

    out: dict[tuple[str, int, int], tuple[float, float]] = {}
    for crs, colrows in by_crs.items():
        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        xs = [(c + 0.5) * TILE_SIZE * 10.0 for c, _ in colrows]
        ys = [(r + 0.5) * TILE_SIZE * -10.0 for _, r in colrows]
        lons, lats = transformer.transform(xs, ys)
        for (c, r), lon, lat in zip(colrows, lons, lats):
            out[(crs, c, r)] = (float(lat), float(lon))
    return out


def run_part_a(rows: list[dict], overpass_hour: float, bound_tol: float) -> dict:
    """Physical / geometric validation over all rows."""
    # Precompute window-centre lat/lon for every unique window (batched per CRS).
    unique_windows = {
        (r["crs"], int(r["col"]), int(r["row"])) for r in rows if r.get("start_time")
    }
    latlon_map = compute_window_latlons(unique_windows)

    total = len(rows)
    empty = 0
    valid_elev: list[float] = []
    below_horizon = 0
    above_90 = 0
    platforms: dict[str, int] = defaultdict(int)
    lc09_too_early = 0

    bound_violations = 0
    residuals: list[float] = []
    # Per-window residuals for the seasonal-shape check.
    per_window_resid: dict[tuple[str, int, int], list[float]] = defaultdict(list)

    for r in rows:
        se_raw = r["sun_elevation"]
        plat = r.get("platform") or ""
        if plat:
            platforms[plat] += 1
        if se_raw in ("", None):
            empty += 1
            continue
        se = float(se_raw)
        valid_elev.append(se)
        if se <= 0:
            below_horizon += 1
        if se > 90:
            above_90 += 1

        if not r.get("start_time"):
            # No month date available -> skip the geometry check for this row.
            continue
        crs, col, row_i = r["crs"], int(r["col"]), int(r["row"])
        start = datetime.fromisoformat(r["start_time"])
        end = (
            datetime.fromisoformat(r["end_time"])
            if r.get("end_time")
            else start + MONTH_DURATION
        )
        mid = start + (end - start) / 2
        if plat == "LC09" and mid.replace(tzinfo=None) < LC09_FIRST_MONTH:
            lc09_too_early += 1

        lat, _lon = latlon_map[(crs, col, row_i)]
        decl = solar_declination_deg(mid.timetuple().tm_yday)
        noon_max = solar_elevation_deg(lat, decl, 0.0)
        expected = solar_elevation_deg(lat, decl, 15.0 * (overpass_hour - 12.0))
        if se > noon_max + bound_tol:
            bound_violations += 1
        resid_i = se - expected
        residuals.append(resid_i)
        per_window_resid[(crs, col, row_i)].append(resid_i)

    elev = np.array(valid_elev, dtype=np.float64)
    resid = np.array(residuals, dtype=np.float64)

    # Seasonal-shape: a window whose worst monthly residual is large is suspicious
    # (a single mis-selected month stands out against the smooth seasonal curve).
    seasonal_flag_threshold = 12.0
    flagged_windows = sum(
        1
        for v in per_window_resid.values()
        if max(abs(x) for x in v) > seasonal_flag_threshold
    )

    def q(a: np.ndarray, p: float) -> float:
        return float(np.percentile(a, p)) if a.size else float("nan")

    return {
        "coverage": {
            "total_rows": total,
            "empty_sun_elevation": empty,
            "empty_fraction": empty / total if total else 0.0,
            "resolved_rows": int(elev.size),
        },
        "range": {
            "min": float(elev.min()) if elev.size else None,
            "p01": q(elev, 1),
            "median": q(elev, 50),
            "p99": q(elev, 99),
            "max": float(elev.max()) if elev.size else None,
            "count_le_0_below_horizon": below_horizon,
            "count_gt_90": above_90,
        },
        "platform": {
            "counts": dict(platforms),
            "lc09_before_2021_11": lc09_too_early,
        },
        "solar_geometry": {
            "overpass_hour_local": overpass_hour,
            "noon_bound_violations": bound_violations,
            "noon_bound_violation_fraction": (
                bound_violations / elev.size if elev.size else 0.0
            ),
            "residual_vs_expected_deg": {
                "median": q(resid, 50),
                "p05": q(resid, 5),
                "p95": q(resid, 95),
                "abs_median": float(np.median(np.abs(resid))) if resid.size else None,
                "count_abs_gt_10": int((np.abs(resid) > 10).sum()),
            },
            "seasonal_shape": {
                "windows_checked": len(per_window_resid),
                "flag_threshold_deg": seasonal_flag_threshold,
                "windows_flagged": flagged_windows,
                "flagged_fraction": (
                    flagged_windows / len(per_window_resid) if per_window_resid else 0.0
                ),
            },
        },
    }


# --------------------------------------------------------------------------- #
# Part B: definitive re-materialize pixel-diff
# --------------------------------------------------------------------------- #
def _window_geometry(crs: str, col: int, row: int, start: datetime) -> Any:
    from rslearn.utils.geometry import Projection, STGeometry

    projection = Projection.deserialize(
        {"crs": crs, "x_resolution": 10, "y_resolution": -10}
    )
    bounds = (
        col * TILE_SIZE,
        row * TILE_SIZE,
        (col + 1) * TILE_SIZE,
        (row + 1) * TILE_SIZE,
    )
    return STGeometry(
        projection, shapely.geometry.box(*bounds), (start, start + MONTH_DURATION)
    )


def _correlate(a: np.ndarray, b: np.ndarray, valid: np.ndarray) -> float:
    if valid.sum() < 32:
        return float("nan")
    av = a[valid].astype(np.float64)
    bv = b[valid].astype(np.float64)
    if av.std() < 1e-6 or bv.std() < 1e-6:
        return float("nan")
    return float(np.corrcoef(av, bv)[0, 1])


def run_part_b(
    rows_by_window: dict,
    landsat_tif_dir: str,
    metadata_cache_dir: str,
    sample: int,
    seed: int,
) -> dict:
    """Pixel-diff a random sample of windows against the re-selected scene."""
    import rasterio
    from rasterio.vrt import WarpedVRT
    from rslearn.data_sources.aws_landsat import LandsatOliTirs
    from rslearn.data_sources.data_source import DataSourceContext, QueryConfig

    src = LandsatOliTirs(
        metadata_cache_dir=metadata_cache_dir,
        sort_by="cloud_cover",
        context=DataSourceContext(),
    )

    keys = [k for k, v in rows_by_window.items() if any(r["sun_elevation"] for r in v)]
    random.Random(seed).shuffle(keys)
    keys = keys[:sample]

    results: list[dict[str, Any]] = []
    same_scene = 0
    multi_scene = 0
    failures = 0
    for crs, col, row, _tile_time in keys:
        window_rows = rows_by_window[(crs, col, row, _tile_time)]
        # Pick the best-illuminated month with a resolved elevation and a date.
        cand = [r for r in window_rows if r["sun_elevation"] and r.get("start_time")]
        cand.sort(key=lambda r: float(r["sun_elevation"]), reverse=True)
        picked = None
        tif_path = f"{landsat_tif_dir}/{crs}_{col}_{row}_10.tif"
        try:
            stored = rasterio.open(tif_path)
        except Exception as e:  # noqa: BLE001
            failures += 1
            results.append(
                {"window": f"{crs}_{col}_{row}", "error": f"open stored: {e}"}
            )
            continue

        with stored:
            n_bands = stored.count
            dst_crs = stored.crs
            dst_transform = stored.transform
            for r in cand:
                m = int(r["image_idx"])
                if m + 1 > n_bands:
                    continue
                a = stored.read(m + 1)
                if (a > 0).mean() < 0.20:  # mostly nodata/cloud; poor comparison
                    continue
                picked = (r, m, a)
                break
            if picked is None:
                failures += 1
                results.append(
                    {"window": f"{crs}_{col}_{row}", "error": "no usable month"}
                )
                continue
            r, m, a = picked

            start = datetime.fromisoformat(r["start_time"])
            try:
                groups = src.get_items(
                    [_window_geometry(crs, col, row, start)], QueryConfig()
                )
            except Exception as e:  # noqa: BLE001
                failures += 1
                results.append(
                    {"window": f"{crs}_{col}_{row}", "error": f"get_items: {e}"}
                )
                continue
            if not groups or not groups[0] or not groups[0][0].items:
                failures += 1
                results.append(
                    {"window": f"{crs}_{col}_{row}", "error": "no scene reselected"}
                )
                continue
            base = groups[0][0].items[0]
            n_in_group = len(groups[0][0].items)

            try:
                url = src.get_asset_url(base, "B8")
                with rasterio.open(f"/vsicurl/{url}") as scene:
                    with WarpedVRT(
                        scene,
                        crs=dst_crs,
                        transform=dst_transform,
                        width=a.shape[1],
                        height=a.shape[0],
                        resampling=rasterio.enums.Resampling.nearest,
                        src_nodata=0,
                        nodata=0,
                    ) as vrt:
                        b = vrt.read(1)
            except Exception as e:  # noqa: BLE001
                failures += 1
                results.append(
                    {
                        "window": f"{crs}_{col}_{row}",
                        "scene": base.name,
                        "error": f"warp: {e}",
                    }
                )
                continue

            valid = (a > 0) & (b > 0)
            scene_cov = float((b[a > 0] > 0).mean()) if (a > 0).any() else 0.0
            corr = _correlate(a, b, valid)
            exact = (
                float((a[valid] == b[valid]).mean()) if valid.any() else float("nan")
            )
            is_same = (not math.isnan(corr)) and corr > 0.98
            is_multi = scene_cov < 0.99
            if is_same:
                same_scene += 1
            if is_multi:
                multi_scene += 1
            results.append(
                {
                    "window": f"{crs}_{col}_{row}",
                    "image_idx": m,
                    "scene": base.name,
                    "n_scenes_in_group": n_in_group,
                    "sun_elevation": float(r["sun_elevation"]),
                    "corr": round(corr, 4) if not math.isnan(corr) else None,
                    "exact_dn_frac": round(exact, 4) if not math.isnan(exact) else None,
                    "scene_coverage": round(scene_cov, 4),
                    "same_scene": is_same,
                    "multi_scene": is_multi,
                }
            )

    checked = sum(1 for r in results if "corr" in r and r["corr"] is not None)
    return {
        "sampled_windows": len(keys),
        "checked": checked,
        "failures": failures,
        "same_scene": same_scene,
        "same_scene_fraction": same_scene / checked if checked else 0.0,
        "multi_scene": multi_scene,
        "multi_scene_fraction": multi_scene / checked if checked else 0.0,
        "worst": sorted(
            (r for r in results if r.get("corr") is not None),
            key=lambda r: r["corr"],
        )[:10],
        "all_results": results,
    }


def main() -> None:
    """Entry point."""
    logging.basicConfig(level=logging.INFO)
    for noisy in ("botocore", "boto3", "s3transfer", "urllib3", "rasterio", "rslearn"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(
        description="Validate acquired Landsat sun elevation"
    )
    parser.add_argument("--sun_elevation_csv", required=True)
    parser.add_argument(
        "--landsat_csv",
        default="/weka/dfive-default/helios/dataset/osm_sampling/10_landsat_monthly.csv",
        help="monthly CSV supplying per-(window, month) start/end dates",
    )
    parser.add_argument(
        "--landsat_tif_dir",
        default="/weka/dfive-default/helios/dataset/osm_sampling/10_landsat_monthly",
    )
    parser.add_argument(
        "--metadata_cache_dir",
        default="/weka/dfive-default/yawenz/landsat_refl_work/landsat_metadata_cache",
    )
    parser.add_argument(
        "--sample", type=int, default=150, help="Part B window sample (0 to skip)"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overpass_hour", type=float, default=DEFAULT_OVERPASS_HOUR)
    parser.add_argument("--bound_tol", type=float, default=2.0)
    parser.add_argument("--output_report", default=None)
    args = parser.parse_args()

    # Dates live in the monthly CSV, not the acquire output; join them in.
    logger.info(f"loading month dates from {args.landsat_csv}")
    date_lookup: dict[tuple, tuple[str, str]] = {}
    with open(args.landsat_csv) as f:
        for r in csv.DictReader(f):
            date_lookup[
                (r["crs"], r["col"], r["row"], r["tile_time"], r["image_idx"])
            ] = (r.get("start_time", ""), r.get("end_time", ""))
    logger.info(f"loaded {len(date_lookup)} month-date entries")

    rows: list[dict] = []
    rows_by_window: dict = defaultdict(list)
    missing_date = 0
    with open(args.sun_elevation_csv) as f:
        for r in csv.DictReader(f):
            start, end = date_lookup.get(
                (r["crs"], r["col"], r["row"], r["tile_time"], r["image_idx"]),
                ("", ""),
            )
            if not start:
                missing_date += 1
            r["start_time"], r["end_time"] = start, end
            rows.append(r)
            rows_by_window[
                (r["crs"], int(r["col"]), int(r["row"]), r["tile_time"])
            ].append(r)

    logger.info(
        f"loaded {len(rows)} rows / {len(rows_by_window)} windows "
        f"({missing_date} rows missing a month date)"
    )

    report: dict = {"input": args.sun_elevation_csv}
    logger.info("running Part A (physical checks over all rows)")
    report["part_a_physical"] = run_part_a(rows, args.overpass_hour, args.bound_tol)

    if args.sample > 0:
        logger.info(f"running Part B (pixel-diff on {args.sample} windows)")
        report["part_b_pixel_diff"] = run_part_b(
            rows_by_window,
            args.landsat_tif_dir,
            args.metadata_cache_dir,
            args.sample,
            args.seed,
        )

    # Console summary.
    a = report["part_a_physical"]
    print("\n==== Part A: physical checks ====")
    print(json.dumps({k: v for k, v in a.items()}, indent=2))
    if "part_b_pixel_diff" in report:
        b = report["part_b_pixel_diff"]
        print("\n==== Part B: re-materialize pixel-diff ====")
        print(
            json.dumps(
                {k: v for k, v in b.items() if k != "all_results"},
                indent=2,
            )
        )

    if args.output_report:
        with open(args.output_report, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"wrote full report to {args.output_report}")


if __name__ == "__main__":
    main()
