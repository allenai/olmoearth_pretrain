r"""Re-acquire per-scene Landsat SUN_ELEVATION for the existing monthly tiles.

The Landsat DN tiles in the OlmoEarth tile store were materialized from an
rslearn dataset that no longer keeps the landsat layers in its ``items.json``.
To convert those DN tiles to TOA reflectance / brightness temperature we need
the per-(window, month) scene sun elevation that produced each tile.

This script reconstructs it *without mutating any dataset*: for every
(window, month) row in the existing landsat monthly CSV it re-runs the same
data-source query that materialization used (least-cloudy single-coverage
mosaic over the month), takes the base scene, and reads ``SUN_ELEVATION`` from
that scene's MTL. The result is written to an output CSV keyed by
``(crs, col, row, tile_time, image_idx)`` which
``join_landsat_sun_elevation.py`` then merges into the modality CSV.

Window geometry is reconstructed directly from the window name grid indices
(``bounds = [col*256, row*256, (col+1)*256, (row+1)*256]`` at 10 m, verified
against materialized windows), so windows that are no longer present in the
source rslearn dataset are still handled. The month time range for each row is
taken directly from the CSV ``start_time`` (``end_time = start_time + 30d``),
which is exactly how ``convert_monthly`` defines the monthly layer windows, so
the reselected scene matches the tile.

Requester-pays AWS creds must be in the environment (AWS_ACCESS_KEY_ID /
AWS_SECRET_ACCESS_KEY). The run is resumable: windows already present in the
output CSV are skipped, the shared ``metadata_cache_dir`` caches per
(year, path, row) STAC listings, and the shared sun-elevation cache
(``<metadata_cache_dir>/../mtl_sun_elevation_cache``) avoids re-fetching MTLs.

Usage:
    python -m olmoearth_pretrain.internal.acquire_landsat_sun_elevation \
        --landsat_csv /weka/dfive-default/helios/dataset/osm_sampling/10_landsat_monthly.csv \
        --output_csv /weka/dfive-default/yawenz/landsat_refl_work/sun_elevation.csv \
        --metadata_cache_dir /weka/dfive-default/yawenz/landsat_refl_work/landsat_metadata_cache \
        --sun_elevation_cache_dir /weka/dfive-default/yawenz/landsat_refl_work/mtl_sun_elevation_cache \
        --workers 32
"""

import argparse
import csv
import logging
import multiprocessing
from collections import defaultdict
from datetime import datetime, timedelta

import shapely.geometry
import tqdm
from rslearn.data_sources.aws_landsat import LandsatOliTirs
from rslearn.data_sources.data_source import DataSourceContext, QueryConfig
from rslearn.utils.geometry import Projection, STGeometry
from upath import UPath

from olmoearth_pretrain.dataset_creation.rslearn_to_olmoearth.landsat_calibration import (
    _scene_id_from_blob_path,
    fetch_sun_elevation,
    platform_from_scene_id,
)

logger = logging.getLogger(__name__)

# Pixels per grid tile for res_10 windows (see dataset_creation PIXELS_PER_TILE).
TILE_SIZE = 256
# Monthly layer duration (matches config_landsat.json data_source "duration").
MONTH_DURATION = timedelta(days=30)
OUTPUT_COLUMNS = [
    "crs",
    "col",
    "row",
    "tile_time",
    "image_idx",
    "sun_elevation",
    "platform",
]

# Worker globals (initialized once per process).
_SRC: LandsatOliTirs | None = None
_SUN_CACHE_DIR: UPath | None = None


def _init_worker(metadata_cache_dir: str, sun_elevation_cache_dir: str) -> None:
    global _SRC, _SUN_CACHE_DIR
    # context defaults (ds_path=None) so metadata_cache_dir is used verbatim.
    _SRC = LandsatOliTirs(
        metadata_cache_dir=metadata_cache_dir,
        sort_by="cloud_cover",
        context=DataSourceContext(),
    )
    _SUN_CACHE_DIR = UPath(sun_elevation_cache_dir)
    _SUN_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _window_geometry(crs: str, col: int, row: int, start_iso: str) -> STGeometry:
    projection = Projection.deserialize(
        {"crs": crs, "x_resolution": 10, "y_resolution": -10}
    )
    bounds = (
        col * TILE_SIZE,
        row * TILE_SIZE,
        (col + 1) * TILE_SIZE,
        (row + 1) * TILE_SIZE,
    )
    start = datetime.fromisoformat(start_iso)
    return STGeometry(
        projection, shapely.geometry.box(*bounds), (start, start + MONTH_DURATION)
    )


def _cached_sun_elevation(blob_path: str) -> float | None:
    """Sun elevation for a scene, backed by a shared on-disk cache."""
    assert _SUN_CACHE_DIR is not None, (
        "_init_worker must run before _cached_sun_elevation"
    )
    scene_id = _scene_id_from_blob_path(blob_path)
    cache_file = _SUN_CACHE_DIR / f"{scene_id}.txt"
    if cache_file.exists():
        try:
            text = cache_file.read_text().strip()
            return float(text) if text else None
        except (ValueError, OSError):
            pass
    sun_elevation = fetch_sun_elevation(blob_path)
    try:
        cache_file.write_text("" if sun_elevation is None else repr(sun_elevation))
    except OSError:
        pass
    return sun_elevation


def _process_window(
    task: tuple[tuple[str, str, str, str], list[tuple[int, str]]],
) -> list[tuple]:
    """Resolve sun elevation for every (image_idx) row of a single window."""
    assert _SRC is not None, "_init_worker must run before _process_window"
    (crs, col, row, tile_time), rows = task
    out_rows: list[tuple] = []
    geometries = [
        _window_geometry(crs, int(col), int(row), start_iso) for _, start_iso in rows
    ]
    try:
        results = _SRC.get_items(geometries, QueryConfig())
    except Exception as e:  # noqa: BLE001 - one bad window shouldn't kill the pool
        logger.warning(f"get_items failed for {crs}_{col}_{row}: {e}")
        return [(crs, col, row, tile_time, image_idx, "", "") for image_idx, _ in rows]

    for (image_idx, _), groups in zip(rows, results):
        sun_elevation: object = ""
        platform = ""
        if groups and groups[0].items:
            base = groups[0].items[0]
            se = _cached_sun_elevation(base.blob_path)
            sun_elevation = "" if se is None else se
            platform = platform_from_scene_id(base.name) or ""
        out_rows.append((crs, col, row, tile_time, image_idx, sun_elevation, platform))
    return out_rows


def _load_tasks(landsat_csv: str) -> dict:
    """Group monthly CSV rows by window -> list of (image_idx, start_time)."""
    tasks: dict = defaultdict(list)
    with open(landsat_csv) as f:
        reader = csv.DictReader(f)
        for r in reader:
            key = (r["crs"], r["col"], r["row"], r["tile_time"])
            tasks[key].append((int(r["image_idx"]), r["start_time"]))
    return tasks


def _load_done_windows(output_csv: str) -> set:
    """Windows already written to the output (for resumability)."""
    done: set = set()
    try:
        with open(output_csv) as f:
            reader = csv.DictReader(f)
            for r in reader:
                done.add((r["crs"], r["col"], r["row"], r["tile_time"]))
    except FileNotFoundError:
        pass
    return done


def main() -> None:
    """Entry point."""
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description="Re-acquire Landsat sun elevation for existing monthly tiles",
    )
    parser.add_argument("--landsat_csv", required=True, help="existing monthly CSV")
    parser.add_argument("--output_csv", required=True, help="output sun-elevation CSV")
    parser.add_argument(
        "--metadata_cache_dir", required=True, help="shared STAC cache dir"
    )
    parser.add_argument(
        "--sun_elevation_cache_dir",
        default=None,
        help="shared per-scene sun elevation cache dir (default: sibling of metadata_cache_dir)",
    )
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument(
        "--limit", type=int, default=0, help="process only first N windows (0=all)"
    )
    args = parser.parse_args()

    sun_cache_dir = args.sun_elevation_cache_dir or str(
        UPath(args.metadata_cache_dir).parent / "mtl_sun_elevation_cache"
    )
    UPath(args.metadata_cache_dir).mkdir(parents=True, exist_ok=True)
    UPath(sun_cache_dir).mkdir(parents=True, exist_ok=True)
    UPath(args.output_csv).parent.mkdir(parents=True, exist_ok=True)

    tasks = _load_tasks(args.landsat_csv)
    done = _load_done_windows(args.output_csv)
    # Sort by key for spatial locality (better STAC cache reuse).
    task_items = sorted(k for k in tasks if k not in done)
    if args.limit:
        task_items = task_items[: args.limit]
    logger.info(
        f"total windows={len(tasks)} already_done={len(done)} to_process={len(task_items)}"
    )
    if not task_items:
        logger.info("nothing to do")
        return

    task_list = [(k, tasks[k]) for k in task_items]

    output_exists = UPath(args.output_csv).exists()
    mode = "a" if output_exists else "w"
    with open(args.output_csv, mode, newline="") as out_f:
        writer = csv.writer(out_f)
        if not output_exists:
            writer.writerow(OUTPUT_COLUMNS)

        with multiprocessing.Pool(
            processes=args.workers,
            initializer=_init_worker,
            initargs=(args.metadata_cache_dir, sun_cache_dir),
        ) as pool:
            for out_rows in tqdm.tqdm(
                pool.imap_unordered(_process_window, task_list, chunksize=8),
                total=len(task_list),
                desc="windows",
            ):
                writer.writerows(out_rows)
                out_f.flush()

    logger.info("done")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver")
    main()
