"""Sample paired pre/post pretraining points from LCC model summary rasters.

Scans the 9-band uint8 summary GeoTIFFs written by the
``rslp.change_finder_v2.lcc_model`` prediction pipeline
(``{EPSG}_{col}_{row}_summary.tif``, 10 m/pixel UTM tiles) and samples
128x128-pixel windows where the model's POST change-category head predicts a
real change: a grid cell is accepted when at least ``--min_change_pixels``
pixels have a ``post_class`` argmax that is a real category (not "none"; 0
means no prediction).

Each accepted cell becomes one sparse point in the open-set label bank format
(``{datasets_root}/{slug}/points.geojson``), with ``pre_time_range`` /
``post_time_range`` derived from the median predicted change start/end months
(``ts_pre_month`` / ``ts_post_month``) over the cell's change pixels: the pre
range ends when the change starts and the post range begins the month after
the change completes, each spanning 180 days (six ~30-day mosaics per side, so
a merged pre+post series stays within the 12-timestep training cap).

The output feeds directly into the paired-window pipeline:

    python -m olmoearth_pretrain.dataset_creation.create_windows.from_open_set \
        --ds_path <new rslearn dataset> \
        --datasets_root <this script's --datasets_root> \
        --slugs olmoearth_lcc_change_predictions \
        --exclude_geojson data/open_set_segmentation_data/eval_exclusion.geojson

The slug is not part of the open-set label bank (separate --datasets_root, no
registry entry) and is not in the frozen class mapping, so from_open_set
writes an all-nodata label layer for it -- fine for SSL-only training.
"""

import argparse
import hashlib
import logging
import multiprocessing
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import rasterio
import shapely
import tqdm
from rslearn.const import WGS84_PROJECTION
from rslearn.utils.geometry import STGeometry
from rslearn.utils.mp import StarImapUnorderedWrapper
from rslearn.utils.raster_format import get_raster_projection_and_bounds
from upath import UPath

from .io import write_dataset_metadata, write_points_table
from .pretrain_constants import OPEN_SET_WINDOW_SIZE

logger = logging.getLogger(__name__)

SLUG = "olmoearth_lcc_change_predictions"

DEFAULT_SUMMARY_PATH = (
    "/weka/dfive-default/rslearn-eai/datasets/change_finder/"
    "lcc_model_outputs_20260826_big"
)

# ---------------------------------------------------------------------------
# Summary raster layout, copied from rslp.change_finder_v2.lcc_model.postprocess
# (SUMMARY_BANDS / POST_CHANGE_CATEGORY_NAMES / NONE_CATEGORY_IDX /
# TIMESTAMP_EPOCH). Keep in sync with that module.
# ---------------------------------------------------------------------------

NUM_SUMMARY_BANDS = 9
# 1-based rasterio band indices into SUMMARY_BANDS.
POST_CLASS_BAND = 3
TS_PRE_MONTH_BAND = 8
TS_POST_MONTH_BAND = 9

# In the class bands, 0 = no prediction and 1 = the "none" category; a real
# change category is any value > NONE_CATEGORY_IDX.
NONE_CATEGORY_IDX = 1

POST_CHANGE_CATEGORY_NAMES = [
    "nodata",
    "none",
    "vegetation_growth",
    "new_building",
    "new_road",
    "new_infrastructure",
    "new_crop_field",
    "new_aquafarm",
    "site_clearing",
    "water_expand",
    "mining",
    "new_crop_structure",
    "selective_logging",
    "landslide",
    "settlement",
]

# Month values in the ts_*_month bands are 0 for no prediction, else 1 + whole
# calendar months since this epoch.
MONTH_EPOCH = datetime(2015, 1, 1, tzinfo=UTC)

# Duration of each of the pre/post observation windows: 6 thirty-day mosaic
# periods per side, so the merged pre+post series is at most 12 timesteps
# (see MAX_PAIRED_WINDOW_DAYS in create_windows.from_open_set).
OBSERVATION_DAYS = 180


def month_value_to_date(month_value: int) -> datetime:
    """Decode a summary month value to the first day of that month (UTC).

    month_value is 1 + whole calendar months since MONTH_EPOCH.
    """
    months_since_epoch = int(month_value) - 1
    year = MONTH_EPOCH.year + months_since_epoch // 12
    month = 1 + months_since_epoch % 12
    return datetime(year, month, 1, tzinfo=UTC)


def count_change_pixels_per_cell(post_class: np.ndarray, cell_size: int) -> np.ndarray:
    """Count post-head change pixels in each non-overlapping cell_size grid cell.

    A change pixel is one whose post_class argmax is a real category (value >
    NONE_CATEGORY_IDX). Rows/columns beyond the last full cell are ignored.

    Returns:
        (H // cell_size, W // cell_size) int array of change-pixel counts.
    """
    grid_h = post_class.shape[0] // cell_size
    grid_w = post_class.shape[1] // cell_size
    change = post_class[: grid_h * cell_size, : grid_w * cell_size]
    change = change > NONE_CATEGORY_IDX
    return change.reshape(grid_h, cell_size, grid_w, cell_size).sum(axis=(1, 3))


def _tile_rng(seed: int, tile_name: str) -> np.random.Generator:
    """A per-tile RNG that is stable regardless of tile processing order."""
    digest = hashlib.sha256(f"{seed}_{tile_name}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def _cell_slice(cell_row: int, cell_col: int, cell_size: int) -> tuple[slice, slice]:
    """Array slices for a grid cell."""
    return (
        slice(cell_row * cell_size, (cell_row + 1) * cell_size),
        slice(cell_col * cell_size, (cell_col + 1) * cell_size),
    )


def _dominant_category(cell_post_class: np.ndarray) -> int:
    """The most common real change category among a cell's change pixels."""
    change_values = cell_post_class[cell_post_class > NONE_CATEGORY_IDX]
    return int(np.bincount(change_values).argmax())


def process_tile(
    tif_path: str,
    min_change_pixels: int,
    max_per_tile: int,
    seed: int,
    cell_size: int = OPEN_SET_WINDOW_SIZE,
) -> tuple[list[dict[str, Any]], int]:
    """Sample accepted change cells from one summary tile.

    The max_per_tile cap is applied per dominant change category rather than to
    the tile as a whole, so a tile dominated by one common change type (e.g.
    new crop fields) still contributes all of its rarer changes.

    Returns:
        (points, num_accepted_cells): the sampled point dicts (at most
        max_per_tile per dominant category, in the write_points_table format)
        and the number of grid cells in the tile that passed the change-pixel
        threshold before the cap.
    """
    path = UPath(tif_path)
    with path.open("rb") as f:
        with rasterio.open(f) as src:
            if src.count != NUM_SUMMARY_BANDS:
                raise ValueError(
                    f"{path} has {src.count} bands, expected {NUM_SUMMARY_BANDS}"
                )
            projection, bounds = get_raster_projection_and_bounds(src)
            post_class = src.read(POST_CLASS_BAND)

            counts = count_change_pixels_per_cell(post_class, cell_size)
            accepted = np.argwhere(counts >= min_change_pixels)
            num_accepted = len(accepted)
            if num_accepted == 0:
                return [], 0

            # Compute each accepted cell's dominant category up front so the
            # cap can be applied per (tile, category).
            cell_categories = np.array(
                [
                    _dominant_category(post_class[_cell_slice(r, c, cell_size)])
                    for r, c in accepted
                ]
            )
            rng = _tile_rng(seed, path.name)
            selected: list[int] = []
            for category_idx in np.unique(cell_categories):
                (idxs,) = np.nonzero(cell_categories == category_idx)
                if len(idxs) > max_per_tile:
                    idxs = rng.choice(idxs, size=max_per_tile, replace=False)
                selected.extend(int(i) for i in idxs)
            selected.sort()
            accepted = accepted[selected]
            cell_categories = cell_categories[selected]

            # Only read the month bands once we know the tile has samples.
            pre_months = src.read(TS_PRE_MONTH_BAND)
            post_months = src.read(TS_POST_MONTH_BAND)

    epsg = projection.crs.to_epsg()
    crs_id = str(epsg) if epsg is not None else projection.crs.to_string()

    points: list[dict[str, Any]] = []
    for (cell_row, cell_col), category_idx in zip(accepted, cell_categories):
        sl = _cell_slice(cell_row, cell_col, cell_size)
        cell_post_class = post_class[sl]
        change = cell_post_class > NONE_CATEGORY_IDX

        pre_m = pre_months[sl][change]
        pre_m = pre_m[pre_m > 0]
        post_m = post_months[sl][change]
        post_m = post_m[post_m > 0]
        if pre_m.size == 0 or post_m.size == 0:
            # No usable change-date prediction in this cell (rare).
            continue

        # The pre window ends when the change starts (first day of the median
        # predicted start month); the post window starts the month AFTER the
        # change completes, so its imagery is fully post-change.
        pre_end = month_value_to_date(int(np.median(pre_m)))
        post_start = month_value_to_date(int(np.median(post_m)) + 1)
        if post_start < pre_end:
            post_start = pre_end
        pre_time_range = (pre_end - timedelta(days=OBSERVATION_DAYS), pre_end)
        post_time_range = (post_start, post_start + timedelta(days=OBSERVATION_DAYS))

        # Dominant predicted change category among the cell's change pixels
        # (used for the per-category cap and distribution reporting, not as a
        # training label).
        if category_idx < len(POST_CHANGE_CATEGORY_NAMES):
            category = POST_CHANGE_CATEGORY_NAMES[category_idx]
        else:
            category = str(category_idx)

        # Absolute pixel coords of the cell within the tile projection; the
        # sample point is the cell center, which from_open_set re-centers a
        # 128x128 UTM window on.
        col0 = bounds[0] + int(cell_col) * cell_size
        row0 = bounds[1] + int(cell_row) * cell_size
        center = shapely.Point(col0 + cell_size / 2, row0 + cell_size / 2)
        geom = STGeometry(projection, center, None).to_projection(WGS84_PROJECTION)

        points.append(
            {
                "id": f"{crs_id}_{col0}_{row0}",
                "lon": float(geom.shp.x),
                "lat": float(geom.shp.y),
                "label": None,
                "time_range": None,
                "pre_time_range": pre_time_range,
                "post_time_range": post_time_range,
                "num_change_pixels": int(counts[cell_row, cell_col]),
                "post_category": category,
                "tile": path.name,
            }
        )
    return points, num_accepted


def main() -> None:
    """CLI entrypoint."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    multiprocessing.set_start_method("forkserver")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary_path",
        type=str,
        default=DEFAULT_SUMMARY_PATH,
        help="Directory containing the *_summary.tif tiles",
    )
    parser.add_argument(
        "--datasets_root",
        type=str,
        required=True,
        help=(
            "Root directory to write the sampled dataset under "
            f"({{datasets_root}}/{SLUG}/points.geojson); pass the same root to "
            "from_open_set --datasets_root"
        ),
    )
    parser.add_argument(
        "--target_samples",
        type=int,
        default=100_000,
        help="Number of points to keep (random subset of all accepted cells)",
    )
    parser.add_argument(
        "--min_change_pixels",
        type=int,
        default=100,
        help="Minimum post-head change pixels for a cell to be accepted",
    )
    parser.add_argument(
        "--max_per_tile",
        type=int,
        default=2000,
        help=(
            "Maximum sampled cells per tile per dominant change category "
            "(limits geographic concentration without crowding out rare "
            "categories)"
        ),
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    summary_root = UPath(args.summary_path)
    tif_paths = sorted(summary_root.glob("*_summary.tif"))
    if not tif_paths:
        raise ValueError(f"no *_summary.tif files found under {args.summary_path}")
    logger.info("found %d summary tiles under %s", len(tif_paths), args.summary_path)

    jobs = [
        dict(
            tif_path=str(p),
            min_change_pixels=args.min_change_pixels,
            max_per_tile=args.max_per_tile,
            seed=args.seed,
        )
        for p in tif_paths
    ]

    points: list[dict[str, Any]] = []
    total_accepted = 0
    tiles_with_samples = 0
    p = multiprocessing.Pool(args.workers)
    outputs = p.imap_unordered(StarImapUnorderedWrapper(process_tile), jobs)
    for tile_points, num_accepted in tqdm.tqdm(
        outputs, total=len(jobs), desc="sampling tiles"
    ):
        total_accepted += num_accepted
        if tile_points:
            tiles_with_samples += 1
            points.extend(tile_points)
    p.close()
    p.join()

    logger.info(
        "%d cells accepted across %d/%d tiles; "
        "%d sampled after the per-tile per-category cap",
        total_accepted,
        tiles_with_samples,
        len(tif_paths),
        len(points),
    )

    # Deterministic global subset: sort by id (pool order is nondeterministic),
    # then shuffle with the global seed and truncate.
    points.sort(key=lambda pt: pt["id"])
    rng = np.random.default_rng(args.seed)
    rng.shuffle(points)  # type: ignore[arg-type]
    if len(points) > args.target_samples:
        points = points[: args.target_samples]
    elif len(points) < args.target_samples:
        logger.warning(
            "only %d points available, below the target of %d",
            len(points),
            args.target_samples,
        )

    category_counts: dict[str, int] = {}
    for pt in points:
        category_counts[pt["post_category"]] = (
            category_counts.get(pt["post_category"], 0) + 1
        )
    for category, count in sorted(category_counts.items(), key=lambda kv: -kv[1]):
        logger.info("  %s: %d (%.1f%%)", category, count, 100 * count / len(points))

    root = UPath(args.datasets_root)
    write_points_table(SLUG, "change", points, root=root)
    write_dataset_metadata(
        SLUG,
        {
            "name": "OlmoEarth LCC change predictions",
            "task_type": "change",
            "source": args.summary_path,
            "min_change_pixels": args.min_change_pixels,
            "max_per_tile": args.max_per_tile,
            "seed": args.seed,
            "count": len(points),
            "notes": (
                "128x128 windows sampled where the LCC model's post change head "
                "predicts a real category; pre/post time ranges from the median "
                "predicted change start/end months. max_per_tile applies per "
                "tile per dominant category. Model-derived (no human labels); "
                "intended for SSL pretraining via paired windows."
            ),
        },
        root=root,
    )
    logger.info("wrote %d points to %s", len(points), root / SLUG / "points.geojson")


if __name__ == "__main__":
    main()
