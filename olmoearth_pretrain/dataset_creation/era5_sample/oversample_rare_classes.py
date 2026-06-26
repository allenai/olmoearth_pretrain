"""Oversample rare Köppen-Geiger classes by adding disjoint temporal windows.

For classes with fewer than a threshold number of windows, this script adds
additional windows at the same spatial locations but with non-overlapping
448-day temporal windows, up to the maximum that fits in [2016-01-01, 2026-05-31].

This is an add-on to the main pipeline — it reads selected.parquet and
writes additional windows directly into the rslearn dataset.

Usage:
    python -m olmoearth_pretrain.dataset_creation.era5_sample.oversample_rare_classes \
        --metadata-dir /weka/dfive-default/helios/dataset/era5enc_pretrain/metadata \
        --output /weka/dfive-default/helios/dataset/era5enc_pretrain/rslearn_dataset \
        --threshold 5000 \
        --max-per-class 5000 \
        --workers 32 \
        --fresh
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
from pyproj import Transformer
from rslearn.utils.get_utm_ups_crs import get_utm_ups_projection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

RESOLUTION = 10
WINDOW_SIZE = 128
WINDOW_DURATION_DAYS = 448

DATE_MIN = datetime(2016, 1, 1, tzinfo=UTC)
DATE_MAX_END = datetime(2026, 5, 31, tzinfo=UTC)

# All possible disjoint start dates within the valid range
_TOTAL_DAYS = (DATE_MAX_END - DATE_MIN).days  # 3803
_MAX_DISJOINT = _TOTAL_DAYS // WINDOW_DURATION_DAYS  # 8


def _all_disjoint_starts() -> list[datetime]:
    """Return all disjoint 448-day window start dates in [2016-01-01, 2026-05-31]."""
    starts = []
    for i in range(_MAX_DISJOINT):
        start = DATE_MIN + timedelta(days=i * WINDOW_DURATION_DAYS)
        end = start + timedelta(days=WINDOW_DURATION_DAYS)
        if end <= DATE_MAX_END:
            starts.append(start)
    return starts


def _compute_window_metadata(job: dict) -> dict:
    """Compute window metadata for a single oversampled window."""
    lon = job["lon"]
    lat = job["lat"]
    name = job["name"]
    start_date = job["start_date"]
    group = job["group"]
    options = job["options"]

    half_m = WINDOW_SIZE * RESOLUTION / 2.0
    utm_proj = get_utm_ups_projection(lon, lat, RESOLUTION, -RESOLUTION)
    transformer = Transformer.from_crs("EPSG:4326", utm_proj.crs, always_xy=True)
    cx, cy = transformer.transform(lon, lat)

    bounds = (
        int((cx - half_m) / utm_proj.x_resolution),
        int((cy + half_m) / utm_proj.y_resolution),
        int((cx + half_m) / utm_proj.x_resolution),
        int((cy - half_m) / utm_proj.y_resolution),
    )

    end_date = start_date + timedelta(days=WINDOW_DURATION_DAYS)

    return {
        "name": name,
        "group": group,
        "metadata": {
            "projection": utm_proj.serialize(),
            "bounds": bounds,
            "time_range": [start_date.isoformat(), end_date.isoformat()],
            "options": options,
        },
    }


def _write_window(result: dict, ds_path: Path, fresh: bool) -> bool:
    """Write a single window metadata.json. Returns True if written."""
    window_dir = ds_path / "windows" / result["group"] / result["name"]
    metadata_path = window_dir / "metadata.json"
    if not fresh and metadata_path.exists():
        return False
    window_dir.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "w") as f:
        json.dump(result["metadata"], f)
    return True


def oversample_rare_classes(
    metadata_dir: str,
    output_path: str,
    threshold: int = 5000,
    max_per_class: int = 5000,
    group: str = "default",
    workers: int = 32,
    fresh: bool = False,
) -> None:
    """Add oversampled windows for rare Köppen classes.

    Args:
        metadata_dir: Path to directory containing selected.parquet.
        output_path: Root path for the rslearn dataset.
        threshold: Classes with fewer total windows than this get oversampled.
        max_per_class: Maximum total windows per class after oversampling.
        group: Window group name.
        workers: Number of parallel workers.
        fresh: If True, skip existence checks.
    """
    selected_path = Path(metadata_dir) / "selected.parquet"
    if not selected_path.exists():
        raise FileNotFoundError(f"selected.parquet not found at {selected_path}")

    df = pd.read_parquet(str(selected_path))
    logger.info("Loaded %d selected points", len(df))

    # Identify rare classes
    class_counts = df["koppen_class"].value_counts()
    rare_classes = class_counts[class_counts < threshold].index.tolist()
    logger.info(
        "Classes below threshold (%d): %s",
        threshold,
        {c: int(class_counts[c]) for c in rare_classes},
    )

    if not rare_classes:
        logger.info("No classes below threshold, nothing to do")
        return

    # Get unique cells for rare classes (use primary/overlap_primary only to avoid dupes)
    primaries = df[
        (df["koppen_class"].isin(rare_classes))
        & (df["role"].isin(["primary", "overlap_primary"]))
    ]

    all_starts = _all_disjoint_starts()
    logger.info(
        "Max disjoint windows per cell: %d (starts: %s to %s)",
        len(all_starts),
        all_starts[0].date(),
        all_starts[-1].date(),
    )

    # Build oversampling jobs per class
    jobs: list[dict] = []
    class_new_counts: dict[str, int] = {}

    for koppen_class in rare_classes:
        class_cells = primaries[primaries["koppen_class"] == koppen_class]
        existing_count = int(class_counts[koppen_class])
        budget = max_per_class - existing_count

        if budget <= 0:
            logger.info("  %s: already at %d, skipping", koppen_class, existing_count)
            continue

        # Unique cells for this class
        unique_cells = class_cells.drop_duplicates(subset="era5_cell_id")
        n_cells = len(unique_cells)

        # How many temporal slots per cell do we need?
        # Distribute budget across cells evenly, up to max_disjoint
        slots_per_cell = min(len(all_starts), max(1, budget // n_cells + 1))
        added = 0

        for _, row in unique_cells.iterrows():
            if added >= budget:
                break

            for slot_idx, start_date in enumerate(all_starts[:slots_per_cell]):
                if added >= budget:
                    break

                name = (
                    f"era5s_rare_{koppen_class}_{int(row['era5_cell_id'])}_{slot_idx}"
                )
                jobs.append(
                    {
                        "lon": float(row["lon"]),
                        "lat": float(row["lat"]),
                        "name": name,
                        "start_date": start_date,
                        "group": group,
                        "options": {
                            "role": "rare_oversample",
                            "pair_id": -1,
                            "koppen_class": koppen_class,
                            "elev_band_label": row["elev_band_label"],
                            "lat_band_label": row["lat_band_label"],
                            "era5_cell_id": int(row["era5_cell_id"]),
                            "oversample_slot": slot_idx,
                        },
                    }
                )
                added += 1

        class_new_counts[koppen_class] = added
        logger.info(
            "  %s: %d existing + %d new = %d total (from %d cells × up to %d slots)",
            koppen_class,
            existing_count,
            added,
            existing_count + added,
            n_cells,
            slots_per_cell,
        )

    logger.info("Total oversampling jobs: %d", len(jobs))

    if not jobs:
        logger.info("Nothing to oversample")
        return

    # Generate windows using multiprocessing
    ds_path = Path(output_path)
    saved = 0
    t0 = time.time()
    batch_size = 5000

    with multiprocessing.Pool(workers) as pool:
        for batch_start in range(0, len(jobs), batch_size):
            batch = jobs[batch_start : batch_start + batch_size]
            results = pool.map(_compute_window_metadata, batch, chunksize=64)

            for result in results:
                if _write_window(result, ds_path, fresh):
                    saved += 1

            elapsed = time.time() - t0
            logger.info(
                "  Written %d / %d (%.0fs elapsed)",
                saved,
                len(jobs),
                elapsed,
            )

    logger.info(
        "Done: %d oversampled windows written in %.0fs",
        saved,
        time.time() - t0,
    )

    # Summary
    logger.info("--- Oversampling Summary ---")
    for koppen_class, new_count in class_new_counts.items():
        existing = int(class_counts[koppen_class])
        logger.info(
            "  %s: %d -> %d windows",
            koppen_class,
            existing,
            existing + new_count,
        )


def main() -> None:
    """Run the rare-class oversampling CLI."""
    parser = argparse.ArgumentParser(
        description="Oversample rare Köppen classes with disjoint temporal windows."
    )
    parser.add_argument(
        "--metadata-dir",
        type=str,
        required=True,
        help="Directory containing selected.parquet.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Root path for the rslearn dataset.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=5000,
        help="Classes with fewer windows than this get oversampled (default: 5000).",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=5000,
        help="Maximum total windows per class after oversampling (default: 5000).",
    )
    parser.add_argument(
        "--group",
        type=str,
        default="default",
        help="Window group name (default: 'default').",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=32,
        help="Number of parallel workers (default: 32).",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Skip existence checks (faster for first-time runs).",
    )
    args = parser.parse_args()

    oversample_rare_classes(
        metadata_dir=args.metadata_dir,
        output_path=args.output,
        threshold=args.threshold,
        max_per_class=args.max_per_class,
        group=args.group,
        workers=args.workers,
        fresh=args.fresh,
    )


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver")
    main()
