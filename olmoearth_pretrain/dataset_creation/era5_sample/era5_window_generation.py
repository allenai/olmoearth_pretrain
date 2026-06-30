"""Generate rslearn windows for ERA5 climate-stratified sampling.

Reads selected.parquet (from stratified_sample.py) and creates one rslearn
Window per selected point. Each window is 128x128 pixels at 10 m/px in the
local UTM projection. Temporal windows are 448 days, uniformly sampled within
[2016-01-01, 2025-03-09] for primaries, and offset 348 days from their paired
primary for overlap_secondary windows (guaranteeing 100-day overlap).

Parallel-safe + batched + resumable: existing windows are skipped (unless --fresh).

Usage:
    python -m olmoearth_pretrain.dataset_creation.era5_sample.era5_window_generation \
        --metadata-dir /weka/dfive-default/helios/dataset/era5enc_pretrain/metadata \
        --output /weka/dfive-default/helios/dataset/era5enc_pretrain/rslearn_dataset \
        --seed 42 \
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

import numpy as np
import pandas as pd
from pyproj import Transformer
from rslearn.utils.get_utm_ups_crs import get_utm_ups_projection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

RESOLUTION = 10  # meters per pixel
WINDOW_SIZE = 128  # pixels (128 * 10m = 1.28 km)
WINDOW_DURATION_DAYS = 448

# Temporal bounds: windows must lie fully within [2016-01-01, 2026-05-31]
# => latest start = 2026-05-31 - 448 days = 2025-03-09
DATE_MIN = datetime(2016, 1, 1, tzinfo=UTC)
DATE_MAX_START = datetime(2025, 3, 9, tzinfo=UTC)

# Overlap pair: offset such that exactly 100 days overlap with the primary's
# 448-day window. Primary=[s, s+448d], secondary=[s+348d, s+348d+448d]
# Overlap = [s+348d, s+448d] = 100 days.
OVERLAP_OFFSET_DAYS = 348

IO_WORKERS = 32
BATCH_SIZE = 10000


def _sample_start_date(rng: np.random.Generator) -> datetime:
    """Sample a random start date uniformly in [DATE_MIN, DATE_MAX_START]."""
    total_days = (DATE_MAX_START - DATE_MIN).days
    offset_days = int(rng.integers(0, total_days + 1))
    return DATE_MIN + timedelta(days=offset_days)


def _compute_overlap_start(
    primary_start: datetime, rng: np.random.Generator
) -> datetime:
    """Compute secondary window start: primary_start + 348 days.

    If that would exceed DATE_MAX_START, shift backwards instead (primary_start - 348d).
    Guarantees exactly 100-day temporal overlap.
    """
    forward = primary_start + timedelta(days=OVERLAP_OFFSET_DAYS)
    if forward <= DATE_MAX_START:
        return forward

    backward = primary_start - timedelta(days=OVERLAP_OFFSET_DAYS)
    if backward >= DATE_MIN:
        return backward

    return max(DATE_MIN, min(forward, DATE_MAX_START))


def _compute_window_metadata(job: dict) -> dict:
    """Compute the full metadata dict for a single window (picklable worker function).

    Returns a dict with keys: name, group, projection_json, bounds, time_range_json, options.
    """
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

    min_x = cx - half_m
    min_y = cy - half_m
    max_x = cx + half_m
    max_y = cy + half_m

    bounds = (
        int(min_x / utm_proj.x_resolution),
        int(max_y / utm_proj.y_resolution),
        int(max_x / utm_proj.x_resolution),
        int(min_y / utm_proj.y_resolution),
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
    """Write a single window's metadata.json to disk. Returns True if written."""
    group = result["group"]
    name = result["name"]
    window_dir = ds_path / "windows" / group / name
    metadata_path = window_dir / "metadata.json"

    if not fresh and metadata_path.exists():
        return False

    window_dir.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "w") as f:
        json.dump(result["metadata"], f)
    return True


def generate_windows(
    metadata_dir: str,
    output_path: str,
    group: str = "default",
    seed: int = 42,
    workers: int = IO_WORKERS,
    batch_size: int = BATCH_SIZE,
    fresh: bool = False,
) -> None:
    """Create rslearn windows from the stratified selection.

    Args:
        metadata_dir: Path to directory containing selected.parquet.
        output_path: Root path for the rslearn dataset.
        group: Window group name.
        seed: Random seed for temporal sampling.
        workers: Number of parallel workers.
        batch_size: Windows per processing batch.
        fresh: If True, skip existence checks (faster for first-time runs).
    """
    selected_path = Path(metadata_dir) / "selected.parquet"
    if not selected_path.exists():
        raise FileNotFoundError(
            f"selected.parquet not found at {selected_path}. "
            "Run stratified_sample.py first."
        )

    df = pd.read_parquet(str(selected_path))
    logger.info("Loaded %d selected points", len(df))

    ds_path = Path(output_path)
    rng = np.random.default_rng(seed)

    # Pre-draw start dates for all primaries (includes overlap_primary)
    primaries_mask = df["role"].isin(["primary", "overlap_primary"])
    n_primaries = primaries_mask.sum()

    all_start_days = rng.integers(
        0, (DATE_MAX_START - DATE_MIN).days + 1, size=n_primaries
    )
    primary_starts: dict[int, datetime] = {}
    start_idx = 0
    for df_idx in df.index[primaries_mask]:
        start = DATE_MIN + timedelta(days=int(all_start_days[start_idx]))
        primary_starts[df_idx] = start
        start_idx += 1

    # Build a mapping from pair_id to primary start date (for secondaries)
    pair_to_start: dict[int, datetime] = {}
    for df_idx, start in primary_starts.items():
        pair_id = df.loc[df_idx, "pair_id"]
        if pair_id >= 0:
            pair_to_start[pair_id] = start

    # Prepare job list
    jobs: list[dict] = []
    for df_idx, row in df.iterrows():
        role = row["role"]
        pair_id = int(row["pair_id"])

        if role in ("primary", "overlap_primary"):
            start_date = primary_starts[df_idx]
            name = f"era5s_{df_idx:06d}"
        else:
            primary_start = pair_to_start.get(pair_id)
            if primary_start is None:
                continue
            start_date = _compute_overlap_start(primary_start, rng)
            name = f"era5s_{df_idx:06d}_ov"

        options = {
            "role": role,
            "pair_id": pair_id,
            "koppen_class": row["koppen_class"],
            "elev_band_label": row["elev_band_label"],
            "lat_band_label": row["lat_band_label"],
            "era5_cell_id": int(row["era5_cell_id"]),
        }

        jobs.append(
            {
                "lon": float(row["lon"]),
                "lat": float(row["lat"]),
                "name": name,
                "start_date": start_date,
                "group": group,
                "options": options,
            }
        )

    rng.shuffle(jobs)
    logger.info(
        "Prepared %d window jobs (fresh=%s, workers=%d)", len(jobs), fresh, workers
    )

    # Process with multiprocessing pool: compute metadata in parallel, write sequentially
    saved = 0
    skipped = 0
    t0 = time.time()

    with multiprocessing.Pool(workers) as pool:
        for batch_start in range(0, len(jobs), batch_size):
            batch = jobs[batch_start : batch_start + batch_size]

            # Parallel computation of window metadata (UTM projections + transforms)
            results = pool.map(_compute_window_metadata, batch, chunksize=64)

            # Write to disk (I/O bound, fast per-window)
            batch_saved = 0
            for result in results:
                if _write_window(result, ds_path, fresh):
                    batch_saved += 1
                else:
                    skipped += 1

            saved += batch_saved
            elapsed = time.time() - t0
            remaining = len(jobs) - (batch_start + len(batch))
            rate = saved / elapsed if elapsed > 0 else 0
            eta_s = remaining / rate if rate > 0 else 0
            logger.info(
                "Batch %d–%d: %d saved, %d skipped, %d remaining "
                "(%.0fs elapsed, %.0f win/s, ETA %.0fs)",
                batch_start,
                batch_start + len(batch) - 1,
                saved,
                skipped,
                remaining,
                elapsed,
                rate,
                eta_s,
            )

    logger.info(
        "Done: %d windows saved, %d skipped, %.0fs total",
        saved,
        skipped,
        time.time() - t0,
    )


def main() -> None:
    """Run the rslearn window generation CLI."""
    parser = argparse.ArgumentParser(
        description="Generate rslearn windows for ERA5 climate-stratified sampling."
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
        help="Root path for the rslearn dataset (FileWindowStorage).",
    )
    parser.add_argument(
        "--group",
        type=str,
        default="default",
        help="Window group name (default: 'default').",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for temporal sampling (default: 42).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=IO_WORKERS,
        help=f"Number of parallel workers (default: {IO_WORKERS}).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Windows per batch (default: {BATCH_SIZE}).",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Skip existence checks (faster for first-time generation).",
    )
    args = parser.parse_args()

    generate_windows(
        metadata_dir=args.metadata_dir,
        output_path=args.output,
        group=args.group,
        seed=args.seed,
        workers=args.workers,
        batch_size=args.batch_size,
        fresh=args.fresh,
    )


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver")
    main()
