"""Landslide ERA5 window dedup — leakage-safe preliminary step.

Scans metadata from the landslide rslearn dataset (sen12_landslides + icimod +
glc), groups windows by (ERA5-Land 0.1° cell, ISO year-week of each window's own
time_range_start), and collapses each conflicting bucket to a single window.

GLC (Global Landslide Catalog) is included because it carries real, precise
event dates and explicit weather triggers (rain / continuous_rain / downpour /
tropical_cyclone / ...) — exactly the signal an ERA5 eval needs. Its lack of
polygons (Piper's reason for down-rating it) is irrelevant for point-based
ERA5 classification. fwn_mtli is intentionally excluded: all its events use a
placeholder July-1 date with no real timestamp, so it cannot be temporally
aligned to the triggering weather.

Produces:
  all_windows.parquet   — cached raw scan (one row per window).
  deduped_windows.parquet — surviving windows after dedup.
  dedup_summary.md      — before/after distribution report.

Usage:
    python -m olmoearth_pretrain.evals.datasets.splits.landslide_era5.dedup_windows \
        --ds-root /weka/dfive-default/piperw/rslearn_projects/data/landslide/sen12landslides/all_positives \
        --out-dir olmoearth_pretrain/olmoearth_pretrain/evals/datasets/splits/landslide_era5 \
        --rescan --workers 1
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ERA5-Land grid helpers (nearest-node, matching the data source registration)
# ---------------------------------------------------------------------------

ERA5_DEG = 0.1
N_LON_CELLS = 3600  # 360 / 0.1


def era5_nearest_cell_id(lat: float, lon: float) -> int:
    """Snap (lat, lon) to the nearest ERA5-Land 0.1° node and return a unique int.

    ERA5-Land nodes sit at integer multiples of 0.1° (pixel centers).
    A sub-cell window resolves to the nearest node — so we round, not floor.
    This matches ``ERA5LandDailyUTCv1._compute_projection_and_bounds`` in
    ``rslearn/rslearn/data_sources/earthdatahub.py``.
    """
    lon_norm = ((lon + 180.0) % 360.0) - 180.0
    lat_idx = round(lat / ERA5_DEG)
    lon_idx = round(lon_norm / ERA5_DEG)
    return lat_idx * N_LON_CELLS + lon_idx


# ---------------------------------------------------------------------------
# Latitude bands (reused from build_candidate_grid.py for the summary)
# ---------------------------------------------------------------------------

LAT_BAND_EDGES = [-90, -60, -30, 0, 30, 60, 90]
LAT_BAND_LABELS = ["[-90,-60)", "[-60,-30)", "[-30,0)", "[0,30)", "[30,60)", "[60,90]"]


def lat_to_band(lat: float) -> str:
    for i in range(len(LAT_BAND_EDGES) - 1):
        lo, hi = LAT_BAND_EDGES[i], LAT_BAND_EDGES[i + 1]
        if lo <= lat < hi or (i == len(LAT_BAND_EDGES) - 2 and lat == hi):
            return LAT_BAND_LABELS[i]
    return "unknown"


# ---------------------------------------------------------------------------
# Metadata scanner
# ---------------------------------------------------------------------------

_FIELDS = [
    "latitude",
    "longitude",
    "event_date",
    "time_range_start",
    "event_year",
    "event_type",
    "location",
    "window_type",
    "num_overlapping_landslides",
    "split",
    "is_cloudy",
]

# Reasonable year range for landslide events with Sentinel-era imagery
_MIN_YEAR = 2014
_MAX_YEAR = 2026


def _parse_one_window(args: tuple[str, str, str]) -> dict[str, Any] | None:
    """Read a single window's metadata.json and extract the fields we need.

    Returns None if the file is missing or has an unparseable date.
    """
    group, window_name, meta_path = args
    try:
        with open(meta_path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    opts = data.get("options", {})
    row: dict[str, Any] = {"group": group, "window_name": window_name}

    for field in _FIELDS:
        row[field] = opts.get(field)

    # Parse time_range_start → iso_year, iso_week
    trs = row.get("time_range_start")
    if not trs:
        return None
    try:
        dt = datetime.fromisoformat(trs)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        iso_cal = dt.isocalendar()
        row["iso_year"] = iso_cal[0]
        row["iso_week"] = iso_cal[1]
    except (ValueError, TypeError):
        return None

    # Validate event_year
    ey = row.get("event_year")
    if ey is not None and not (_MIN_YEAR <= ey <= _MAX_YEAR):
        return None

    # Compute ERA5 cell id
    lat, lon = row.get("latitude"), row.get("longitude")
    if lat is None or lon is None:
        return None
    row["era5_cell_id"] = era5_nearest_cell_id(lat, lon)

    return row


def scan_windows(
    ds_root: Path,
    groups: list[str],
    workers: int = 32,
) -> pd.DataFrame:
    """Scan all metadata.json files for the given groups and return a DataFrame.

    Uses sequential I/O: on network filesystems (weka) the bottleneck is IOPS,
    not CPU, so multiprocessing adds overhead without throughput gain.
    Falls back to ``multiprocessing.Pool`` only when running on local disk
    (toggled via ``workers > 1`` and ``--workers 1`` being the default now).
    """
    jobs: list[tuple[str, str, str]] = []
    for group in groups:
        group_dir = ds_root / "windows" / group
        if not group_dir.is_dir():
            logger.warning("Group directory not found: %s", group_dir)
            continue
        for entry in os.scandir(group_dir):
            if entry.is_dir():
                meta_path = os.path.join(entry.path, "metadata.json")
                jobs.append((group, entry.name, meta_path))

    logger.info("Scanning %d windows across groups %s ...", len(jobs), groups)
    t0 = time.time()

    rows: list[dict[str, Any]] = []
    dropped = 0
    report_every = 10000

    if workers > 1:
        with multiprocessing.Pool(workers) as pool:
            results = pool.map(_parse_one_window, jobs, chunksize=256)
        for r in results:
            if r is not None:
                rows.append(r)
            else:
                dropped += 1
    else:
        for i, job in enumerate(jobs):
            r = _parse_one_window(job)
            if r is not None:
                rows.append(r)
            else:
                dropped += 1
            if (i + 1) % report_every == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (len(jobs) - i - 1) / rate if rate > 0 else 0
                logger.info(
                    "  ... %d / %d scanned (%.0f/s, ETA %.0fs)",
                    i + 1, len(jobs), rate, eta,
                )

    elapsed = time.time() - t0
    logger.info(
        "Scanned %d windows in %.1fs — %d valid, %d dropped (bad date / missing fields)",
        len(jobs), elapsed, len(rows), dropped,
    )

    df = pd.DataFrame(rows)
    for col in ("num_overlapping_landslides", "event_year", "era5_cell_id", "iso_year", "iso_week"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    if "is_cloudy" in df.columns:
        df["is_cloudy"] = df["is_cloudy"].astype("boolean")

    return df


# ---------------------------------------------------------------------------
# Dedup
# ---------------------------------------------------------------------------


def _tiebreak_key(row: pd.Series) -> tuple:
    """Sort key for picking the best window in a bucket (highest wins)."""
    is_curated_val = 1 if row.get("split") == "val" else 0
    overlap = int(row.get("num_overlapping_landslides") or 0)
    # Deterministic fallback: window_name ascending → flip for descending sort
    return (is_curated_val, overlap)


def dedup_windows(df: pd.DataFrame) -> pd.DataFrame:
    """Group by (era5_cell_id, iso_year, iso_week) and collapse conflicts.

    - Buckets with ≥1 positive: keep 1 positive (best tiebreak), drop rest.
    - Negative-only buckets: keep 1 negative (deterministic by window_name).
    """
    bucket_col = "bucket_id"
    df = df.copy()
    df[bucket_col] = (
        df["era5_cell_id"].astype(str)
        + "_"
        + df["iso_year"].astype(str)
        + "_"
        + df["iso_week"].astype(str)
    )

    survivors: list[pd.Series] = []
    n_buckets_pos = 0
    n_buckets_neg = 0
    n_dropped_from_pos_buckets = 0
    n_dropped_from_neg_buckets = 0

    for _bucket_id, bucket_df in df.groupby(bucket_col):
        positives = bucket_df[bucket_df["window_type"] == "positive"]

        if len(positives) > 0:
            n_buckets_pos += 1
            # Pick best positive by tiebreak
            best_idx = max(
                positives.index,
                key=lambda idx: (
                    _tiebreak_key(positives.loc[idx]),
                    positives.loc[idx, "window_name"],
                ),
            )
            survivors.append(bucket_df.loc[best_idx])
            n_dropped_from_pos_buckets += len(bucket_df) - 1
        else:
            n_buckets_neg += 1
            # Pick deterministic negative
            best_idx = min(bucket_df.index, key=lambda idx: bucket_df.loc[idx, "window_name"])
            survivors.append(bucket_df.loc[best_idx])
            n_dropped_from_neg_buckets += len(bucket_df) - 1

    logger.info(
        "Dedup: %d buckets (%d with positives, %d negative-only). "
        "Dropped %d from positive buckets, %d from negative-only buckets.",
        n_buckets_pos + n_buckets_neg,
        n_buckets_pos,
        n_buckets_neg,
        n_dropped_from_pos_buckets,
        n_dropped_from_neg_buckets,
    )

    result = pd.DataFrame(survivors).reset_index(drop=True)
    # Stable sort for determinism
    result = result.sort_values(["group", "window_name"]).reset_index(drop=True)
    return result


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------


def _count_table(before: pd.DataFrame, after: pd.DataFrame, col: str) -> str:
    """Markdown table of before/after value counts for a column."""
    bc = before[col].value_counts().sort_index()
    ac = after[col].value_counts().sort_index()
    all_vals = sorted(set(bc.index) | set(ac.index), key=str)

    lines = [f"| {col} | before | after | delta |", "|---|---|---|---|"]
    for v in all_vals:
        b = int(bc.get(v, 0))
        a = int(ac.get(v, 0))
        lines.append(f"| {v} | {b:,} | {a:,} | {a - b:+,} |")
    # Totals
    lines.append(f"| **total** | **{len(before):,}** | **{len(after):,}** | **{len(after) - len(before):+,}** |")
    return "\n".join(lines)


def generate_summary(before: pd.DataFrame, after: pd.DataFrame, out_path: Path) -> None:
    """Write dedup_summary.md with before/after distribution tables."""
    before = before.copy()
    after = after.copy()

    # Add lat_band column
    before["lat_band"] = before["latitude"].apply(lat_to_band)
    after["lat_band"] = after["latitude"].apply(lat_to_band)

    # Curated val counts
    val_before = int((before["split"] == "val").sum())
    val_after = int((after["split"] == "val").sum())

    sections = [
        "# Landslide ERA5 Window Dedup — Distribution Report\n",
        f"**Before**: {len(before):,} windows | **After**: {len(after):,} windows | "
        f"**Removed**: {len(before) - len(after):,}\n",
        f"**Curated val windows**: {val_before} before → {val_after} after\n",
        "## By window_type (positive / negative)\n",
        _count_table(before, after, "window_type"),
        "\n## By group\n",
        _count_table(before, after, "group"),
        "\n## By latitude band\n",
        _count_table(before, after, "lat_band"),
        "\n## By event_year\n",
        _count_table(before, after, "event_year"),
        "\n## By event_type\n",
        _count_table(before, after, "event_type"),
    ]

    text = "\n".join(sections) + "\n"
    out_path.write_text(text)
    logger.info("Summary written to %s", out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DEFAULT_DS_ROOT = "/weka/dfive-default/piperw/rslearn_projects/data/landslide/sen12landslides/all_positives"
DEFAULT_GROUPS = ["sen12_landslides", "icimod", "glc"]
DEFAULT_OUT_DIR = str(
    Path(__file__).resolve().parent
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Landslide ERA5 window dedup.")
    parser.add_argument("--ds-root", type=str, default=DEFAULT_DS_ROOT)
    parser.add_argument("--groups", nargs="+", default=DEFAULT_GROUPS)
    parser.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR)
    parser.add_argument("--workers", type=int, default=1,
                        help="Workers for scanning. 1 (sequential) is best on weka.")
    parser.add_argument(
        "--rescan",
        action="store_true",
        help="Force re-scan even if all_windows.parquet exists.",
    )
    args = parser.parse_args()

    ds_root = Path(args.ds_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_pq = out_dir / "all_windows.parquet"
    deduped_pq = out_dir / "deduped_windows.parquet"
    summary_md = out_dir / "dedup_summary.md"

    # Step 1: Scan (or load cached)
    if all_pq.exists() and not args.rescan:
        logger.info("Loading cached scan from %s", all_pq)
        df_all = pd.read_parquet(str(all_pq))
    else:
        df_all = scan_windows(ds_root, args.groups, workers=args.workers)
        df_all.to_parquet(str(all_pq), index=False)
        logger.info("Cached scan to %s (%d rows)", all_pq, len(df_all))

    # Step 2: Dedup
    df_deduped = dedup_windows(df_all)
    df_deduped.to_parquet(str(deduped_pq), index=False)
    logger.info("Deduped windows saved to %s (%d rows)", deduped_pq, len(df_deduped))

    # Step 3: Summary
    generate_summary(df_all, df_deduped, summary_md)


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver")
    main()
