"""Check satellite imagery availability for studio corpus locations via STAC.

Queries the Planetary Computer STAC API to estimate how many monthly
observations exist at each corpus location, WITHOUT downloading anything.
Reports per-theme survival rates against the ≥12 timestep filter.

Usage:
    # Quick check on 100 samples per theme
    python scripts/tools/check_corpus_coverage.py \
        --corpus /path/to/pretraining_corpus.csv \
        --samples-per-theme 100

    # Full check (slow but thorough)
    python scripts/tools/check_corpus_coverage.py \
        --corpus /path/to/pretraining_corpus.csv \
        --samples-per-theme 500 \
        --output coverage_report.parquet
"""

from __future__ import annotations

import argparse
import logging
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import planetary_computer
import pystac_client

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PC_ENDPOINT = "https://planetarycomputer.microsoft.com/api/stac/v1"

COLLECTIONS = {
    "sentinel-2-l2a": {"min_date": "2015-06-01", "alias": "s2"},
    "sentinel-1-grd": {"min_date": "2014-10-01", "alias": "s1"},
    "landsat-c2-l2": {"min_date": "2013-01-01", "alias": "landsat"},
}


def get_stac_catalog() -> pystac_client.Client:
    return pystac_client.Client.open(
        PC_ENDPOINT, modifier=planetary_computer.sign_inplace
    )


def count_monthly_observations(
    catalog: pystac_client.Client,
    lon: float,
    lat: float,
    start_time: datetime,
    end_time: datetime,
    collection: str,
    buffer_deg: float = 0.01,
) -> dict:
    """Count how many months have at least one observation in the time range.

    Returns dict with monthly counts and summary stats.
    """
    bbox = [lon - buffer_deg, lat - buffer_deg, lon + buffer_deg, lat + buffer_deg]

    # Extend to a full year if the range is shorter
    duration = (end_time - start_time).days
    if duration < 330:
        center = start_time + (end_time - start_time) / 2
        start_time = center - timedelta(days=183)
        end_time = center + timedelta(days=183)

    time_str = f"{start_time.strftime('%Y-%m-%d')}/{end_time.strftime('%Y-%m-%d')}"

    try:
        search = catalog.search(
            collections=[collection],
            bbox=bbox,
            datetime=time_str,
            max_items=500,
        )
        items = list(search.items())
    except Exception as e:
        logger.debug(f"STAC query failed for ({lat:.3f}, {lon:.3f}): {e}")
        return {"n_items": 0, "n_months": 0, "months_with_data": []}

    # Count unique year-months
    months_with_data: set[tuple[int, int]] = set()
    for item in items:
        dt = item.datetime or item.properties.get("datetime")
        if dt is None:
            continue
        if isinstance(dt, str):
            dt = datetime.fromisoformat(dt.replace("Z", "+00:00"))
        months_with_data.add((dt.year, dt.month))

    return {
        "n_items": len(items),
        "n_months": len(months_with_data),
        "months_with_data": sorted(months_with_data),
    }


def check_one_location(
    catalog: pystac_client.Client,
    row: dict,
    collections: list[str],
) -> dict:
    """Check all collections for one corpus location."""
    lat, lon = row["lat"], row["lon"]
    start_str = row.get("start_time")
    end_str = row.get("end_time")

    if pd.isna(start_str) or pd.isna(end_str):
        start_time = datetime(2020, 1, 1, tzinfo=timezone.utc)
        end_time = datetime(2021, 1, 1, tzinfo=timezone.utc)
    else:
        start_time = pd.to_datetime(start_str, format="ISO8601").to_pydatetime()
        end_time = pd.to_datetime(end_str, format="ISO8601").to_pydatetime()
        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=timezone.utc)
        if end_time.tzinfo is None:
            end_time = end_time.replace(tzinfo=timezone.utc)

    result = {
        "lat": lat,
        "lon": lon,
        "theme": row["theme"],
        "start_time": start_str,
        "end_time": end_str,
        "record_id": row.get("record_id", ""),
    }

    max_months = 0
    for coll in collections:
        alias = COLLECTIONS[coll]["alias"]
        stats = count_monthly_observations(catalog, lon, lat, start_time, end_time, coll)
        result[f"{alias}_items"] = stats["n_items"]
        result[f"{alias}_months"] = stats["n_months"]
        max_months = max(max_months, stats["n_months"])

    result["max_months"] = max_months
    result["would_survive_filter"] = max_months >= 12
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Check satellite coverage for corpus")
    parser.add_argument(
        "--corpus",
        required=True,
        help="Path to pretraining_corpus.csv",
    )
    parser.add_argument(
        "--samples-per-theme",
        type=int,
        default=100,
        help="How many locations to sample per theme (default: 100)",
    )
    parser.add_argument(
        "--collections",
        nargs="*",
        default=["sentinel-2-l2a"],
        help="STAC collections to query",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output parquet path for detailed results",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel threads for STAC queries",
    )
    parser.add_argument(
        "--themes",
        nargs="*",
        default=None,
        help="Only check these themes (default: all)",
    )
    args = parser.parse_args()

    logger.info(f"Loading corpus from {args.corpus}")
    df = pd.read_csv(args.corpus, low_memory=False)
    logger.info(f"Corpus shape: {df.shape}")

    if args.themes:
        df = df[df["theme"].isin(args.themes)]
        logger.info(f"Filtered to themes {args.themes}: {len(df)} rows")

    # Stratified sample
    sampled = df.groupby("theme").apply(
        lambda g: g.sample(n=min(len(g), args.samples_per_theme), random_state=42),
        include_groups=False,
    ).reset_index(level=0)
    logger.info(f"Sampled {len(sampled)} locations across {sampled['theme'].nunique()} themes")

    catalog = get_stac_catalog()
    logger.info(f"Connected to Planetary Computer STAC")
    logger.info(f"Checking collections: {args.collections}")

    rows_to_check = sampled.to_dict("records")
    results = []
    t0 = time.time()
    errors = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(check_one_location, catalog, row, args.collections): i
            for i, row in enumerate(rows_to_check)
        }
        for i, future in enumerate(as_completed(futures)):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                errors += 1
                logger.warning(f"Error: {e}")

            if (i + 1) % 50 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                remaining = (len(futures) - i - 1) / rate
                logger.info(
                    f"Progress: {i+1}/{len(futures)} "
                    f"({rate:.1f}/s, ~{remaining/60:.1f} min remaining)"
                )

    elapsed = time.time() - t0
    logger.info(f"Done: {len(results)} checked in {elapsed:.0f}s, {errors} errors")

    results_df = pd.DataFrame(results)

    # Print summary
    print("\n" + "=" * 90)
    print("  SATELLITE COVERAGE REPORT")
    print("=" * 90)

    for coll in args.collections:
        alias = COLLECTIONS[coll]["alias"]
        items_col = f"{alias}_items"
        months_col = f"{alias}_months"

        if items_col not in results_df.columns:
            continue

        print(f"\n--- {coll} ---")
        print(f"{'Theme':<30s} {'N':>5s} {'Items':>8s} {'Months':>8s} "
              f"{'≥12mo':>7s} {'Survive%':>9s}")
        print("-" * 75)

        for theme in sorted(results_df["theme"].unique()):
            sub = results_df[results_df["theme"] == theme]
            n = len(sub)
            mean_items = sub[items_col].mean()
            mean_months = sub[months_col].mean()
            survive = (sub[months_col] >= 12).sum()
            pct = survive / n * 100 if n > 0 else 0
            print(f"{theme:<30s} {n:>5d} {mean_items:>8.1f} {mean_months:>8.1f} "
                  f"{survive:>5d}/{n:<3d} {pct:>8.1f}%")

        total_n = len(results_df)
        total_survive = (results_df[months_col] >= 12).sum()
        print("-" * 75)
        print(f"{'TOTAL':<30s} {total_n:>5d} {results_df[items_col].mean():>8.1f} "
              f"{results_df[months_col].mean():>8.1f} "
              f"{total_survive:>5d}/{total_n:<3d} "
              f"{total_survive/total_n*100:>8.1f}%")

    # Overall survival with any collection
    print(f"\n--- Combined (any collection with ≥12 months) ---")
    print(f"{'Theme':<30s} {'N':>5s} {'MaxMo':>8s} {'Survive':>9s} {'%':>8s}")
    print("-" * 65)
    for theme in sorted(results_df["theme"].unique()):
        sub = results_df[results_df["theme"] == theme]
        n = len(sub)
        mean_max = sub["max_months"].mean()
        survive = sub["would_survive_filter"].sum()
        pct = survive / n * 100 if n > 0 else 0
        print(f"{theme:<30s} {n:>5d} {mean_max:>8.1f} {survive:>5d}/{n:<3d} {pct:>8.1f}%")

    total_survive = results_df["would_survive_filter"].sum()
    print("-" * 65)
    print(f"{'TOTAL':<30s} {total_n:>5d} {results_df['max_months'].mean():>8.1f} "
          f"{total_survive:>5d}/{total_n:<3d} "
          f"{total_survive/total_n*100:>8.1f}%")

    # Vessel detection deep dive
    vd = results_df[results_df["theme"] == "vessel_detection"]
    if len(vd) > 0:
        print(f"\n--- Vessel Detection Deep Dive ---")
        print(f"N sampled: {len(vd)}")
        for coll in args.collections:
            alias = COLLECTIONS[coll]["alias"]
            months_col = f"{alias}_months"
            if months_col in vd.columns:
                print(f"  {coll}:")
                print(f"    months: mean={vd[months_col].mean():.1f}, "
                      f"median={vd[months_col].median():.0f}, "
                      f"min={vd[months_col].min()}, max={vd[months_col].max()}")
                print(f"    ≥12 months: {(vd[months_col] >= 12).sum()}/{len(vd)} "
                      f"({(vd[months_col] >= 12).mean()*100:.1f}%)")
                print(f"    ≥6 months: {(vd[months_col] >= 6).sum()}/{len(vd)} "
                      f"({(vd[months_col] >= 6).mean()*100:.1f}%)")

    if args.output:
        results_df.to_parquet(args.output, index=False)
        logger.info(f"Saved detailed results to {args.output}")


if __name__ == "__main__":
    main()
