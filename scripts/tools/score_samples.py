"""Score all samples in an H5 dataset directory.

Writes numbered shard files during scoring, then merges at the end.
Supports resume by skipping existing shards.

Usage:
    # Full scoring run
    python scripts/tools/score_samples.py score \
        --h5-dir /path/to/h5py_data/.../1138828 \
        --output scores.parquet \
        --workers 32

    # Benchmark: time each scorer on N samples (single-process, detailed stats)
    python scripts/tools/score_samples.py benchmark \
        --h5-dir /path/to/h5py_data/.../1138828 \
        --max-samples 1000
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

H5_DIR_DEFAULT = (
    "/weka/dfive-default/helios/dataset/osm_sampling/"
    "h5py_data_w_missing_timesteps_zstd_3_128_x_4/"
    "cdl_gse_landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_"
    "worldcereal_worldcover_worldpop_wri_canopy_height_map/1138828"
)

# Globals set once per worker via initializer
_latlon_map: dict[int, tuple[float, float]] = {}
_scorer_names: list[str] | None = None


def _init_worker(
    latlon_map: dict[int, tuple[float, float]], scorer_names: list[str] | None
) -> None:
    """Initialize per-worker globals."""
    global _latlon_map, _scorer_names
    _latlon_map = latlon_map
    _scorer_names = scorer_names


def _score_one(h5_path: str) -> dict[str, Any] | None:
    """Score a single H5 file. Called in worker processes."""
    from olmoearth_pretrain.data.sample_scorer import load_h5_sample, score_sample

    try:
        sample, meta = load_h5_sample(h5_path)
    except Exception as e:
        logger.warning(f"Failed to load {h5_path}: {e}")
        return None

    sample_idx = meta.get("sample_index")
    if sample_idx is not None and sample_idx in _latlon_map:
        meta["lat"], meta["lon"] = _latlon_map[sample_idx]

    features = score_sample(sample, meta, scorers=_scorer_names)
    features["sample_index"] = float(sample_idx) if sample_idx is not None else -1.0
    features["filename"] = os.path.basename(h5_path)
    return features


def load_latlon_map(h5_dir: str) -> dict[int, tuple[float, float]]:
    """Load latlon_distribution.npy and build sample_index -> (lat, lon) map."""
    latlon_path = os.path.join(h5_dir, "latlon_distribution.npy")
    if not os.path.exists(latlon_path):
        logger.warning(f"No latlon_distribution.npy at {latlon_path}")
        return {}

    latlons = np.load(latlon_path)
    return {
        i: (float(latlons[i, 0]), float(latlons[i, 1])) for i in range(latlons.shape[0])
    }


def _shard_dir(output_path: str) -> str:
    """Return the directory for intermediate shard files."""
    base = output_path.rsplit(".", 1)[0]
    return f"{base}_shards"


def _write_shard(results: list[dict[str, Any]], shard_dir: str, shard_idx: int) -> str:
    """Write a batch of results as a numbered parquet shard."""
    os.makedirs(shard_dir, exist_ok=True)
    path = os.path.join(shard_dir, f"shard_{shard_idx:06d}.parquet")
    pd.DataFrame(results).to_parquet(path, index=False)
    return path


def _merge_shards(shard_dir: str, output_path: str) -> None:
    """Merge all shard files into a single output file."""
    shard_files = sorted(Path(shard_dir).glob("shard_*.parquet"))
    if not shard_files:
        logger.warning("No shards to merge")
        return

    logger.info(f"Merging {len(shard_files)} shards into {output_path}")
    dfs = [pd.read_parquet(f) for f in shard_files]
    merged = pd.concat(dfs, ignore_index=True)

    if output_path.endswith(".parquet"):
        merged.to_parquet(output_path, index=False)
    else:
        merged.to_csv(output_path, index=False)

    logger.info(f"Merged output shape: {merged.shape}")
    logger.info(f"Columns ({len(merged.columns)}): {sorted(merged.columns.tolist())}")


def _get_done_filenames(shard_dir: str) -> set[str]:
    """Read all existing shards to find already-scored filenames."""
    done: set[str] = set()
    shard_files = sorted(Path(shard_dir).glob("shard_*.parquet"))
    for f in shard_files:
        try:
            df = pd.read_parquet(f, columns=["filename"])
            done.update(df["filename"].values)
        except Exception:
            pass
    return done


# ============================================================================
# Subcommands
# ============================================================================


def cmd_score(args: argparse.Namespace) -> None:
    """Run full parallel scoring with shard-based checkpointing."""
    h5_dir = args.h5_dir
    output_path = args.output
    shard_dir = _shard_dir(output_path)
    shard_size = args.shard_size

    logger.info(f"Scanning {h5_dir} for H5 files...")
    h5_files = sorted(str(p) for p in Path(h5_dir).glob("sample_*.h5"))
    logger.info(f"Found {len(h5_files)} H5 files")

    if args.max_samples:
        h5_files = h5_files[: args.max_samples]
        logger.info(f"Limited to {len(h5_files)} files")

    # Resume: find already-scored files from existing shards
    existing_shards = (
        sorted(Path(shard_dir).glob("shard_*.parquet"))
        if os.path.isdir(shard_dir)
        else []
    )
    next_shard_idx = len(existing_shards)

    if args.resume and existing_shards:
        done_filenames = _get_done_filenames(shard_dir)
        logger.info(
            f"Resuming: {len(done_filenames)} already scored in {len(existing_shards)} shards"
        )
        h5_files = [f for f in h5_files if os.path.basename(f) not in done_filenames]
        logger.info(f"{len(h5_files)} remaining to score")

    if not h5_files:
        logger.info("Nothing to score. Merging existing shards.")
        _merge_shards(shard_dir, output_path)
        return

    latlon_map = load_latlon_map(h5_dir)
    logger.info(f"Loaded latlon for {len(latlon_map)} samples")

    from olmoearth_pretrain.data.sample_scorer import SCORER_REGISTRY

    scorer_names = args.scorers
    if scorer_names:
        unknown = set(scorer_names) - set(SCORER_REGISTRY.keys())
        if unknown:
            logger.error(
                f"Unknown scorers: {unknown}. Available: {sorted(SCORER_REGISTRY.keys())}"
            )
            sys.exit(1)
    logger.info(f"Running scorers: {scorer_names or sorted(SCORER_REGISTRY.keys())}")

    batch: list[dict[str, Any]] = []
    shard_idx = next_shard_idx
    scored = 0
    t0 = time.time()

    with Pool(
        processes=args.workers,
        initializer=_init_worker,
        initargs=(latlon_map, scorer_names),
    ) as pool:
        for result in tqdm(
            pool.imap_unordered(_score_one, h5_files, chunksize=64),
            total=len(h5_files),
            desc="Scoring",
        ):
            if result is not None:
                batch.append(result)
                scored += 1

            if len(batch) >= shard_size:
                path = _write_shard(batch, shard_dir, shard_idx)
                logger.info(f"Wrote shard {shard_idx} ({len(batch)} samples) -> {path}")
                shard_idx += 1
                batch = []

    # Write remaining
    if batch:
        path = _write_shard(batch, shard_dir, shard_idx)
        logger.info(f"Wrote shard {shard_idx} ({len(batch)} samples) -> {path}")

    elapsed = time.time() - t0
    logger.info(
        f"Scoring done. {scored} samples in {elapsed:.1f}s "
        f"({scored / max(elapsed, 1):.0f} samples/s)"
    )

    # Merge all shards
    _merge_shards(shard_dir, output_path)


def cmd_benchmark(args: argparse.Namespace) -> None:
    """Benchmark per-scorer timing on N samples (single-process for accurate timing)."""
    from olmoearth_pretrain.data.sample_scorer import (
        SCORER_REGISTRY,
        load_h5_sample,
        score_sample_timed,
    )

    h5_dir = args.h5_dir
    max_samples = args.max_samples or 1000

    logger.info(f"Scanning {h5_dir} for H5 files...")
    h5_files = sorted(str(p) for p in Path(h5_dir).glob("sample_*.h5"))
    logger.info(f"Found {len(h5_files)} H5 files, benchmarking on {max_samples}")
    h5_files = h5_files[:max_samples]

    latlon_map = load_latlon_map(h5_dir)

    scorer_names = args.scorers
    if scorer_names:
        unknown = set(scorer_names) - set(SCORER_REGISTRY.keys())
        if unknown:
            logger.error(
                f"Unknown scorers: {unknown}. Available: {sorted(SCORER_REGISTRY.keys())}"
            )
            sys.exit(1)

    all_timings: list[dict[str, float]] = []
    load_times: list[float] = []
    total_times: list[float] = []
    errors = 0

    for h5_path in tqdm(h5_files, desc="Benchmarking"):
        t_load_start = time.perf_counter()
        try:
            sample, meta = load_h5_sample(h5_path)
        except Exception:
            errors += 1
            continue
        t_load = time.perf_counter() - t_load_start
        load_times.append(t_load)

        sample_idx = meta.get("sample_index")
        if sample_idx is not None and sample_idx in latlon_map:
            meta["lat"], meta["lon"] = latlon_map[sample_idx]

        t_total_start = time.perf_counter()
        _features, timings = score_sample_timed(sample, meta, scorers=scorer_names)
        total_times.append(time.perf_counter() - t_total_start)
        all_timings.append(timings)

    if not all_timings:
        logger.error("No samples processed successfully")
        return

    timing_df = pd.DataFrame(all_timings)
    n = len(timing_df)

    load_arr = np.array(load_times)
    total_arr = np.array(total_times)

    print("\n" + "=" * 72)
    print(f"  BENCHMARK RESULTS  ({n} samples, {errors} errors)")
    print("=" * 72)

    print(f"\n{'Phase':<30s} {'Mean':>10s} {'Median':>10s} {'P95':>10s} {'Total':>10s}")
    print("-" * 72)
    print(
        f"{'h5_load':<30s} "
        f"{load_arr.mean() * 1000:>9.2f}ms "
        f"{np.median(load_arr) * 1000:>9.2f}ms "
        f"{np.percentile(load_arr, 95) * 1000:>9.2f}ms "
        f"{load_arr.sum():>9.2f}s"
    )
    print(
        f"{'all_scorers':<30s} "
        f"{total_arr.mean() * 1000:>9.2f}ms "
        f"{np.median(total_arr) * 1000:>9.2f}ms "
        f"{np.percentile(total_arr, 95) * 1000:>9.2f}ms "
        f"{total_arr.sum():>9.2f}s"
    )

    print(
        f"\n{'Scorer':<30s} {'Mean':>10s} {'Median':>10s} {'P95':>10s} {'Total':>10s} {'%':>6s}"
    )
    print("-" * 72)

    scorer_stats = []
    for col in sorted(timing_df.columns):
        vals = timing_df[col].values
        scorer_stats.append(
            (col, vals.mean(), np.median(vals), np.percentile(vals, 95), vals.sum())
        )

    scorer_stats.sort(key=lambda x: -x[4])
    grand_total = total_arr.sum()

    for name, mean, median, p95, total in scorer_stats:
        pct = (total / grand_total * 100) if grand_total > 0 else 0
        print(
            f"{name:<30s} "
            f"{mean * 1000:>9.2f}ms "
            f"{median * 1000:>9.2f}ms "
            f"{p95 * 1000:>9.2f}ms "
            f"{total:>9.2f}s "
            f"{pct:>5.1f}%"
        )

    print("-" * 72)
    total_wall = load_arr.sum() + total_arr.sum()
    throughput = n / total_wall if total_wall > 0 else 0
    print(f"\nTotal wall time: {total_wall:.1f}s")
    print(f"Throughput (single-process): {throughput:.0f} samples/s")
    print(f"Estimated throughput (16 workers): ~{throughput * 14:.0f} samples/s")
    print(
        f"Estimated time for 1M samples (16 workers): "
        f"~{1_000_000 / max(throughput * 14, 1) / 60:.0f} min"
    )


def main() -> None:
    """Entry point with subcommands: score, benchmark."""
    parser = argparse.ArgumentParser(
        description="Sample scoring engine for dataset characterization"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # -- score subcommand --
    p_score = subparsers.add_parser("score", help="Score samples in parallel")
    p_score.add_argument(
        "--h5-dir", default=H5_DIR_DEFAULT, help="Path to H5 dataset directory"
    )
    p_score.add_argument(
        "--output", required=True, help="Output path (.parquet or .csv)"
    )
    p_score.add_argument(
        "--workers", type=int, default=16, help="Number of parallel workers"
    )
    p_score.add_argument(
        "--max-samples", type=int, default=None, help="Max samples to process"
    )
    p_score.add_argument(
        "--scorers", nargs="*", default=None, help="Scorer names to run"
    )
    p_score.add_argument(
        "--resume", action="store_true", help="Skip already-scored samples"
    )
    p_score.add_argument(
        "--shard-size",
        type=int,
        default=100_000,
        help="Write a checkpoint shard every N samples (default: 100k)",
    )

    # -- benchmark subcommand --
    p_bench = subparsers.add_parser("benchmark", help="Benchmark per-scorer timing")
    p_bench.add_argument(
        "--h5-dir", default=H5_DIR_DEFAULT, help="Path to H5 dataset directory"
    )
    p_bench.add_argument(
        "--max-samples", type=int, default=1000, help="Number of samples to benchmark"
    )
    p_bench.add_argument(
        "--scorers", nargs="*", default=None, help="Scorer names to benchmark"
    )

    args = parser.parse_args()
    if args.command == "score":
        cmd_score(args)
    elif args.command == "benchmark":
        cmd_benchmark(args)


if __name__ == "__main__":
    main()
