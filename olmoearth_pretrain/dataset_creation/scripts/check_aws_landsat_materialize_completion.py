#!/usr/bin/env python3
"""Sample windows from an rslearn S3 dataset and report landsat materialization completion.

Usage:
    # Random sample of 10000 windows from a specific group:
    python scripts/check_materialize_completion.py \
        --root s3://rslearn-data-acquisition-368613568044-us-west-2-an/landsat_job/candidates/ \
        --k 1 --group res_10_s50ix24

    # Quick check: first 10000 windows found (skips full listing):
    python scripts/check_materialize_completion.py \
        --root s3://rslearn-data-acquisition-368613568044-us-west-2-an/landsat_job/candidates/ \
        --k 1 --group res_10_s50ix24 --first

Checks K*10000 windows for which landsat_moXX layers (01-12) have a `completed`
marker in S3. Reports per-layer, at-least-one, and all-layers completion rates.
"""

import argparse
import random
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
from urllib.parse import urlparse

import boto3
from tqdm import tqdm

_thread_local = threading.local()


def _get_s3_client() -> Any:
    """Return a per-thread boto3 S3 client (reuses TCP connections)."""
    if not hasattr(_thread_local, "s3"):
        _thread_local.s3 = boto3.client("s3")
    return _thread_local.s3


LANDSAT_LAYERS = [f"landsat_mo{i:02d}" for i in range(1, 13)]
_LANDSAT_LAYER_SET = set(LANDSAT_LAYERS)


def parse_s3_uri(uri: str) -> tuple[str, str]:
    """Return (bucket, prefix) from an s3:// URI. Prefix has no leading /."""
    parsed = urlparse(uri)
    bucket = parsed.netloc
    prefix = parsed.path.lstrip("/")
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    return bucket, prefix


def list_groups(s3: Any, bucket: str, windows_prefix: str) -> list[str]:
    """List group-level prefixes under windows/."""
    paginator = s3.get_paginator("list_objects_v2")
    groups: list[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=windows_prefix, Delimiter="/"):
        for cp in page.get("CommonPrefixes", []):
            groups.append(cp["Prefix"])
    return groups


def list_windows_in_group(
    s3: Any,
    bucket: str,
    group_prefix: str,
    pbar: tqdm | None = None,
    limit: int | None = None,
) -> list[str]:
    """List window-level prefixes within a group. Returns full prefixes.

    If limit is set, stops listing once that many windows have been collected.
    """
    paginator = s3.get_paginator("list_objects_v2")
    windows: list[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=group_prefix, Delimiter="/"):
        prefixes = page.get("CommonPrefixes", [])
        for cp in prefixes:
            windows.append(cp["Prefix"])
        if pbar is not None:
            pbar.update(len(prefixes))
        if limit is not None and len(windows) >= limit:
            break
    return windows


def check_window_completion(
    s3: Any, bucket: str, window_prefix: str
) -> dict[str, bool]:
    """Check which landsat layers have a completed marker for a window.

    Uses a single list-objects-v2 call scoped to the layers/ prefix and filters
    for completed markers. One API call beats 12 sequential head_object calls.
    """
    layers_prefix = f"{window_prefix}layers/"
    paginator = s3.get_paginator("list_objects_v2")
    completed_layers: set[str] = set()

    for page in paginator.paginate(Bucket=bucket, Prefix=layers_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith("/completed"):
                continue
            parts = key[len(layers_prefix) :].split("/")
            if len(parts) == 2 and parts[0] in _LANDSAT_LAYER_SET:
                completed_layers.add(parts[0])

    return {layer: (layer in completed_layers) for layer in LANDSAT_LAYERS}


def main() -> None:
    """Sample windows from S3 and report landsat materialization completion rates."""
    parser = argparse.ArgumentParser(
        description="Sample windows and report landsat materialization completion."
    )
    parser.add_argument(
        "--root",
        required=True,
        help="S3 dataset root URI (e.g. s3://bucket/landsat_job/candidates/)",
    )
    parser.add_argument(
        "--group",
        default=None,
        help="Restrict to a single group prefix (e.g. res_10_s50ix24). "
        "If omitted, samples across all groups.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=1,
        help="Sample size = K * 10000 windows (default: 1)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=64,
        help="Number of concurrent S3 threads (default: 64)",
    )
    parser.add_argument(
        "--first",
        action="store_true",
        help="Take the first K*10000 windows found instead of random sampling. "
        "Much faster since it stops listing early.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    sample_size = args.k * 10_000
    rng = random.Random(args.seed)

    bucket, prefix = parse_s3_uri(args.root)
    windows_prefix = f"{prefix}windows/"

    s3 = boto3.client("s3")

    # --- Step 1: Discover windows ---
    print(f"Listing windows under s3://{bucket}/{windows_prefix} ...")

    if args.group:
        group_prefixes = [f"{windows_prefix}{args.group}/"]
    else:
        group_prefixes = list_groups(s3, bucket, windows_prefix)
        print(f"  Found {len(group_prefixes)} groups")

    all_window_prefixes: list[str] = []
    limit = sample_size if args.first else None

    if len(group_prefixes) == 1:
        with tqdm(desc="Listing windows", unit=" windows") as pbar:
            wins = list_windows_in_group(
                s3,
                bucket,
                group_prefixes[0],
                pbar=pbar,
                limit=limit,
            )
            all_window_prefixes.extend(wins)
    else:
        lock = threading.Lock()
        with tqdm(desc="Listing windows", unit=" windows") as pbar:

            def _list_group(gp: str) -> list[str]:
                if limit is not None:
                    remaining = limit - len(all_window_prefixes)
                    if remaining <= 0:
                        return []
                else:
                    remaining = None
                client = _get_s3_client()
                wins = list_windows_in_group(
                    client,
                    bucket,
                    gp,
                    pbar=pbar,
                    limit=remaining,
                )
                with lock:
                    group_name = gp.rstrip("/").split("/")[-1]
                    pbar.set_postfix_str(f"{group_name} ({len(wins):,})")
                return wins

            with ThreadPoolExecutor(max_workers=min(16, len(group_prefixes))) as pool:
                for wins in pool.map(_list_group, group_prefixes):
                    all_window_prefixes.extend(wins)
                    if limit is not None and len(all_window_prefixes) >= limit:
                        break

    total_windows = len(all_window_prefixes)
    if args.first:
        print(f"  Listed {total_windows:,} windows (stopped early with --first)")
    else:
        print(f"  Found {total_windows:,} windows total")

    if total_windows == 0:
        print("No windows found. Check --root and --group.")
        sys.exit(1)

    if args.first:
        sampled = all_window_prefixes[:sample_size]
    else:
        sampled = rng.sample(all_window_prefixes, min(sample_size, total_windows))
    actual_sample_size = len(sampled)
    mode = "first" if args.first else "random"
    print(f"  Selected {actual_sample_size:,} windows ({mode}, K={args.k})\n")

    # --- Step 2: Check completion in parallel ---
    per_layer_completed: dict[str, int] = defaultdict(int)
    at_least_one = 0
    all_completed = 0
    errors = 0

    def _check(wp: str) -> dict[str, bool] | None:
        try:
            return check_window_completion(_get_s3_client(), bucket, wp)
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_check, wp): wp for wp in sampled}
        for fut in tqdm(
            as_completed(futures), total=len(futures), desc="Checking completion"
        ):
            result = fut.result()
            if result is None:
                errors += 1
                continue

            completed_count = sum(1 for v in result.values() if v)
            if completed_count > 0:
                at_least_one += 1
            if completed_count == len(LANDSAT_LAYERS):
                all_completed += 1
            for layer, done in result.items():
                if done:
                    per_layer_completed[layer] += 1

    # --- Step 3: Report ---
    checked = actual_sample_size - errors
    print(f"\n{'=' * 60}")
    print("Materialization Completion Report")
    print(f"{'=' * 60}")
    print(f"  Dataset root : s3://{bucket}/{prefix}")
    if args.group:
        print(f"  Group filter : {args.group}")
    print(f"  Windows listed: {total_windows:,}")
    print(
        f"  Sampled      : {actual_sample_size:,} ({'first' if args.first else 'random'})"
    )
    if errors:
        print(f"  Errors       : {errors}")
    print(f"  Checked      : {checked:,}")
    print()

    print("Per-layer completion:")
    print(f"  {'Layer':<16} {'Completed':>10} {'Rate':>8}")
    print(f"  {'-' * 16} {'-' * 10} {'-' * 8}")
    for layer in LANDSAT_LAYERS:
        n = per_layer_completed.get(layer, 0)
        rate = n / checked * 100 if checked > 0 else 0
        print(f"  {layer:<16} {n:>10,} {rate:>7.2f}%")

    print()
    rate_any = at_least_one / checked * 100 if checked > 0 else 0
    rate_all = all_completed / checked * 100 if checked > 0 else 0
    print(
        f"  At least 1 layer completed : {at_least_one:>10,} / {checked:,}  ({rate_any:.2f}%)"
    )
    print(
        f"  All 12 layers completed    : {all_completed:>10,} / {checked:,}  ({rate_all:.2f}%)"
    )
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
