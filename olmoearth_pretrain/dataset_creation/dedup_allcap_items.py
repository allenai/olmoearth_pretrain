"""Deduplicate same-pass item groups in prepared allcap layers.

The same sensing pass can appear as multiple STAC items that all fully contain a
small window: Sentinel-2 MGRS tile overlaps yield same-instant items, and Landsat
adjacent-row scenes from one pass are ~24s apart. With space_mode=CONTAINS and a
large max_matches, prepare therefore records duplicate captures. This script
rewrites items.json after `rslearn dataset prepare` and before `rslearn dataset
materialize`, keeping only the first item group per pass (captures within
DEDUP_WINDOW_SECONDS of the previously kept capture are dropped), so duplicates
are never downloaded or materialized. No legitimate revisit is anywhere near that
close: S2 A/B are >=5 days apart and L8/L9 >=1 day.

Layers that are already materialized are skipped with a warning: dropping groups
after materialization would break the group_idx -> layer.N directory mapping.

Usage:
    python -m olmoearth_pretrain.dataset_creation.dedup_allcap_items \
        --ds_path /weka/.../dataset_creation/osm_allcaptures_pilot1k \
        --group res_10 --workers 16
"""

from __future__ import annotations

import argparse
import multiprocessing
from collections import Counter

import tqdm
from rslearn.data_sources import Item
from rslearn.dataset import Dataset, Window
from rslearn.dataset.window import WindowLayerData
from upath import UPath

from .create_windows.util import star_imap

DEFAULT_LAYERS = ["landsat", "sentinel1", "sentinel2_l2a"]

# Captures closer together than this are the same pass (S2 tile overlap: 0s apart;
# Landsat adjacent-row scenes: ~24s apart). The closest legitimate revisit is >=1 day.
DEDUP_WINDOW_SECONDS = 120


def dedup_layer_data(layer_data: WindowLayerData) -> tuple[WindowLayerData, int]:
    """Return layer data with same-pass duplicate groups dropped, and drop count."""
    times = []
    for group in layer_data.serialized_item_groups:
        if len(group) != 1:
            raise ValueError(
                f"expected allcap groups to have length 1 but got {len(group)}"
            )
        times.append(Item.deserialize(group[0]).geometry.time_range[0])

    # Decide keeps in chronological order (groups are usually already sorted, but we
    # do not rely on it); preserve the original group order for the kept groups.
    keep = [False] * len(times)
    last_kept = None
    for idx in sorted(range(len(times)), key=lambda i: times[i]):
        if (
            last_kept is None
            or (times[idx] - last_kept).total_seconds() >= DEDUP_WINDOW_SECONDS
        ):
            keep[idx] = True
            last_kept = times[idx]

    kept_groups = []
    kept_time_ranges = []
    for group_idx, group in enumerate(layer_data.serialized_item_groups):
        if not keep[group_idx]:
            continue
        kept_groups.append(group)
        if layer_data.group_time_ranges is not None:
            kept_time_ranges.append(layer_data.group_time_ranges[group_idx])

    num_dropped = len(layer_data.serialized_item_groups) - len(kept_groups)
    deduped = WindowLayerData(
        layer_name=layer_data.layer_name,
        serialized_item_groups=kept_groups,
        group_time_ranges=(
            kept_time_ranges if layer_data.group_time_ranges is not None else None
        ),
        materialized=layer_data.materialized,
    )
    return deduped, num_dropped


def dedup_window(window: Window, layers: list[str]) -> dict[str, tuple[int, int]]:
    """Dedup the given layers in one window; returns {layer: (before, after)}."""
    layer_datas = window.load_layer_datas()
    stats: dict[str, tuple[int, int]] = {}
    changed = False
    for layer_name in layers:
        if layer_name not in layer_datas:
            continue
        layer_data = layer_datas[layer_name]
        if layer_data.materialized:
            print(
                f"warning: skipping already-materialized layer {layer_name} "
                f"in window {window.name}"
            )
            continue
        before = len(layer_data.serialized_item_groups)
        deduped, num_dropped = dedup_layer_data(layer_data)
        stats[layer_name] = (before, before - num_dropped)
        if num_dropped > 0:
            layer_datas[layer_name] = deduped
            changed = True
    if changed:
        window.save_layer_datas(layer_datas)
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Dedup same-instant item groups in prepared allcap layers"
    )
    parser.add_argument("--ds_path", type=str, required=True)
    parser.add_argument("--group", type=str, default="res_10")
    parser.add_argument(
        "--layers",
        type=str,
        default=",".join(DEFAULT_LAYERS),
        help="Comma-separated allcap layer names to dedup",
    )
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    layers = [layer.strip() for layer in args.layers.split(",")]
    dataset = Dataset(UPath(args.ds_path))
    windows = dataset.load_windows(groups=[args.group], workers=args.workers)
    print(f"deduping layers {layers} in {len(windows)} windows")

    jobs = [dict(window=window, layers=layers) for window in windows]
    totals: Counter = Counter()
    if args.workers <= 1 or len(jobs) <= 1:
        all_stats = [dedup_window(**job) for job in tqdm.tqdm(jobs)]
    else:
        with multiprocessing.Pool(args.workers) as p:
            all_stats = list(
                tqdm.tqdm(star_imap(p, dedup_window, jobs), total=len(jobs))
            )
    for stats in all_stats:
        for layer_name, (before, after) in stats.items():
            totals[f"{layer_name}_before"] += before
            totals[f"{layer_name}_after"] += after
    for layer_name in layers:
        before = totals[f"{layer_name}_before"]
        after = totals[f"{layer_name}_after"]
        if before:
            print(
                f"{layer_name}: {before} -> {after} groups "
                f"({100 * (before - after) / before:.1f}% duplicates removed)"
            )
