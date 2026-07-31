"""Report the structural facts the S1 + Tessera v2 export plan depends on.

Phase 0 recon, part A: pure metadata, no network. Run this before writing any
fetch config for the AEF supplemental datasets, because two design decisions
hang on what it prints:

1. **Are the windows calendar-year aligned?** Tessera v2 embeds one *calendar
   year* of observations per pixel. If every window's time range already is
   its label year, the ``*_all`` fetch layers can ride the eval windows
   directly (CASE A). If not, v2 needs mirrored fetch windows in a second
   group, the way ``pastis_tessera_v2.create_windows`` builds
   ``pastis_tessera_v2`` from ``pastis`` (CASE B) -- generalized to take the
   year per window instead of a module constant.

2. **Are the imagery layer's item groups chronological monthly mosaics?** The
   supplemental datasets store imagery as one layer with N item groups
   (``sentinel2``, ``sentinel2.1``, ...), not as named monthly layers. If the
   groups come back cloud-sorted rather than in calendar order, then a
   Sentinel-1 layer built from 12 monthly mosaics has no timestep-for-timestep
   correspondence with them -- the eval will still run (timestamps are
   synthesized in rslearn_dataset.get_timestamps), but S1 month *i* and S2
   group *i* would describe different parts of the year.

It also reports where a per-window label year could be read from, the window
group names needed for ``--group``, and whether the sampled windows share one
layer/group shape (a mixed shape means some windows are only partly
materialized, which the shard list has to tolerate).

Sampling matters: label years span 2018-2024 *within* a single dataset, so
inspecting one window says nothing about whether the set as a whole is
calendar-aligned. Defaults to 40 windows per dataset.

Example:
    python scripts/extract_tesserav2/dataset_shape.py --datasets ethiopia_crops
    python scripts/extract_tesserav2/dataset_shape.py --json_out shape.json
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import Counter
from datetime import datetime
from typing import Any

from rslearn.data_sources.data_source import Item
from rslearn.dataset import Dataset, Window
from upath import UPath

from olmoearth_pretrain.evals.studio_ingest.registry import get_dataset_entry
from olmoearth_pretrain.internal.all_evals import AEF_SUPPLEMENTAL_DATASETS

logger = logging.getLogger(__name__)

# A window time range counts as a calendar year if it starts on January 1 and
# spans most of a year. The lower bound is loose because rslearn windows are
# often built as [Jan 1, Dec 31] rather than a full 365-day span.
CALENDAR_MIN_DAYS = 330
CALENDAR_MAX_DAYS = 400

# Monthly mosaic spacing, used to judge whether consecutive item groups look
# like the 30-day mosaics the pretraining dataset uses.
MONTHLY_MIN_DAYS = 20
MONTHLY_MAX_DAYS = 45


def sample_windows(windows: list[Window], sample: int, seed: int) -> list[Window]:
    """Return a deterministic random subset of windows.

    Args:
        windows: all windows loaded from the dataset.
        sample: maximum number of windows to return.
        seed: RNG seed, so re-runs inspect the same windows.

    Returns:
        up to ``sample`` windows.
    """
    if len(windows) <= sample:
        return list(windows)
    return random.Random(seed).sample(windows, sample)


def classify_time_ranges(windows: list[Window]) -> tuple[Counter, str]:
    """Bucket window time ranges and decide CASE A vs CASE B.

    Args:
        windows: the sampled windows.

    Returns:
        (histogram, verdict) where the histogram is keyed by
        (start ISO date, span in days, "calendar"/"offset") and the verdict is
        one of "CASE A", "CASE B", "CASE B (mixed)", or "UNKNOWN".
    """
    histogram: Counter = Counter()
    calendar = 0
    dated = 0
    for window in windows:
        if window.time_range is None:
            histogram[("<none>", None, "none")] += 1
            continue
        start, end = window.time_range
        span_days = (end - start).days
        is_calendar = (start.month, start.day) == (
            1,
            1,
        ) and CALENDAR_MIN_DAYS <= span_days <= CALENDAR_MAX_DAYS
        dated += 1
        calendar += int(is_calendar)
        histogram[
            (f"{start:%m-%d}", span_days, "calendar" if is_calendar else "offset")
        ] += 1

    if dated == 0:
        return histogram, "UNKNOWN (no window carries a time range)"
    if calendar == dated:
        return histogram, "CASE A (v2 can ride the eval windows)"
    if calendar == 0:
        return histogram, "CASE B (v2 needs mirrored calendar-year windows)"
    return histogram, "CASE B (mixed -- some windows are calendar-aligned)"


def group_first_times(window: Window, layer_name: str) -> list[datetime | None]:
    """Return the earliest acquisition time of each item group in a layer.

    Args:
        window: the window to read prepared layer data from.
        layer_name: the imagery layer (e.g. "sentinel2").

    Returns:
        one datetime per item group, or None where the group carries no
        deserializable item time.
    """
    layer_datas = window.load_layer_datas()
    if layer_name not in layer_datas:
        return []

    # These datasets were built by a different pipeline, so an item may not
    # deserialize into rslearn's Item. That is worth reporting rather than
    # swallowing: undated groups make the ordering verdict UNKNOWN, and the
    # reader needs to know whether that means "no dates recorded" or "dates
    # recorded in a shape this script cannot read".
    failures: Counter = Counter()
    times: list[datetime | None] = []
    for group in layer_datas[layer_name].serialized_item_groups:
        stamp: datetime | None = None
        for serialized in group:
            try:
                item = Item.deserialize(serialized)
            except (AttributeError, KeyError, TypeError, ValueError) as e:
                failures[type(e).__name__] += 1
                continue
            if item.geometry.time_range is None:
                continue
            start = item.geometry.time_range[0]
            stamp = start if stamp is None else min(stamp, start)
        times.append(stamp)

    if failures:
        logger.warning(
            f"{window.group}/{window.name}: {sum(failures.values())} "
            f"'{layer_name}' item(s) did not deserialize ({dict(failures)}); "
            f"inspect the raw serialized_item_groups to read their dates"
        )
    return times


def classify_group_ordering(times: list[datetime | None]) -> str:
    """Decide whether item groups are chronological, and monthly-spaced.

    Args:
        times: per-group earliest acquisition times.

    Returns:
        a human-readable verdict.
    """
    known = [t for t in times if t is not None]
    if len(known) < 2:
        return "UNKNOWN (fewer than two dated groups)"
    ascending = all(a <= b for a, b in zip(known, known[1:]))
    if not ascending:
        return (
            "NOT CHRONOLOGICAL -- group index does not follow calendar order, so "
            "S1 monthly mosaics will not align with these groups"
        )
    gaps = [(b - a).days for a, b in zip(known, known[1:])]
    gaps.sort()
    median_gap = gaps[len(gaps) // 2]
    if MONTHLY_MIN_DAYS <= median_gap <= MONTHLY_MAX_DAYS:
        return f"CHRONOLOGICAL, ~monthly (median gap {median_gap}d)"
    return (
        f"CHRONOLOGICAL but not monthly (median gap {median_gap}d) -- check the "
        "S1 time_offset spacing against this"
    )


def layer_shapes(windows: list[Window]) -> Counter:
    """Count distinct (layer -> completed group count) shapes across windows.

    Args:
        windows: the sampled windows.

    Returns:
        a counter keyed by the sorted (layer, num_groups) tuple.
    """
    shapes: Counter = Counter()
    for window in windows:
        counts: Counter = Counter()
        for layer_name, _ in window.list_completed_layers():
            counts[layer_name] += 1
        shapes[tuple(sorted(counts.items()))] += 1
    return shapes


def inspect_dataset(
    name: str, imagery_layer: str, sample: int, seed: int
) -> dict[str, Any]:
    """Print the Phase 0 report for one dataset and return it as a dict.

    Args:
        name: registry dataset name.
        imagery_layer: layer whose item groups are inspected (e.g. "sentinel2").
        sample: number of windows to inspect.
        seed: RNG seed for sampling.

    Returns:
        a JSON-serializable summary of the findings.

    Raises:
        ValueError: if the registry entry has no weka_path.
    """
    entry = get_dataset_entry(name)
    if not entry.weka_path:
        raise ValueError(f"Registry entry '{name}' has no weka_path.")
    root = UPath(entry.weka_path)

    print(f"\n{'=' * 72}\n{name}  ({root})\n{'=' * 72}")

    config = json.loads((root / "config.json").read_text())
    declared = sorted(config.get("layers", {}))
    print(f"declared layers: {declared}")
    if imagery_layer in config.get("layers", {}):
        print(f"\n--- config.json layers[{imagery_layer!r}] ---")
        print(json.dumps(config["layers"][imagery_layer], indent=2))
    else:
        print(f"\n!! no '{imagery_layer}' layer in config.json -- pass --imagery_layer")

    windows = Dataset(root).load_windows()
    groups = Counter(w.group for w in windows)
    print(f"\n{len(windows)} windows; groups: {dict(groups)}")

    chosen = sample_windows(windows, sample, seed)
    histogram, case_verdict = classify_time_ranges(chosen)
    print(f"\n--- window time ranges (n={len(chosen)}) ---")
    for key, count in sorted(histogram.items(), key=lambda kv: -kv[1]):
        start, span, kind = key
        print(f"  {count:4d}x  start={start}  span={span}d  [{kind}]")
    print(f"  => {case_verdict}")

    probe = chosen[0]
    print(f"\n--- example window {probe.group}/{probe.name} ---")
    print(f"  time_range: {probe.time_range}")
    print(f"  options   : {getattr(probe, 'options', None)}")
    print(f"  bounds    : {probe.bounds}")
    print(f"  crs       : {probe.projection.crs}")

    times = group_first_times(probe, imagery_layer)
    order_verdict = classify_group_ordering(times)
    print(f"\n--- {imagery_layer} item groups on that window ---")
    for idx, stamp in enumerate(times):
        shown = f"{stamp:%Y-%m-%d}" if stamp else "<no dated item>"
        print(f"  group {idx:2d}  {shown}")
    print(f"  => {order_verdict}")

    shapes = layer_shapes(chosen)
    print(f"\n--- completed-layer shapes (n={len(chosen)}) ---")
    for shape, count in shapes.most_common():
        print(f"  {count:4d}x  {dict(shape)}")
    if len(shapes) > 1:
        print("  !! shapes differ -- some windows are only partly materialized")

    return {
        "dataset": name,
        "weka_path": str(root),
        "declared_layers": declared,
        "num_windows": len(windows),
        "window_groups": dict(groups),
        "sampled": len(chosen),
        "time_range_case": case_verdict,
        "time_range_histogram": {str(k): v for k, v in histogram.items()},
        "example_window": {
            "group": probe.group,
            "name": probe.name,
            "time_range": [t.isoformat() for t in probe.time_range]
            if probe.time_range
            else None,
            "options": getattr(probe, "options", None),
        },
        "item_group_times": [t.isoformat() if t else None for t in times],
        "item_group_ordering": order_verdict,
        "layer_shapes": {str(dict(k)): v for k, v in shapes.items()},
    }


def main() -> None:
    """Run the dataset-shape report over the requested datasets."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        type=str,
        default=",".join(AEF_SUPPLEMENTAL_DATASETS),
        help="Comma-separated registry dataset names to inspect.",
    )
    parser.add_argument(
        "--imagery_layer",
        type=str,
        default="sentinel2",
        help="Layer whose item groups are inspected for calendar ordering.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=40,
        help="Windows to sample per dataset (label years vary within a dataset).",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="RNG seed, so re-runs match."
    )
    parser.add_argument(
        "--json_out",
        type=str,
        default=None,
        help="Optional path to write the machine-readable summary.",
    )
    args = parser.parse_args()

    summaries = []
    for name in args.datasets.split(","):
        summaries.append(
            inspect_dataset(name.strip(), args.imagery_layer, args.sample, args.seed)
        )

    print(f"\n{'=' * 72}\nVERDICTS\n{'=' * 72}")
    for summary in summaries:
        print(f"{summary['dataset']:22s} {summary['time_range_case']}")
        print(f"{'':22s} {summary['item_group_ordering']}")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(summaries, f, indent=2)
        logger.info(f"Wrote summary to {args.json_out}")


if __name__ == "__main__":
    main()
