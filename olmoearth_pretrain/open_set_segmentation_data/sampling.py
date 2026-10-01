"""Sampling helpers: class balancing, regression bucketing, detection encoding.

Note on performance: weka small-file I/O is slow (~70 ms/file cold). Always read and
write label patches with a multiprocessing Pool (e.g. 64 workers), never a serial loop.
"""

import random
from collections import Counter, defaultdict
from collections.abc import Callable
from typing import Any

import numpy as np

# Hard cap on total label samples per dataset (see spec 5). Keeps any single dataset from
# dominating the corpus. geolifeclef_geoplant (50,800) predates this cap and is grandfathered.
MAX_SAMPLES_PER_DATASET = 25000


def _as_keyfn(
    key: str | Callable[[dict[str, Any]], Any],
) -> Callable[[dict[str, Any]], Any]:
    """Normalize a key spec (property name or function) to a record lookup function."""
    if callable(key):
        return key
    key_name = key
    return lambda r: r[key_name]


def _stable_order(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deterministically order records by their scalar-field content.

    Selection helpers below build their buckets by iterating ``records`` and then apply a
    seeded shuffle. That is only reproducible if ``records`` itself arrives in a stable
    order — but callers commonly build ``records`` from
    ``rslearn.utils.mp.star_imap_unordered``, whose completion order is nondeterministic, so
    a fixed seed still yielded a different selected subset on each fresh run. Sorting here by
    each record's scalar fields (lon/lat, name, tile indices, ids, …; arrays and class-lists
    are ignored for ordering) makes the downstream shuffle reproducible regardless of input
    order. Values are compared via ``str(...)`` so mixed/None/NumPy-scalar fields never raise.
    """

    def keyfn(r: dict[str, Any]) -> tuple[tuple[str, str], ...]:
        return tuple(
            (k, str(r[k]))
            for k in sorted(r)
            if not isinstance(r[k], list | dict | set | tuple | np.ndarray)
        )

    return sorted(records, key=keyfn)


def balance_by_class(
    records: list[dict[str, Any]],
    key: str | Callable[[dict[str, Any]], Any],
    per_class: int = 1000,
    seed: int = 42,
    total_cap: int | None = MAX_SAMPLES_PER_DATASET,
) -> list[dict[str, Any]]:
    """Return up to ``per_class`` records per class value (shuffled, seeded).

    ``key`` is a property name or a function mapping a record to its class value.
    Records whose class value is None are dropped. If ``total_cap`` is set (default
    25,000), the effective per-class limit is lowered to ``total_cap // n_classes`` so the
    dataset total stays under the cap while remaining class-balanced. Pass
    ``total_cap=None`` to disable.
    """
    keyfn = _as_keyfn(key)
    records = _stable_order(records)  # deterministic regardless of input order
    buckets: dict[Any, list] = defaultdict(list)
    for r in records:
        v = keyfn(r)
        if v is not None:
            buckets[v].append(r)
    if total_cap is not None and buckets:
        per_class = min(per_class, max(1, total_cap // len(buckets)))
    rng = random.Random(seed)
    out: list[dict[str, Any]] = []
    for v, items in buckets.items():
        rng.shuffle(items)
        out.extend(items[:per_class])
    return out


def balance_tiles_by_class(
    records: list[dict[str, Any]],
    classes_key: str | Callable[[dict[str, Any]], list[int]] = "classes_present",
    per_class: int = 1000,
    seed: int = 42,
    total_cap: int | None = MAX_SAMPLES_PER_DATASET,
) -> list[dict[str, Any]]:
    """Tiles-per-class balanced selection for dense multi-class rasters (spec 5).

    Each record lists the class ids present in its tile (``classes_key`` -> list[int]);
    a selected tile counts toward EVERY class it contains. Classes are filled from
    rarest to most common (prioritizing rare classes so they reach the target), adding
    shuffled tiles that contain the class until that class reaches ``per_class`` selected
    tiles. ``total_cap`` bounds the overall selection. Returns the selected records
    (deduplicated, in a deterministic order independent of input order).
    """
    keyfn = _as_keyfn(classes_key)
    records = _stable_order(records)  # deterministic regardless of input order
    rng = random.Random(seed)
    cand_counts: dict[int, int] = defaultdict(int)
    for r in records:
        for c in set(keyfn(r)):
            cand_counts[c] += 1
    order = sorted(cand_counts, key=lambda c: cand_counts[c])
    sel_ids: set[int] = set()
    sel_counts: dict[int, int] = defaultdict(int)
    for cls in order:
        if sel_counts[cls] >= per_class:
            continue
        cands = [
            i
            for i, r in enumerate(records)
            if cls in set(keyfn(r)) and i not in sel_ids
        ]
        rng.shuffle(cands)
        for i in cands:
            if sel_counts[cls] >= per_class:
                break
            if total_cap is not None and len(sel_ids) >= total_cap:
                break
            sel_ids.add(i)
            for c in set(keyfn(records[i])):
                sel_counts[c] += 1
        if total_cap is not None and len(sel_ids) >= total_cap:
            break
    return [records[i] for i in sorted(sel_ids)]


def bucket_balance_regression(
    records: list[dict[str, Any]],
    value_key: str | Callable[[dict[str, Any]], float],
    total: int = 5000,
    n_buckets: int = 10,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[float]]:
    """Return up to ``total`` records approximately balanced across value buckets.

    Returns (selected_records, bucket_edges). Use only when the raw value distribution
    is very skewed; otherwise a plain random sample of ``total`` is fine.
    """
    valfn = _as_keyfn(value_key)
    records = _stable_order(records)  # deterministic regardless of input order
    vals = np.array([valfn(r) for r in records], dtype=float)
    edges = list(np.quantile(vals, np.linspace(0, 1, n_buckets + 1)))
    idx_buckets: dict[int, list[int]] = defaultdict(list)
    for i, v in enumerate(vals):
        b = min(int(np.searchsorted(edges, v, side="right")) - 1, n_buckets - 1)
        idx_buckets[max(b, 0)].append(i)
    rng = random.Random(seed)
    per = max(1, total // max(1, len(idx_buckets)))
    out: list[dict[str, Any]] = []
    for b, idxs in idx_buckets.items():
        rng.shuffle(idxs)
        out.extend(records[i] for i in idxs[:per])
    return out[:total], edges


def select_tiles_per_class(
    records: list[dict[str, Any]],
    classes_key: str | Callable[[dict[str, Any]], Any] = "classes_present",
    per_class: int = 1000,
    total_cap: int | None = MAX_SAMPLES_PER_DATASET,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Tiles-per-class balanced selection for multi-label (dense_raster) tiles.

    Each record exposes an iterable of class ids present in the tile (via ``classes_key``,
    a property name or a function). Tiles are selected greedily, **rarest class first**, so
    sparse classes reach ``per_class`` before common ones consume the budget. A tile counts
    toward every class it contains. Selection stops for a class once it hits ``per_class``,
    and overall once ``total_cap`` tiles are selected. Returns the selected records in a
    deterministic order independent of input order.
    """
    keyfn = _as_keyfn(classes_key)
    records = _stable_order(records)  # deterministic regardless of input order
    freq: Counter = Counter()
    by_class: dict[Any, list[int]] = defaultdict(list)
    for i, r in enumerate(records):
        for c in keyfn(r):
            freq[c] += 1
            by_class[c].append(i)
    rng = random.Random(seed)
    for c in by_class:
        rng.shuffle(by_class[c])
    selected: set[int] = set()
    sel_counts: Counter = Counter()
    for c in sorted(freq, key=lambda c: freq[c]):
        if total_cap is not None and len(selected) >= total_cap:
            break
        for i in by_class[c]:
            if sel_counts[c] >= per_class:
                break
            if total_cap is not None and len(selected) >= total_cap:
                break
            if i in selected:
                continue
            selected.add(i)
            for cc in keyfn(records[i]):
                sel_counts[cc] += 1
    return [records[i] for i in sorted(selected)]


def encode_detection_tile(
    positives: list[tuple[int, int, int]],
    tile_size: int,
    positive_size: int = 1,
    buffer_size: int = 10,
    nodata: int = 255,
    background: int = 0,
) -> np.ndarray:
    """Build a (tile_size, tile_size) uint8 detection label.

    positives: list of (row, col, class_id) in tile-local pixel coords. Each detection
    is a positive_size square of class_id, ringed by a buffer_size band of nodata; all
    other pixels are background. Tunable per dataset.

    ``buffer_size`` should be **>= 10 px** (default 10): point/detection coordinates are
    rarely pixel-exact, so a thick nodata ring around each positive avoids penalizing the
    model for the true object landing a few pixels off. With positive_size=1 and
    buffer_size=10 the ignore region is 21x21 (center positive, rest nodata), which still
    leaves ample background in a 32x32 or 64x64 tile.
    """
    t = tile_size
    arr = np.full((t, t), background, dtype=np.uint8)
    # First lay down buffer rings, then positives on top so positives win.
    for r, c, _cid in positives:
        r0 = max(0, r - positive_size // 2 - buffer_size)
        r1 = min(t, r + positive_size // 2 + buffer_size + 1)
        c0 = max(0, c - positive_size // 2 - buffer_size)
        c1 = min(t, c + positive_size // 2 + buffer_size + 1)
        arr[r0:r1, c0:c1] = nodata
    for r, c, cid in positives:
        r0 = max(0, r - positive_size // 2)
        r1 = min(t, r + positive_size // 2 + 1)
        c0 = max(0, c - positive_size // 2)
        c1 = min(t, c + positive_size // 2 + 1)
        arr[r0:r1, c0:c1] = cid
    return arr
