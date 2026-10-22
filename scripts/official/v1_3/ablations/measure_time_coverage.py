"""Time covered by one training sample, per arm and mode (CPU only, not training).

Draws ``N`` samples through each arm's real sampler and dataset (grid, time range,
composite mode, token budget) and prints one JSON line per sample: the mode, grid,
time-range draw, union time steps, calendar span (first to last step, months),
distinct calendar months, and real tokens per cell. ``v1_3`` uses v1.3's monthly
corpus and sampler; ``allcap`` uses the D' arm (``allcap_slot_n4d192_fe``).

For every allcap sample it also simulates a "thinned" mode from the same tile and
grid: the whole series (or the last 365 / 730 days), with union steps drawn at random
without replacement until the next one would exceed the token budget.

Usage: ``python measure_time_coverage.py {v1_3,allcap} N [procs]``; ``summarize
FILE...`` prints the per-mode distributions.
"""

import importlib
import json
import sys
from collections import defaultdict
from multiprocessing import get_context
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from olmoearth_pretrain.data.constants import MISSING_VALUE, Modality  # noqa: E402
from olmoearth_pretrain.data.dataloader import _IterableDatasetWrapper  # noqa: E402
from olmoearth_pretrain.data.dataset import (  # noqa: E402
    build_union_timeline,
    time_index_field,
    timestamps_to_ordinals,
)
from olmoearth_pretrain.internal.experiment import SubCmd  # noqa: E402

TIME_MODALITIES = ("sentinel2_l2a", "sentinel1", "landsat", "landsat_l2")
THIN_WINDOWS_DAYS = (None, 365.0, 730.0)

_STATE: dict = {}


def _build(arm: str):  # type: ignore[no-untyped-def]
    if arm == "v1_3":
        module = importlib.import_module("base")
        script = "scripts/official/v1_3/base.py"
    else:
        module = importlib.import_module("allcap_slot_n4d192_fe")
        script = module.MODULE_PATH
    narrow = importlib.import_module("allcap_narrow") if arm != "v1_3" else module
    common = narrow.build_common_components(
        script, SubCmd.dry_run, "coverage", "ai2/jupiter", []
    )
    dataset_config = (
        narrow.build_dataset_config(common)
        if arm != "v1_3"
        else module.build_dataset_config(common)
    )
    dataset = dataset_config.build()
    dataset.prepare()
    config = module.build_dataloader_config(common)
    config.work_dir = "/tmp/coverage_work"
    config.target_device_type = "cpu"
    config.num_workers = 0
    loader = config.build(dataset)
    return dataset, loader


def _months(timestamps: np.ndarray) -> tuple[int, int, float]:
    """``[day, month-1, year]`` rows -> (calendar span in months, distinct months, days)."""
    if len(timestamps) == 0:
        return 0, 0, 0.0
    month = timestamps[:, 2].astype(int) * 12 + timestamps[:, 1].astype(int)
    ordinals = timestamps_to_ordinals(timestamps)
    return (
        int(month.max() - month.min() + 1),
        len(np.unique(month)),
        float(ordinals.max() - ordinals.min()),
    )


def _thin(dataset, loader, idx: int, hw_p: int, rng) -> list[dict]:  # type: ignore[no-untyped-def]
    """The simulated thinned mode for this tile and grid, per window."""
    path = dataset._local_h5_path(dataset._get_h5_file_path(idx))
    with h5py.File(path, "r") as f:
        timestamps = {
            m: f[f"timestamps_{m}"][()]
            for m in dataset.training_modalities
            if m in f and Modality.get(m).is_multitemporal and m in TIME_MODALITIES
        }
    rows = []
    for window in THIN_WINDOWS_DAYS:
        kept = timestamps
        if window is not None:
            ordinals = {m: timestamps_to_ordinals(t) for m, t in timestamps.items()}
            last = max(int(o.max()) for o in ordinals.values() if len(o))
            kept = {m: t[ordinals[m] > last - window] for m, t in timestamps.items()}
        union, step_index = build_union_timeline(kept)
        cost = np.zeros(len(union), dtype=np.int64)
        for m, steps in step_index.items():
            cost[steps] += hw_p * hw_p  # one band set per modality (tokenization)
        order = rng.permutation(len(union))
        total, chosen = 0, []
        for s in order:
            if total + cost[s] > loader.token_budget:
                break
            total += cost[s]
            chosen.append(s)
        chosen = np.sort(np.array(chosen, dtype=np.int64))
        span, distinct, days = _months(union[chosen])
        rows.append(
            {
                "mode": f"thin_{int(window) if window else 'all'}",
                "steps": len(chosen),
                "span_months": span,
                "distinct_months": distinct,
                "span_days": days,
                "tokens_per_cell": int(total // max(hw_p * hw_p, 1)),
                "series_days": _months(union)[2],
            }
        )
    return rows


def _init(arm: str) -> None:
    _STATE["dataset"], _STATE["loader"] = _build(arm)
    _STATE["arm"] = arm


def _measure(item) -> list[dict]:  # type: ignore[no-untyped-def]
    idx, patch_size, hw_p, target_t, time_range, composite = item
    dataset, loader, arm = _STATE["dataset"], _STATE["loader"], _STATE["arm"]
    args = loader._get_item_args(
        idx, patch_size, hw_p, target_t, time_range, composite, seed=idx
    )
    _, sample = dataset[args]
    timestamps = np.asarray(sample.timestamps)
    tokens = 0
    for m in TIME_MODALITIES:
        value = getattr(sample, m, None)
        if value is None:
            continue
        if arm == "v1_3":
            # [H, W, T, C]: a timestep counts if any pixel holds data.
            # v1.3's model makes one token per modality and timestep.
            v = np.asarray(value)
            present = (v != MISSING_VALUE).reshape(-1, v.shape[2], v.shape[3])
            tokens += int(present.any(axis=(0, 2)).sum())
        else:
            tokens += len(getattr(sample, time_index_field(m)))
    span, distinct, days = _months(timestamps)
    base = {"arm": arm, "idx": int(idx), "patch_size": patch_size, "hw_p": hw_p}
    rows = [
        {
            **base,
            "mode": "monthly_v1_3"
            if arm == "v1_3"
            else {0.0: "every_capture", 14.0: "biweekly", 30.0: "monthly"}[composite],
            "time_range_days": time_range,
            "target_t": target_t,
            "steps": len(timestamps),
            "span_months": span,
            "distinct_months": distinct,
            "span_days": days,
            "tokens_per_cell": tokens,
        }
    ]
    if arm != "v1_3":
        rng = np.random.default_rng(idx)
        rows += [{**base, **r} for r in _thin(dataset, loader, idx, hw_p, rng)]
    return rows


def _summarize(files: list[str]) -> None:
    groups = defaultdict(list)
    for name in files:
        for line in open(name):
            if line.startswith("{"):
                row = json.loads(line)
                groups[(row["arm"], row["mode"])].append(row)
    for (arm, mode), rows in sorted(groups.items()):
        out = {"arm": arm, "mode": mode, "n": len(rows)}
        for key in ("span_months", "distinct_months", "steps", "tokens_per_cell"):
            v = np.array([r[key] for r in rows], dtype=float)
            out[key] = {
                "mean": round(v.mean(), 1),
                "p10": float(np.percentile(v, 10)),
                "p50": float(np.percentile(v, 50)),
                "p90": float(np.percentile(v, 90)),
            }
        span = np.array([r["span_months"] for r in rows])
        out["frac_span_ge_6"] = round(float((span >= 6).mean()), 3)
        out["frac_span_ge_12"] = round(float((span >= 12).mean()), 3)
        print(json.dumps(out))
        if "time_range_days" in rows[0]:
            by_range = defaultdict(list)
            for r in rows:
                by_range[r["time_range_days"]].append(r["span_months"])
            print(
                "   span_months p50 by time_range_days:",
                {k: float(np.median(v)) for k, v in sorted(by_range.items(), key=str)},
            )


def main() -> None:
    """Run the measurement (see the module docstring)."""
    if sys.argv[1] == "summarize":
        _summarize(sys.argv[2:])
        return
    arm, n = sys.argv[1], int(sys.argv[2])
    procs = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    dataset, loader = _build(arm)
    wrapper = _IterableDatasetWrapper(loader)
    rng = np.random.default_rng(0)
    indices = rng.choice(len(dataset), n, replace=False)
    # Grid / composite mode are drawn per rank batch; 8-sample batches keep the
    # marginal distribution and give many independent draws.
    params = list(
        wrapper._get_batch_item_params_iterator(
            indices, loader.patch_sizes, loader.sampled_hw_p_list, 8
        )
    )
    with get_context("fork").Pool(procs, initializer=_init, initargs=(arm,)) as pool:
        for rows in pool.imap_unordered(_measure, params, chunksize=4):
            for row in rows:
                print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
