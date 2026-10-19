"""Loader-only benchmark of the AllCaptures arms (CPU, no model; not a training run).

Builds config D's dataset and dataloader exactly as the arm does, at one rank's share
of the global batch (512 / 8 = 64), and iterates ``N`` batches. Prints one JSON line
per batch: seconds since the previous batch, the sampled patch size / grid / composite
mode, and per modality the collated tensor shape with its count of tokens by mask value
(so padded vs real tokens are visible). Usage: ``python bench_allcap_loader.py [N]
[num_workers]``; ``python bench_allcap_loader.py parts`` instead times the pieces of
one sample in one process (h5 layout and read cost, getitem per composite mode with a
profile, and the batch collate). With ``BENCH_SAMPLE_LOG=<dir>`` every loader worker
appends one JSON line per sample to ``<dir>/<pid>.jsonl``: wall time, process / thread CPU
time, and wall / CPU time inside h5py file opens and dataset reads (so waiting on
storage separates from decompression and compute); ``summarize <dir>`` aggregates them.
"""

import argparse
import cProfile
import io
import json
import os
import pstats
import sys
import tempfile
import time
from dataclasses import fields, is_dataclass
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import allcap_narrow as A  # noqa: E402
import allcap_slot_n4d192 as D  # noqa: E402

from olmoearth_pretrain.internal.experiment import SubCmd  # noqa: E402


def _instrument(log_dir: str) -> None:
    """Log per-sample wall / CPU / h5 time from each process (see module doc)."""
    import h5py

    from olmoearth_pretrain.data.dataset import OlmoEarthDataset

    os.makedirs(log_dir, exist_ok=True)
    h5t = {"wall": 0.0, "cpu": 0.0, "opens": 0, "reads": 0, "depth": 0}

    def timed(fn, key):  # type: ignore[no-untyped-def]
        # Only the outermost h5py call counts: a dataset read builds File wrappers
        # around the already-open file internally (not real opens).
        def wrapper(*a, **k):  # type: ignore[no-untyped-def]
            # File(<FileID>) wraps an open file (group lookups do this); not an open.
            wraps_open_file = len(a) > 1 and isinstance(a[1], h5py.h5f.FileID)
            if h5t["depth"] or wraps_open_file:
                return fn(*a, **k)
            h5t["depth"] = 1
            w, c = time.perf_counter(), time.thread_time()
            try:
                return fn(*a, **k)
            finally:
                h5t["depth"] = 0
                h5t["wall"] += time.perf_counter() - w
                h5t["cpu"] += time.thread_time() - c
                h5t[key] += 1

        return wrapper

    h5py.File.__init__ = timed(h5py.File.__init__, "opens")  # type: ignore[method-assign]
    h5py.Dataset.__getitem__ = timed(h5py.Dataset.__getitem__, "reads")  # type: ignore[method-assign]
    getitem = OlmoEarthDataset.__getitem__

    def logged(self, args):  # type: ignore[no-untyped-def]
        h5t.update(wall=0.0, cpu=0.0, opens=0, reads=0, depth=0)
        w, c, t = time.perf_counter(), time.process_time(), time.thread_time()
        out = getitem(self, args)
        row = {
            "pid": os.getpid(),
            "wall": time.perf_counter() - w,
            "proc_cpu": time.process_time() - c,
            "thread_cpu": time.thread_time() - t,
            "h5_wall": h5t["wall"],
            "h5_cpu": h5t["cpu"],
            "h5_opens": h5t["opens"],
            "h5_reads": h5t["reads"],
            "composite": args.composite_days,
            "hw_p": args.sampled_hw_p,
            "ps": args.patch_size,
            "torch_threads": torch.get_num_threads(),
        }
        with open(f"{log_dir}/{os.getpid()}.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
        return out

    OlmoEarthDataset.__getitem__ = logged  # type: ignore[method-assign]


def _summarize(log_dir: str) -> None:
    """Aggregate the per-sample logs of ``_instrument``."""
    rows = []
    for name in os.listdir(log_dir):
        with open(os.path.join(log_dir, name)) as f:
            rows += [json.loads(line) for line in f]
    out: dict = {"samples": len(rows), "processes": len({r["pid"] for r in rows})}
    for key in ("wall", "proc_cpu", "thread_cpu", "h5_wall", "h5_cpu"):
        v = np.array([r[key] for r in rows])
        out[key] = {
            "mean": round(float(v.mean()), 3),
            "p50": round(float(np.median(v)), 3),
            "p90": round(float(np.quantile(v, 0.9)), 3),
        }
    for mode in sorted({r["composite"] for r in rows}):
        sub = [r for r in rows if r["composite"] == mode]
        out[f"composite_{mode}"] = {
            k: round(float(np.mean([r[k] for r in sub])), 3)
            for k in ("wall", "proc_cpu", "h5_wall", "h5_cpu")
        } | {"n": len(sub)}
    out["torch_threads"] = sorted({r["torch_threads"] for r in rows})
    out["h5_opens_per_sample"] = round(float(np.mean([r["h5_opens"] for r in rows])), 2)
    print(json.dumps(out), flush=True)


if os.environ.get("BENCH_SAMPLE_LOG"):
    # Module level so spawned loader workers (which re-import this module) apply it.
    _instrument(os.environ["BENCH_SAMPLE_LOG"])


def _describe(batch: object) -> dict:
    """Shapes and per-mask-value token counts of every modality in a collated batch."""
    out: dict = {}
    items = batch if isinstance(batch, tuple | list) else (batch,)
    for item in items:
        if isinstance(item, int | float):
            out.setdefault("scalars", []).append(item)
            continue
        if not is_dataclass(item):
            continue
        for f in fields(item):
            value = getattr(item, f.name)
            if not isinstance(value, torch.Tensor):
                continue
            if f.name.endswith("_mask"):
                values, counts = np.unique(value.numpy(), return_counts=True)
                out.setdefault(f.name[: -len("_mask")], {})["mask_counts"] = {
                    str(int(v)): int(c) for v, c in zip(values, counts)
                }
            else:
                out.setdefault(f.name, {})["shape"] = list(value.shape)
    return out


def _parts() -> None:
    """Time the pieces of one sample in this process."""
    import h5py

    from olmoearth_pretrain.data.dataset import GetItemArgs

    common = A.build_common_components(
        D.MODULE_PATH, SubCmd.dry_run, "bench", "ai2/jupiter", []
    )
    dataset = D.build_dataset_config(common).build()
    dataset.prepare()
    config = D.build_dataloader_config(common)
    rng = np.random.default_rng(0)
    indices = rng.choice(len(dataset), 40, replace=False)
    # 1. h5 layout and read cost (whole dataset vs a 32 x 32 crop of every capture).
    for idx in indices[:4]:
        path = dataset._get_h5_file_path(int(idx))
        with path.open("rb") as f, h5py.File(f, "r") as h5:
            for name in h5:
                ds = h5[name]
                if not isinstance(ds, h5py.Dataset) or name.startswith("timestamps"):
                    continue
                t0 = time.perf_counter()
                ds[()]
                full = time.perf_counter() - t0
                t0 = time.perf_counter()
                if ds.ndim >= 3:
                    ds[:32, :32]
                crop = time.perf_counter() - t0
                print(
                    json.dumps(
                        {
                            "file": int(idx),
                            "name": name,
                            "shape": list(ds.shape),
                            "dtype": str(ds.dtype),
                            "chunks": ds.chunks,
                            "compression": ds.compression,
                            "filters": list(ds._filters),
                            "full_s": round(full, 4),
                            "crop32_s": round(crop, 4),
                            "mb": round(ds.size * ds.dtype.itemsize / 1e6, 1),
                        }
                    ),
                    flush=True,
                )
    # 2. getitem per composite mode (hw_p, time range drawn as the sampler does).
    for composite in config.composite_days_choices:
        profile = cProfile.Profile()
        times = []
        for idx in indices:
            patch_size = int(
                rng.integers(config.min_patch_size, config.max_patch_size + 1)
            )
            # Grids that fit the 128-pixel tile at this patch size (as the sampler).
            grids = [g for g in config.sampled_hw_p_list if g * patch_size <= 128]
            args = GetItemArgs(
                idx=int(idx),
                patch_size=patch_size,
                sampled_hw_p=int(rng.choice(grids)),
                token_budget=config.token_budget,
                tokenization_config=config.tokenization_config,
                time_range_days=float(rng.choice(config.time_range_days_choices)),
                composite_days=composite,
            )
            t0 = time.perf_counter()
            profile.enable()
            dataset[args]
            profile.disable()
            times.append(time.perf_counter() - t0)
        stream = io.StringIO()
        pstats.Stats(profile, stream=stream).sort_stats("cumulative").print_stats(18)
        print(
            json.dumps(
                {
                    "composite_days": composite,
                    "mean_s": round(float(np.mean(times)), 3),
                    "p90_s": round(float(np.quantile(times, 0.9)), 3),
                }
            ),
            flush=True,
        )
        print(stream.getvalue(), flush=True)
    # 3. collate + masking of one rank batch (64 samples at one shape).
    loader = config.build(dataset)
    for composite in config.composite_days_choices:
        items = [
            dataset[
                GetItemArgs(
                    idx=int(idx),
                    patch_size=2,
                    sampled_hw_p=8,
                    token_budget=config.token_budget,
                    tokenization_config=config.tokenization_config,
                    time_range_days=365.0,
                    composite_days=composite,
                )
            ]
            for idx in np.resize(indices, 64)
        ]
        t0 = time.perf_counter()
        loader.collator(items)
        print(
            json.dumps(
                {
                    "collate64_s": round(time.perf_counter() - t0, 3),
                    "composite_days": composite,
                }
            ),
            flush=True,
        )


def main() -> None:
    """Run the benchmark."""
    if len(sys.argv) > 1 and sys.argv[1] == "parts":
        _parts()
        return
    if len(sys.argv) > 2 and sys.argv[1] == "summarize":
        _summarize(sys.argv[2])
        return
    parser = argparse.ArgumentParser()
    parser.add_argument("n", type=int, nargs="?", default=60)
    parser.add_argument("workers", type=int, nargs="?", default=12)
    parser.add_argument("--h5py_dir", default=None)
    # Emulate one rank of a ``world``-rank node: global batch 64 * world, this rank's
    # slice of it (so concurrent ranks read disjoint samples, as in training).
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world", type=int, default=1)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    common = A.build_common_components(
        D.MODULE_PATH, SubCmd.dry_run, "bench", "ai2/jupiter", []
    )
    dataset_config = D.build_dataset_config(common)
    if args.h5py_dir is not None:
        dataset_config.h5py_dir = args.h5py_dir
    dataset = dataset_config.build()
    config = D.build_dataloader_config(common)
    config.global_batch_size = 64 * args.world
    config.num_workers = args.workers
    config.work_dir = tempfile.mkdtemp(prefix="bench_allcap_loader_")
    config.target_device_type = "cpu"
    loader = config.build(dataset)
    loader.dp_world_size = args.world
    loader.dp_rank = args.rank
    loader.reshuffle(epoch=1)
    t0 = time.perf_counter()
    last = t0
    for i, batch in enumerate(loader):
        now = time.perf_counter()
        row = {"rank": args.rank, "batch": i, "s": round(now - last, 3)}
        if not args.quiet:
            row.update(_describe(batch))
        print(json.dumps(row), flush=True)
        last = now
        if i + 1 >= args.n:
            break
    total = time.perf_counter() - t0
    print(
        json.dumps(
            {
                "rank": args.rank,
                "batches": args.n,
                "total_s": round(total, 1),
                "workers": args.workers,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
