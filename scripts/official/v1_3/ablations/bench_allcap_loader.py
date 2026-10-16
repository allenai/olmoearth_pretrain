"""Loader-only benchmark of the AllCaptures arms (CPU, no model; not a training run).

Builds config D's dataset and dataloader exactly as the arm does, at one rank's share
of the global batch (512 / 8 = 64), and iterates ``N`` batches. Prints one JSON line
per batch: seconds since the previous batch, the sampled patch size / grid / composite
mode, and per modality the collated tensor shape with its count of tokens by mask value
(so padded vs real tokens are visible). Usage: ``python bench_allcap_loader.py [N]
[num_workers]``; ``python bench_allcap_loader.py parts`` instead times the pieces of
one sample in one process (h5 layout and read cost, getitem per composite mode with a
profile, and the batch collate).
"""

import cProfile
import io
import json
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
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    common = A.build_common_components(
        D.MODULE_PATH, SubCmd.dry_run, "bench", "ai2/jupiter", []
    )
    dataset = D.build_dataset_config(common).build()
    config = D.build_dataloader_config(common)
    config.global_batch_size = 64
    config.num_workers = workers
    config.work_dir = tempfile.mkdtemp(prefix="bench_allcap_loader_")
    config.target_device_type = "cpu"
    loader = config.build(dataset)
    loader.reshuffle(epoch=1)
    t0 = time.perf_counter()
    last = t0
    for i, batch in enumerate(loader):
        now = time.perf_counter()
        row = {"batch": i, "s": round(now - last, 3), **_describe(batch)}
        print(json.dumps(row), flush=True)
        last = now
        if i + 1 >= n:
            break
    total = time.perf_counter() - t0
    print(
        json.dumps({"batches": n, "total_s": round(total, 1), "workers": workers}),
        flush=True,
    )


if __name__ == "__main__":
    main()
