"""Loader-only benchmark of the AllCaptures arms (CPU, no model; not a training run).

Builds config D's dataset and dataloader exactly as the arm does, at one rank's share
of the global batch (512 / 8 = 64), and iterates ``N`` batches. Prints one JSON line
per batch: seconds since the previous batch, the sampled patch size / grid / composite
mode, and per modality the collated tensor shape with its count of tokens by mask value
(so padded vs real tokens are visible). Usage: ``python bench_allcap_loader.py [N]
[num_workers]``.
"""

import json
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


def main() -> None:
    """Run the benchmark."""
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
