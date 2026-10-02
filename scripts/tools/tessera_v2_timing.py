"""Time Tessera v2 (released student) inference per km^2, comparably to the AOI runs.

Tessera v2 embeds each pixel from its own year of observations: every S2 scene
(masked by SCL) and every merged S1 ascending + descending pass, each pixel's series
binned to 8..256 steps. Its cost therefore depends on how many observations a pixel
has, so this times it on REAL fetched windows (the PASTIS Tessera fetch: every 2019
acquisition over its 128 px windows, assembled exactly as our tessera_v2 export
does) and also on a synthetic sweep of observation counts.

Two clocks per real window, both CUDA-synchronized:

* ``model_s``: time inside ``model.encode`` only, the analogue of the AOI runs'
  model-forward time;
* ``pipeline_s``: the whole ``encode_tile`` call, which adds Tessera's per-pixel
  CPU binning / gathering and host-device copies (I/O excluded in both).

Each is measured in Tessera's own precision (fp32, as our export ran it) and under
bf16 autocast (as the OlmoEarth AOI runs ran). Prints a table and writes JSON.
"""

from __future__ import annotations

import argparse
import json
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from rslearn.dataset import Dataset
from upath import UPath

from olmoearth_pretrain.evals.datasets.tessera_v2_export import build_dpixel_inputs
from olmoearth_pretrain.evals.models.tessera.tessera_v2_infer import encode_tile
from olmoearth_pretrain.evals.models.tessera.tessera_v2_model import load_model


class TimedEncode:
    """Wrap ``model.encode`` and accumulate CUDA-synchronized seconds."""

    def __init__(self, model: torch.nn.Module, autocast: bool) -> None:
        """Wrap ``model``; ``autocast`` runs encode under bf16 autocast."""
        self.model = model
        self.inner = model.encode
        self.autocast = autocast
        self.seconds = 0.0

    def __call__(self, *a: object, **k: object) -> torch.Tensor:
        """Run and time one encode call."""
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        ctx = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if self.autocast
            else nullcontext()
        )
        with ctx:
            out = self.inner(*a, **k)
        torch.cuda.synchronize()
        self.seconds += time.perf_counter() - t0
        return out


def run_tile(
    model: torch.nn.Module, inputs: dict, batch_pixels: int, autocast: bool
) -> tuple[float, float]:
    """(pipeline seconds, model seconds) for one encode_tile call."""
    timer = TimedEncode(model, autocast)
    model.encode = timer  # type: ignore[method-assign]
    try:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        encode_tile(
            model, **inputs, batch_pixels=batch_pixels, device=torch.device("cuda")
        )
        torch.cuda.synchronize()
        return time.perf_counter() - t0, timer.seconds
    finally:
        model.encode = timer.inner  # type: ignore[method-assign]


def obs_counts(inputs: dict) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel valid S2 count and merged S1 count (what Tessera bins on)."""
    s2 = (
        inputs["s2_masks"].sum(0)
        if inputs.get("s2_masks") is not None
        else np.full(inputs["s2_bands"].shape[1:3], inputs["s2_bands"].shape[0])
    )
    s1 = np.zeros_like(s2)
    for k in ("s1_asc_bands", "s1_desc_bands"):
        b = inputs.get(k)
        if b is not None and b.size:
            s1 = s1 + (np.abs(b).sum(-1) > 0).sum(0)
    return s2, s1


def synthetic(h: int, w: int, n_s2: int, n_s1: int, rng: np.random.Generator) -> dict:
    """A tile where every pixel has n_s2 valid S2 and n_s1 S1 observations."""

    def doys(n: int) -> np.ndarray:
        return np.sort(rng.choice(np.arange(1, 366), size=n, replace=n > 365)).astype(
            np.int64
        )

    na, nd = n_s1 // 2, n_s1 - n_s1 // 2
    return {
        "s2_bands": rng.integers(200, 4000, size=(n_s2, h, w, 10)).astype(np.uint16),
        "s2_doys": doys(n_s2),
        "s2_masks": np.ones((n_s2, h, w), dtype=np.uint8),
        "s1_asc_bands": rng.integers(3000, 9000, size=(na, h, w, 2)).astype(np.int16),
        "s1_asc_doys": doys(na),
        "s1_desc_bands": rng.integers(3000, 9000, size=(nd, h, w, 2)).astype(np.int16),
        "s1_desc_doys": doys(nd),
    }


def main() -> None:
    """Run the real-window and synthetic timings."""
    p = argparse.ArgumentParser()
    p.add_argument(
        "--checkpoint",
        default="/weka/dfive-default/helios/models/tessera_v2/ckpt/student_large.pt",
    )
    p.add_argument(
        "--ds_path", default="/weka/dfive-default/rslearn-eai/datasets/pastis_rslearn"
    )
    p.add_argument("--fetch_group", default="pastis_tessera_v2")
    p.add_argument("--windows", type=int, default=24)
    p.add_argument("--batch_pixels", type=int, default=4096)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    dev = torch.device("cuda")
    model = load_model(a.checkpoint, device=dev)
    model.eval()
    gpu = torch.cuda.get_device_name(dev)
    print(
        f"gpu {gpu} | checkpoint {a.checkpoint} | repr_dim {model.repr_dim}", flush=True
    )

    ds = Dataset(UPath(a.ds_path))
    windows = sorted(ds.load_windows(groups=[a.fetch_group]), key=lambda w: w.name)[
        : a.windows + 1
    ]
    record: dict = {
        "gpu": gpu,
        "checkpoint": a.checkpoint,
        "batch_pixels": a.batch_pixels,
        "real": [],
        "synthetic": [],
    }

    # Warm up on the first window (kernels, allocator), untimed.
    first = build_dpixel_inputs(windows[0], allow_unmaterialized_s1=True)
    for ac in (False, True):
        run_tile(model, first, a.batch_pixels, ac)

    for w in windows[1:]:
        try:
            inputs = build_dpixel_inputs(
                w, allow_unmaterialized_s1=True
            )  # I/O: untimed
        except Exception as e:  # noqa: BLE001 - a window without scenes is skipped
            print(f"skip {w.name}: {e}", flush=True)
            continue
        h, wd = inputs["s2_bands"].shape[1:3]
        s2n, s1n = obs_counts(inputs)
        row = {
            "window": w.name,
            "H": int(h),
            "W": int(wd),
            "km2": h * wd / 1e4,
            "s2_valid_median": float(np.median(s2n)),
            "s1_median": float(np.median(s1n)),
        }
        for ac, tag in ((False, "fp32"), (True, "bf16")):
            pipe, mod = run_tile(model, inputs, a.batch_pixels, ac)
            row[f"pipeline_s_{tag}"], row[f"model_s_{tag}"] = pipe, mod
        record["real"].append(row)
        print(json.dumps(row), flush=True)

    rng = np.random.default_rng(0)
    for n_s2, n_s1 in ((16, 16), (32, 32), (64, 64), (96, 96), (128, 128), (256, 256)):
        tile = synthetic(128, 128, n_s2, n_s1, rng)
        row = {"n_s2": n_s2, "n_s1": n_s1, "km2": 128 * 128 / 1e4}
        for ac, tag in ((False, "fp32"), (True, "bf16")):
            run_tile(model, tile, a.batch_pixels, ac)  # warm this bin shape
            pipe, mod = run_tile(model, tile, a.batch_pixels, ac)
            row[f"pipeline_s_{tag}"], row[f"model_s_{tag}"] = pipe, mod
        record["synthetic"].append(row)
        print(json.dumps(row), flush=True)

    real = record["real"]
    km2 = sum(r["km2"] for r in real)
    summary = {}
    for k in ("model_s_fp32", "model_s_bf16", "pipeline_s_fp32", "pipeline_s_bf16"):
        summary[k.replace("_s_", "_s_per_km2_")] = (
            sum(r[k] for r in real) / km2 if km2 else None
        )
    summary["windows"] = len(real)
    summary["km2"] = km2
    summary["s2_valid_median"] = (
        float(np.median([r["s2_valid_median"] for r in real])) if real else None
    )
    summary["s1_median"] = (
        float(np.median([r["s1_median"] for r in real])) if real else None
    )
    record["summary"] = summary
    Path(a.out).write_text(json.dumps(record, indent=1))
    print("SUMMARY", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
