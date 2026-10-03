"""Where Lighthouse time goes vs tiled, per kernel category (torch.profiler).

Random-init model from a run's ``config.json``; one real crop; for each token patch
size, profiles (after a warm-up) one Lighthouse chunk (``side + 2 * halo`` px) and
the tiled forward over the ``side`` px core (16 px / overlap 4, batch 64,
``fast_pass``), and reports per run:

* wall seconds, summed GPU kernel seconds, and GPU idle = wall - kernels (Python,
  launch and sync overhead);
* kernel seconds by category -- window attention (FlexAttention / NATTEN), dense
  attention (flash / cuDNN / SDPA), GEMM, copy + indexing, elementwise + norm,
  other -- and the top kernels;
* everything also normalized per processed token, so the two runs compare per unit
  of work (tiled processes each pixel ~1.77x, Lighthouse ~(chunk/side)^2).

Chrome traces go to ``--trace_dir``.
"""

import argparse
import json
import sys
import time
from collections import defaultdict

import torch
from torch.profiler import ProfilerActivity, profile
from upath import UPath

sys.path.insert(0, "scripts/tools")
from lighthouse_aoi_inference import (  # noqa: E402
    WindowSamples,
    _batched,
    _crop,
    _stack_crops,
    build_window_dataset,
)
from rslearn.train.all_crops_dataset import get_window_crop_options  # noqa: E402

from olmoearth_pretrain.model_loader import _load_model_from_config  # noqa: E402
from olmoearth_pretrain.nn.lighthouse_rc import RCLighthouseSettings  # noqa: E402

CATEGORIES = [
    ("window_attention", ("flex_attention", "triton_tem", "natten", "fna")),
    ("dense_attention", ("flash", "fmha", "cudnn", "sdpa", "attention")),
    ("gemm", ("gemm", "sm90_xmma", "sm80_xmma", "cutlass", "nvjet", "matmul")),
    (
        "copy_index",
        ("copy", "index", "scatter", "gather", "cat", "fill", "memcpy", "memset"),
    ),
    (
        "elementwise_norm",
        (
            "elementwise",
            "vectorized",
            "reduce",
            "norm",
            "triton_",
            "softmax",
            "gelu",
            "mul",
            "add",
        ),
    ),
]


def category(name: str) -> str:
    """Kernel category from its name."""
    low = name.lower()
    for cat, keys in CATEGORIES:
        if any(k in low for k in keys):
            return cat
    return "other"


def forward(encoder, sample, ps, settings=None, fast_pass=False):
    """Encoder outputs (stock forward when ``settings`` is None)."""
    encoder.lighthouse = settings
    try:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            return encoder(sample, patch_size=ps, input_res=10, fast_pass=fast_pass)
    finally:
        encoder.lighthouse = None


def profile_run(fn, trace_path: str) -> dict:
    """Wall time, kernel time by category, top kernels of one call of ``fn``."""
    fn()  # warm-up (compile, kernel selection)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
    prof.export_chrome_trace(trace_path)
    by_cat: dict[str, float] = defaultdict(float)
    by_kernel: dict[str, float] = defaultdict(float)
    for evt in prof.events():
        if evt.device_type != torch.autograd.DeviceType.CUDA:
            continue
        sec = (
            evt.device_time / 1e6
            if hasattr(evt, "device_time")
            else evt.cuda_time / 1e6
        )
        by_cat[category(evt.name)] += sec
        by_kernel[evt.name[:90]] += sec
    kernels = sum(by_cat.values())
    top = sorted(by_kernel.items(), key=lambda kv: -kv[1])[:15]
    return {
        "wall_s": wall,
        "kernel_s": kernels,
        "gpu_idle_s": max(wall - kernels, 0.0),
        "by_category_s": {k: round(v, 4) for k, v in sorted(by_cat.items())},
        "top_kernels_s": [[k, round(v, 4)] for k, v in top],
    }


def main() -> None:
    """Profile Lighthouse and tiled at each patch size."""
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--name", default="sundarbans_tidal")
    p.add_argument("--pss", type=int, nargs="+", default=[4, 2])
    p.add_argument("--side", type=int, nargs="+", default=[768, 448])
    p.add_argument("--halo", type=int, default=16)
    p.add_argument("--trace_dir", required=True)
    p.add_argument("--out_json", default=None)
    a = p.parse_args()
    dev = torch.device("cuda")
    torch.manual_seed(0)
    encoder = _load_model_from_config(UPath(a.config)).encoder.to(dev).eval()
    _, sample, _ = WindowSamples(build_window_dataset(a.dataset, "predict", [a.name]))[
        0
    ]
    full = _batched(sample, dev)
    results = []
    for ps, side in zip(a.pss, a.side):
        chunk = side + 2 * a.halo
        crop_dict = {
            k: (v if k == "timestamps" else v[:, 64 : 64 + chunk, 64 : 64 + chunk])
            for k, v in full.items()
        }
        core_dict = {
            k: (
                v
                if k == "timestamps"
                else v[:, a.halo : a.halo + side, a.halo : a.halo + side]
            )
            for k, v in crop_dict.items()
        }
        crop = _crop(crop_dict, slice(0, chunk), slice(0, chunk))
        boxes = get_window_crop_options((16, 16), (4, 4), (0, 0, side, side))

        def tiled():
            for i in range(0, len(boxes), 64):
                forward(
                    encoder,
                    _stack_crops(core_dict, boxes[i : i + 64]),
                    ps,
                    fast_pass=True,
                )

        settings = RCLighthouseSettings(fov_px=16)
        lh = profile_run(
            lambda: forward(encoder, crop, ps, settings),
            f"{a.trace_dir}/lighthouse_ps{ps}.json",
        )
        forward(encoder, crop, ps, RCLighthouseSettings(16, profile=True))
        tokens_lh = encoder.last_lighthouse_stats["tokens"]
        ti = profile_run(tiled, f"{a.trace_dir}/tiled_ps{ps}.json")
        tokens_ti = tokens_lh * (len(boxes) * 256) / (chunk * chunk)
        km2 = side * side / 1e4
        for name, r, tok in (("lighthouse", lh, tokens_lh), ("tiled", ti, tokens_ti)):
            row = {
                "run": name,
                "ps": ps,
                "core_px": side,
                "s_per_km2_output": r["wall_s"] / km2,
                "processed_tokens": tok,
                "us_per_token_wall": r["wall_s"] / tok * 1e6,
                "us_per_token_by_category": {
                    k: round(v / tok * 1e6, 4) for k, v in r["by_category_s"].items()
                },
                **r,
            }
            results.append(row)
            print(json.dumps(row), flush=True)
        torch.cuda.empty_cache()
    if a.out_json:
        with open(a.out_json, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
