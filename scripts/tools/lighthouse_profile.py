"""Time Lighthouse variants on one real chunk, with a per-phase breakdown.

Variants: mask implementation (``gather`` reference vs ``packed``) x attention dtype
(whatever RoPE returns vs bf16). Each runs once to warm up and then ``--reps``
times; prints seconds, the attention / QKV+RoPE / proj+MLP split, peak memory, and
the deviation of the d128 output from the first variant.
"""

import argparse
import sys
import time

import torch

sys.path.insert(0, "scripts/tools")
from lighthouse_aoi_inference import (  # noqa: E402
    WindowSamples,
    _batched,
    _crop,
    build_window_dataset,
)

from olmoearth_pretrain.model_loader import load_pretrain_checkpoint  # noqa: E402
from olmoearth_pretrain.nn.lighthouse import LighthouseSettings  # noqa: E402


def main() -> None:
    """Run the variants."""
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--name", default="barbizon_apremont")
    p.add_argument("--reps", type=int, default=2)
    a = p.parse_args()
    dev = torch.device("cuda")
    encoder = load_pretrain_checkpoint(a.checkpoint, device=dev).encoder
    _, sample, _ = WindowSamples(build_window_dataset(a.dataset, "predict", [a.name]))[
        0
    ]
    full = _batched(sample, dev)
    variants = [("gather", None), ("packed", None), ("packed", "bfloat16")]
    for ps, (h, w) in ((1, (200, 304)), (2, (360, 360)), (4, (360, 360))):
        chunk = _crop(full, slice(0, h), slice(0, w))
        ref = None
        for mask_impl, dtype in variants:
            settings = LighthouseSettings(
                fov_px=16, mask_impl=mask_impl, attn_dtype=dtype, profile=True
            )
            for rep in range(a.reps + 1):
                encoder.perceiver.lighthouse = settings
                torch.cuda.reset_peak_memory_stats(dev)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    out = encoder(chunk, patch_size=ps, input_res=10, fast_pass=True)
                torch.cuda.synchronize()
                dt = time.perf_counter() - t0
                encoder.perceiver.lighthouse = None
                stats = encoder.perceiver.last_lighthouse_stats
                emb = out["student_registers"][0, ..., :128].float()
                if ref is None:
                    ref = emb
                d = (emb - ref).abs().max().item()
                cos = (
                    torch.nn.functional.cosine_similarity(emb, ref, dim=-1).min().item()
                )
                print(
                    f"ps{ps} {h}x{w} mask={mask_impl:6s} dtype={dtype!s:8s} "
                    f"{'warm' if rep == 0 else 'rep' + str(rep)}: {dt:7.2f} s | "
                    f"attn {stats.get('attention_s', 0):6.2f} qkv+rope "
                    f"{stats.get('qkv_rope_s', 0):6.2f} proj+mlp "
                    f"{stats.get('proj_mlp_s', 0):6.2f} | elements "
                    f"{stats['elements']:.0f} kv/q {stats['kv_blocks_mean']:.1f} | "
                    f"peak {torch.cuda.max_memory_allocated(dev) / 2**30:5.1f} GiB | "
                    f"vs first: max {d:.2e} min cos {cos:.6f}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
