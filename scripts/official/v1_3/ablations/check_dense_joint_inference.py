r"""GPU check of the mask-free joint attention: equivalence on a real checkpoint + speed.

For a trained joint-latent arm (default: rstride_fast2), on one GPU:

1. Equivalence. Registers and the shipped student output from four forwards of the
   same batch: the masked FlexAttention path in fp32 (the reference), FlexAttention
   under bf16 autocast (what evals run today), the dense path under bf16 autocast (the
   proposal), and the dense decomposition in fp32 with plain-torch kernels (checks the
   split itself at the real shape). Run on a fully visible batch and on one with
   missing Landsat timesteps (the eval's ``fast_pass=False`` path, with padding).
2. Speed. ms/window at the ws16 / T12 / S1+S2+L8 embedding window, ps1 and ps4 tokens,
   batch 64 (the eval batch) and 128, for both paths, and for the RC encoder shape
   (random weights) on the same GPU.
3. A kernel breakdown (torch.profiler) of one ps1 batch-64 forward for each path.

Usage (on a GPU node, from the repo root):
    python scripts/official/v1_3/ablations/check_dense_joint_inference.py \
        /weka/.../v1_3_vit0_rstride_ps8_lb512_fast2_latentread_joint12/step667200
Results print as RESULT / EQUIV lines and go to ``$RESULTS_DIR/dense_check.json``.
"""

import importlib
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import bench_inference_throughput as bench  # noqa: E402
from olmo_core.distributed.checkpoint import load_model_and_optim_state  # noqa: E402

import olmoearth_pretrain.nn.joint_latent as joint_latent  # noqa: E402
from olmoearth_pretrain.train.masking import (  # noqa: E402
    MaskedOlmoEarthSample,
    MaskValue,
)

ARM_MODULE = "pure_perceiver_joint_latentread_rstride_fast"
EQUIV_BATCH = 4
TIMING_BATCHES = [64, 128]
WARMUP = 10
ITERS = 30


def build_trained_encoder(step_dir: str) -> torch.nn.Module:
    """The arm's encoder with the checkpoint's weights, on CUDA in eval mode."""
    from base import build_common_components

    from olmoearth_pretrain.internal.common import SubCmd

    common = build_common_components("x.py", SubCmd.launch, "check", "ai2/jupiter", [])
    model = importlib.import_module(ARM_MODULE).build_model_config(common).build()
    load_model_and_optim_state(os.path.join(step_dir, "model_and_optim"), model)
    return model.encoder.cuda().eval()


def with_missing_landsat(x: MaskedOlmoEarthSample) -> MaskedOlmoEarthSample:
    """Mark the first four Landsat timesteps MISSING (padding inside the joint blocks)."""
    d = x.as_dict()
    mask = d["landsat_mask"].clone()
    mask[:, :, :, :4] = MaskValue.MISSING.value
    d["landsat_mask"] = mask
    return MaskedOlmoEarthSample.from_dict(d)


def forward(
    enc: torch.nn.Module,
    x: MaskedOlmoEarthSample,
    ps: int,
    *,
    dense: bool,
    autocast: bool,
    fast_pass: bool,
) -> dict[str, torch.Tensor]:
    """Registers and student output of one forward on the chosen attention path."""
    enc.perceiver.dense_inference_attention = dense
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16, enabled=autocast):
        out = enc(x, patch_size=ps, fast_pass=fast_pass)
    return {k: out[k].float() for k in ("registers", "student_registers") if k in out}


def compare(out: dict, ref: dict) -> dict:
    """Relative L2 error, max abs error and per-pixel cosine against the reference."""
    res = {}
    for key, r in ref.items():
        o = out[key]
        cos = F.cosine_similarity(o.flatten(0, -2), r.flatten(0, -2), dim=-1)
        res[key] = {
            "rel_l2": float((o - r).norm() / r.norm()),
            "max_abs": float((o - r).abs().max()),
            "cos_min": float(cos.min()),
            "cos_mean": float(cos.mean()),
        }
    return res


def equivalence(enc: torch.nn.Module) -> dict:
    """Each path vs the fp32 FlexAttention reference, visible and missing-token batches."""
    torch.manual_seed(0)
    results = {}
    for name, fast_pass in (("visible", True), ("missing_landsat", False)):
        x = bench.sample(EQUIV_BATCH, 12)
        if not fast_pass:
            x = with_missing_landsat(x)
        ref = forward(enc, x, 1, dense=False, autocast=False, fast_pass=fast_pass)
        runs = {
            "flex_bf16": forward(
                enc, x, 1, dense=False, autocast=True, fast_pass=fast_pass
            ),
            "dense_bf16": forward(
                enc, x, 1, dense=True, autocast=True, fast_pass=fast_pass
            ),
        }
        flash = joint_latent.flash_attn
        joint_latent.flash_attn = None  # plain-torch kernels: the split alone, fp32
        try:
            runs["dense_reference_fp32"] = forward(
                enc, x, 1, dense=True, autocast=False, fast_pass=fast_pass
            )
        finally:
            joint_latent.flash_attn = flash
        results[name] = {k: compare(v, ref) for k, v in runs.items()}
        for k, v in results[name].items():
            s = v["student_registers"]
            print(
                f"EQUIV {name:16s} {k:22s} student rel_l2 {s['rel_l2']:.2e} "
                f"max_abs {s['max_abs']:.2e} cos_min {s['cos_min']:.6f} "
                f"cos_mean {s['cos_mean']:.6f}",
                flush=True,
            )
    return results


def time_forward(fn, batch: int) -> float:
    """Ms per window over ITERS after WARMUP (CUDA events)."""
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(ITERS):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / ITERS / batch


def timings(enc: torch.nn.Module, rc: torch.nn.Module) -> dict:
    """ms/window for flex, dense and the RC encoder at the embedding-window shapes."""
    results: dict = {}
    for batch in TIMING_BATCHES:
        for ps in (1, 4):
            for missing in (False, True):
                x = bench.sample(batch, 12)
                if missing:
                    x = with_missing_landsat(x)
                fast_pass = not missing
                label = f"b{batch}_ps{ps}_{'missing' if missing else 'visible'}"
                row = {}
                for path, dense in (("flex", False), ("dense", True)):

                    def fn(dense: bool = dense) -> None:
                        enc.perceiver.dense_inference_attention = dense
                        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
                            enc(x, patch_size=ps, fast_pass=fast_pass)

                    try:
                        row[path] = time_forward(fn, batch)
                    except torch.OutOfMemoryError:
                        row[path] = None
                    torch.cuda.empty_cache()

                def rc_fn() -> None:
                    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
                        rc(x, patch_size=ps, fast_pass=fast_pass)

                try:
                    row["rc"] = time_forward(rc_fn, batch)
                except torch.OutOfMemoryError:
                    row["rc"] = None
                torch.cuda.empty_cache()
                results[label] = row
                fmt = lambda v: f"{v:7.3f}" if v else "    OOM"  # noqa: E731
                print(
                    f"RESULT {label:20s} ms/window flex {fmt(row['flex'])} dense "
                    f"{fmt(row['dense'])} rc {fmt(row['rc'])}  (x64 = ms/eval batch)",
                    flush=True,
                )
    return results


def kernel_breakdown(enc: torch.nn.Module) -> None:
    """Print the top CUDA kernels of one ps1 batch-64 forward for each path."""
    x = bench.sample(64, 12)
    for path, dense in (("flex", False), ("dense", True)):
        enc.perceiver.dense_inference_attention = dense
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            for _ in range(3):
                enc(x, patch_size=1, fast_pass=True)
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                enc(x, patch_size=1, fast_pass=True)
                torch.cuda.synchronize()
        print(f"PROFILE {path} (one ps1 batch-64 forward)", flush=True)
        print(
            prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=15),
            flush=True,
        )


def main() -> None:
    """Equivalence, timings and kernel breakdown for the checkpoint in argv[1]."""
    step_dir = sys.argv[1]
    print(f"torch {torch.__version__}, {torch.cuda.get_device_name()}", flush=True)
    enc = build_trained_encoder(step_dir)
    results = {"equivalence": equivalence(enc)}
    rc = bench.build_encoder("rc")
    results["timings_ms_per_window"] = timings(enc, rc)
    kernel_breakdown(enc)
    out = os.environ.get("RESULTS_DIR")
    if out:
        Path(out).mkdir(parents=True, exist_ok=True)
        with open(Path(out) / "dense_check.json", "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
