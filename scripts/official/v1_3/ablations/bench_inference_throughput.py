"""Single-GPU INFERENCE throughput benchmark for the shape arms (not a training run).

Measures windows/second for ``encoder(sample, patch_size, fast_pass=True)`` under
``torch.no_grad`` + bf16 autocast in eval mode, with randomly initialised weights (speed
does not depend on the values). The window is the embedding-product shape: 16 x 16
pixels, S2 L2A + S1 + Landsat, 12 monthly timesteps. "7 modalities" is emulated as the
same 3 modalities x 28 timesteps -- every single-bandset multi-timestep modality adds T
tokens per pixel, so the token count matches.

For each model and shape the largest batch in ``BATCH_SIZES`` that fits is used; each
run warms up ``WARMUP`` iterations (so torch.compile / FlexAttention recompiles for the
shape settle) and times ``ITERS`` iterations with CUDA events.

Usage (on a GPU node, from the repo root):
    python scripts/official/v1_3/ablations/bench_inference_throughput.py rc joint_rstride ...
Model keys: see ``MODELS`` (arm-script modules) plus ``rc`` (the mlpgram1 encoder shape,
from ``scripts/tools/20251111_flops.py``). Results print as a table and are written to
``$RESULTS_DIR/bench_inference.json`` when that is set.
"""

import importlib
import json
import os
import sys
import time
import types
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "experiments"))
sys.path.insert(0, str(HERE.parents[2] / "tools"))

from olmoearth_pretrain.data.constants import Modality  # noqa: E402
from olmoearth_pretrain.train.masking import (  # noqa: E402
    MaskedOlmoEarthSample,
    MaskValue,
)

# Model key -> arm module (its build_model_config gives the encoder).
MODELS = {
    "joint_rstride": "pure_perceiver_joint_latentread_rstride",
    "joint_latentread": "pure_perceiver_joint_latentread",
    "mix8pre_rl2": "pure_perceiver_mix8pre_rl2",
    "mix5pre_read4": "pure_perceiver_mix5pre_read4",
    "mix6_rl6": "pure_perceiver_mix6_read6",
    "mix6d128_rl6": "pure_perceiver_mix6_read6_d128",
    "favyen_pixreg_w1": "pixreg_pixrecon_w1",
}
# (label, patch size, timesteps): T=12 is 3 modalities, T=28 emulates 7.
SHAPES = [
    ("ps1_3mod", 1, 12),
    ("ps4_3mod", 4, 12),
    ("ps1_7mod", 1, 28),
    ("ps4_7mod", 4, 28),
]
BATCH_SIZES = [512, 256, 128, 64, 32, 16, 8]
WARMUP = 20
ITERS = 50
WINDOW = 16


def build_encoder(key: str) -> torch.nn.Module:
    """The model's encoder, randomly initialised, on CUDA in eval mode."""
    if key == "rc":
        sys.modules.setdefault(
            "thop", types.SimpleNamespace(clever_format=lambda *a, **k: a)
        )
        enc = importlib.import_module("20251111_flops").build_v1_3_rc_encoder(True)
    else:
        from base import build_common_components  # the arm's v1.3 base

        from olmoearth_pretrain.internal.common import SubCmd

        common = build_common_components(
            "x.py", SubCmd.launch, "bench", "ai2/jupiter", []
        )
        cfg = importlib.import_module(MODELS[key]).build_model_config(common)
        cfg.encoder_config.max_sequence_length = max(t for _, _, t in SHAPES)
        enc = cfg.encoder_config.build()
    return enc.cuda().eval()


def sample(batch: int, timesteps: int) -> MaskedOlmoEarthSample:
    """``batch`` fully visible 16 x 16 windows of S2 L2A + S1 + Landsat."""
    ts = torch.tensor([[1, m % 12, 2020 + m // 12] for m in range(timesteps)]).long()
    kw = {"timestamps": ts[None].expand(batch, -1, -1).contiguous().cuda()}
    for mod in (Modality.SENTINEL2_L2A, Modality.SENTINEL1, Modality.LANDSAT):
        shape = (batch, WINDOW, WINDOW, timesteps, mod.num_bands)
        kw[mod.name] = torch.rand(shape, device="cuda")
        kw[f"{mod.name}_mask"] = torch.full(
            shape, MaskValue.ONLINE_ENCODER.value, dtype=torch.long, device="cuda"
        )
    return MaskedOlmoEarthSample(**kw)


def time_shape(enc: torch.nn.Module, ps: int, timesteps: int) -> dict:
    """Largest batch that fits, then windows/s over ITERS iterations."""
    for batch in BATCH_SIZES:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        x = None
        try:
            x = sample(batch, timesteps)

            def fwd(x: MaskedOlmoEarthSample = x) -> None:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    enc(x, patch_size=ps, fast_pass=True)

            t0 = time.time()
            for _ in range(WARMUP):
                fwd()
            torch.cuda.synchronize()
            warm_s = time.time() - t0
            start, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            start.record()
            for _ in range(ITERS):
                fwd()
            end.record()
            torch.cuda.synchronize()
            ms_per_iter = start.elapsed_time(end) / ITERS
            return {
                "batch": batch,
                "windows_per_s": batch / (ms_per_iter / 1000),
                "ms_per_window": ms_per_iter / batch,
                "peak_gib": torch.cuda.max_memory_allocated() / 2**30,
                "warmup_s": warm_s,
            }
        except torch.OutOfMemoryError:
            x = None  # free the batch before trying a smaller one
            continue
    return {"batch": 0, "error": "OOM at every batch size"}


def main() -> None:
    """Benchmark the models named on the command line (all by default)."""
    keys = sys.argv[1:] or ["rc", *MODELS]
    print(
        f"torch {torch.__version__}, {torch.cuda.get_device_name()}, flash SDPA "
        f"{torch.backends.cuda.flash_sdp_enabled()}",
        flush=True,
    )
    results: dict = {}
    for key in keys:
        enc = build_encoder(key)
        params = sum(p.numel() for p in enc.parameters()) / 1e6
        results[key] = {"params_M": params}
        for label, ps, t in SHAPES:
            r = time_shape(enc, ps, t)
            results[key][label] = r
            if r.get("batch"):
                print(
                    f"RESULT {key:18s} {label:9s} batch {r['batch']:4d} "
                    f"{r['windows_per_s']:9.1f} win/s {r['ms_per_window']:8.3f} ms/win "
                    f"peak {r['peak_gib']:5.1f} GiB (warm-up {r['warmup_s']:.0f} s)",
                    flush=True,
                )
            else:
                print(f"RESULT {key:18s} {label:9s} {r['error']}", flush=True)
        del enc
        torch.cuda.empty_cache()
    out = os.environ.get("RESULTS_DIR")
    if out:
        Path(out).mkdir(parents=True, exist_ok=True)
        with open(Path(out) / "bench_inference.json", "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
