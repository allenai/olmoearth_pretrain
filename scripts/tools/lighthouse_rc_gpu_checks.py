"""GPU gates + a chunk-size scan for RC Lighthouse (``nn/lighthouse_rc.py``).

On a real checkpoint (v1.3 RC or an rc_pix512 arm) and a real AOI window:

1. Parity: on one 16x16 crop (the FOV is the whole domain), the FlexAttention
   Lighthouse path equals the stock masked forward (batch 1, ``fast_pass=False``).
2. Flex vs dense: on a domain a few FOVs wide, the block-sparse path equals the
   dense-mask SDPA path.
3. Chunking: a chunk with the exact halo reproduces the full-domain core (at ps2,
   where the exact 144 px halo fits in memory).
4. Scan: seconds, s/km2 of the chunk and peak memory per chunk side at ps1, with the
   per-phase profile and block statistics, plus the tiled forward on the same crop.

Prints one line per check; exits non-zero if a gate fails.
"""

import argparse
import json
import sys
import time

import torch

sys.path.insert(0, "scripts/tools")
from lighthouse_aoi_inference import (  # noqa: E402
    WindowSamples,
    _batched,
    _crop,
    _stack_crops,
    build_window_dataset,
)
from rslearn.train.all_crops_dataset import get_window_crop_options  # noqa: E402

from olmoearth_pretrain.model_loader import load_pretrain_checkpoint  # noqa: E402
from olmoearth_pretrain.nn.lighthouse_rc import (  # noqa: E402
    RCLighthouseSettings,
    lighthouse_rc_reach_px,
)


def student(encoder, sample, ps, lh=None, fast_pass=False):
    """d128 student embeddings of one sample (stock forward when ``lh`` is None)."""
    encoder.lighthouse = lh
    try:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = encoder(sample, patch_size=ps, input_res=10, fast_pass=fast_pass)
    finally:
        encoder.lighthouse = None
    return out["student_registers"][..., :128].float()


def compare(name, a, b, min_cos):
    """Print and return whether ``a`` matches ``b`` (cosine per embedding)."""
    d = (a - b).abs().max().item()
    cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)
    ok = cos.min().item() >= min_cos
    print(
        f"{'PASS' if ok else 'FAIL'} {name}: max|diff|={d:.3e} "
        f"cos min={cos.min().item():.6f} mean={cos.mean().item():.6f}",
        flush=True,
    )
    return ok


def timed(fn):
    """(result, seconds, peak GiB) of ``fn()``."""
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, time.perf_counter() - t0, torch.cuda.max_memory_allocated() / 2**30


def main() -> None:
    """Run the gates and the scan."""
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--name", default="sundarbans_tidal")
    p.add_argument("--scan_sides", type=int, nargs="*", default=[128, 192, 256, 320])
    p.add_argument("--out_json", default=None)
    a = p.parse_args()
    dev = torch.device("cuda")
    encoder = load_pretrain_checkpoint(a.checkpoint, device=dev).encoder.eval()
    ds = WindowSamples(build_window_dataset(a.dataset, "predict", [a.name]))
    _, sample, _ = ds[0]
    full = _batched(sample, dev)
    ok = True
    report: dict = {"gpu": torch.cuda.get_device_name(dev), "scan": []}
    # Patch sizes whose student output is checked; the RC only ships ps1.
    pss = (1, 2, 4) if encoder.perceiver.pixel_latents else (1, 2)

    # 1. One-window parity.
    for r0 in (500, 1200):
        win = _crop(full, slice(r0, r0 + 16), slice(r0, r0 + 16))
        for ps in pss:
            ref = student(encoder, win, ps)
            lh = student(encoder, win, ps, RCLighthouseSettings(fov_px=16))
            ok &= compare(f"parity ps{ps} @{r0}", lh, ref, 0.999)
        ref_fp = student(encoder, win, 1, fast_pass=True)
        ref1 = student(encoder, win, 1)
        compare(f"(info) stock fast_pass vs masked ps1 @{r0}", ref_fp, ref1, 0.0)

    # 2. Flex vs dense (dense mask is quadratic: 24 px at ps1, 48 px at ps2).
    for ps, side in ((1, 24), (2, 48)):
        dom = _crop(full, slice(300, 300 + side), slice(300, 300 + side))
        flex = student(encoder, dom, ps, RCLighthouseSettings(fov_px=16))
        dense = student(encoder, dom, ps, RCLighthouseSettings(fov_px=16, dense=True))
        ok &= compare(f"flex vs dense ps{ps} {side}px", flex, dense, 0.999)

    # 3. Chunk with the exact halo vs the full domain (ps2).
    ps, core = 2, 32
    halo = lighthouse_rc_reach_px(
        16, ps, len(encoder.blocks), len(encoder.perceiver.latent_blocks), 8
    )
    side = core + 2 * halo + 64
    base = 200
    dom = {
        k: (v if k == "timestamps" else v[:, base : base + side, base : base + side])
        for k, v in full.items()
    }
    unit = 1 if encoder.perceiver.pixel_latents else ps
    whole = student(
        encoder,
        _crop(dom, slice(0, side), slice(0, side)),
        ps,
        RCLighthouseSettings(16),
    )[0]
    c0 = (side // 2 - core // 2) // ps * ps
    for h in (halo, 64, 32, 16, 8, 0):
        lo, hi = c0 - h, c0 + core + h
        chunk = student(
            encoder,
            _crop(dom, slice(lo, hi), slice(lo, hi)),
            ps,
            RCLighthouseSettings(16, origin_px=(lo, lo)),
        )[0]
        cc = slice((c0 - lo) // unit, (c0 - lo + core) // unit)
        wc = slice(c0 // unit, (c0 + core) // unit)
        res = compare(
            f"chunk vs full ps{ps} halo {h}{' (exact)' if h == halo else ''}",
            chunk[cc, cc],
            whole[wc, wc],
            0.999 if h == halo else 0.0,
        )
        if h == halo:
            ok &= res

    # 4. Chunk-size scan at ps1 (memory + speed), Lighthouse vs tiled on the same crop.
    for side in a.scan_sides:
        crop_dict = {
            k: (v if k == "timestamps" else v[:, :side, :side]) for k, v in full.items()
        }
        crop = _crop(crop_dict, slice(0, side), slice(0, side))
        try:
            settings = RCLighthouseSettings(fov_px=16, profile=True)
            student(encoder, crop, 1, settings)  # warm-up / compile
            settings = RCLighthouseSettings(fov_px=16)
            _, sec, peak = timed(lambda: student(encoder, crop, 1, settings))
            prof = RCLighthouseSettings(fov_px=16, profile=True)
            student(encoder, crop, 1, prof)
            stats = dict(encoder.last_lighthouse_stats)
        except torch.OutOfMemoryError:
            print(f"scan {side}px: OOM", flush=True)
            torch.cuda.empty_cache()
            report["scan"].append({"side_px": side, "oom": True})
            break
        boxes = get_window_crop_options((16, 16), (0, 0), (0, 0, side, side))
        tiled_s = 0.0
        for i in range(0, len(boxes), 64):
            batch = _stack_crops(crop_dict, boxes[i : i + 64])
            _, s, _ = timed(lambda: student(encoder, batch, 1, fast_pass=True))
            tiled_s += s
        km2 = side * side / 1e4
        row = {
            "side_px": side,
            "lh_s": sec,
            "lh_s_per_km2": sec / km2,
            "tiled_ov0_fastpass_s_per_km2": tiled_s / km2,
            "peak_gib": peak,
            **{k: round(v, 4) for k, v in stats.items()},
        }
        report["scan"].append(row)
        print(f"scan {json.dumps(row)}", flush=True)
        torch.cuda.empty_cache()

    print("ALL GATES PASSED" if ok else "SOME GATES FAILED", flush=True)
    if a.out_json:
        with open(a.out_json, "w") as f:
            json.dump(report, f, indent=1)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
