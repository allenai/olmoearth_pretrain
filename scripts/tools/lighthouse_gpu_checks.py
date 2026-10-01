"""GPU gates for Lighthouse on a real checkpoint and a real AOI window.

1. Parity: on one 16x16 crop (the FOV is the whole domain), the FlexAttention
   Lighthouse path equals the stock forward, at patch sizes 1 and 4.
2. Flex vs dense: on a domain a few FOVs wide, the block-sparse FlexAttention path
   equals the dense-mask SDPA path (ps4 only: dense is quadratic).
3. Chunking: a chunk with the exact halo reproduces the full-domain core.

Prints max |diff| and min cosine per check; exits non-zero if a gate fails.
"""

import argparse
import sys

import torch

sys.path.insert(0, "scripts/tools")
from lighthouse_aoi_inference import (  # noqa: E402
    WindowSamples,
    _batched,
    _crop,
    build_window_dataset,
)

from olmoearth_pretrain.model_loader import load_pretrain_checkpoint  # noqa: E402
from olmoearth_pretrain.nn.lighthouse import (  # noqa: E402
    LighthouseSettings,
    lighthouse_reach_px,
)


def student(encoder, sample, ps, lh=None):
    """d128 student embeddings of one sample."""
    encoder.perceiver.lighthouse = lh
    try:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = encoder(sample, patch_size=ps, input_res=10, fast_pass=True)
    finally:
        encoder.perceiver.lighthouse = None
    return out["student_registers"][0, ..., :128].float()


def compare(name, a, b, max_diff, min_cos):
    """Print and return whether ``a`` matches ``b``."""
    d = (a - b).abs().max().item()
    cos = torch.nn.functional.cosine_similarity(a, b, dim=-1).min().item()
    ok = d <= max_diff and cos >= min_cos
    print(
        f"{'PASS' if ok else 'FAIL'} {name}: max|diff|={d:.3e} min cos={cos:.6f}",
        flush=True,
    )
    return ok


def main() -> None:
    """Run the gates."""
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--name", default="barbizon_apremont")
    a = p.parse_args()
    dev = torch.device("cuda")
    encoder = load_pretrain_checkpoint(a.checkpoint, device=dev).encoder
    depth = len(encoder.perceiver.joint_blocks)
    ds = WindowSamples(build_window_dataset(a.dataset, "predict", [a.name]))
    _, sample, _ = ds[0]
    full = _batched(sample, dev)
    ok = True

    # 1. One-window parity.
    for ps in (1, 4):
        win = _crop(full, slice(500, 516), slice(500, 516))
        ref = student(encoder, win, ps)
        lh = student(encoder, win, ps, LighthouseSettings(fov_px=16))
        # bf16 autocast: the two paths group work differently.
        ok &= compare(f"parity ps{ps}", lh, ref, 5e-2, 0.999)

    # 2. Flex vs dense on a 64 px domain at ps4.
    dom = _crop(full, slice(400, 464), slice(400, 464))
    flex = student(encoder, dom, 4, LighthouseSettings(fov_px=16))
    dense = student(encoder, dom, 4, LighthouseSettings(fov_px=16, dense=True))
    ok &= compare("flex vs dense ps4 64px", flex, dense, 5e-2, 0.999)

    # 3. Chunk with exact halo vs full domain.
    for ps, side, core in ((4, 448, 64), (1, 288, 32)):
        halo = lighthouse_reach_px(16, ps, depth, encoder.max_patch_size)
        base = 200
        dom = _crop(full, slice(base, base + side), slice(base, base + side))
        whole = student(encoder, dom, ps, LighthouseSettings(fov_px=16))
        c0 = side // 2 - core // 2
        lo, hi = c0 - halo, c0 + core + halo
        assert lo >= 0 and hi <= side, (lo, hi, side)
        chunk = student(
            encoder,
            _crop(dom, slice(lo, hi), slice(lo, hi)),
            ps,
            LighthouseSettings(fov_px=16, origin_px=(lo, lo)),
        )
        cc = slice(c0 - lo, c0 - lo + core)
        ok &= compare(
            f"chunk vs full ps{ps} halo {halo}",
            chunk[cc, cc],
            whole[c0 : c0 + core, c0 : c0 + core],
            5e-2,
            0.999,
        )
    print("ALL GATES PASSED" if ok else "SOME GATES FAILED", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
