"""Is the RC Lighthouse FlexAttention plan computed correctly on this GPU?

Pure attention, no model: random bf16 Q/K/V on a real token layout (``side`` x
``side`` cells, ``per_cell`` tokens each, some dropped), the plan
``nn/lighthouse_rc.py`` builds, and the dense SDPA reference with the same rule as
a boolean mask. Each variant runs in its own subprocess (a CUDA illegal access
poisons the context):

  compiled flex (dynamic=True, as the model runs it) | compiled dynamic=False |
  eager flex  x  packed int64 mask | column int64 mask | one chunk | many chunks

Usage: python scripts/tools/lighthouse_rc_flex_check.py [side] [per_cell]
"""

import json
import subprocess  # nosec
import sys

import numpy as np
import torch
import torch.nn.functional as F


def run_variant(name: str, side: int, per_cell: int) -> dict:
    """One variant in this process; returns max |diff| and min cosine vs dense."""
    from torch.nn.attention.flex_attention import flex_attention

    from olmoearth_pretrain.nn.lighthouse_rc import (
        RCLighthouseSettings,
        _codes,
        _make_plan,
        _rule,
        _slot_layout,
    )

    compile_mode, mask, chunks = name.split("/")
    dev = torch.device("cuda")
    rng = np.random.default_rng(0)
    cells = np.repeat(np.arange(side * side), per_cell)
    cells = cells[rng.random(cells.size) > 0.1]
    lay = _slot_layout(cells // side, cells % side, side, (1, side), 128)
    settings = RCLighthouseSettings(
        fov_px=16,
        column_mask=(mask == "column"),
        q_chunk=(1 << 30) if chunks == "one" else 128 * 7,
    )
    plan = _make_plan(lay, lay, 16, side, side, settings, dev)
    g = torch.Generator(device=dev).manual_seed(0)
    shape = (1, 12, lay.length, 64)
    q, k, v = (
        torch.randn(shape, device=dev, dtype=torch.bfloat16, generator=g)
        for _ in range(3)
    )
    if compile_mode == "dyn":
        fn = torch.compile(flex_attention, dynamic=True)
    elif compile_mode == "static":
        fn = torch.compile(flex_attention, dynamic=False)
    else:
        fn = flex_attention
    out = torch.empty_like(q)
    for qs, bm in plan.chunks:
        out[:, :, qs] = fn(q[:, :, qs].contiguous(), k, v, block_mask=bm)
    torch.cuda.synchronize()
    qc, kc = _codes(lay, lay, 16, side, side, dev)
    dense_mask = _rule(16, qc[:, None], kc[None, :])
    ref = F.scaled_dot_product_attention(q, k, v, attn_mask=dense_mask[None, None])
    valid = torch.from_numpy(lay.valid).to(dev)
    a, b = out[0, :, valid].float(), ref[0, :, valid].float()
    cos = F.cosine_similarity(a, b, dim=-1)
    return {
        "variant": name,
        "slots": lay.length,
        "chunks": len(plan.chunks),
        "max_abs_diff": (a - b).abs().max().item(),
        "cos_min": cos.min().item(),
        "nan": int(torch.isnan(a).any(-1).sum()),
        **plan.stats,
    }


def main() -> None:
    """Spawn every variant; print one JSON line each."""
    if len(sys.argv) > 1 and sys.argv[1] == "--variant":
        print(json.dumps(run_variant(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))))
        return
    side = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    per_cell = int(sys.argv[2]) if len(sys.argv) > 2 else 36
    print(torch.cuda.get_device_name(), torch.__version__, flush=True)
    for name in (
        "eager/packed/one",
        "dyn/packed/one",
        "static/packed/one",
        "dyn/packed/many",
        "dyn/column/one",
        "dyn/column/many",
        "eager/column/one",
    ):
        r = subprocess.run(  # nosec
            [sys.executable, __file__, "--variant", name, str(side), str(per_cell)],
            capture_output=True,
            text=True,
        )
        line = [x for x in r.stdout.splitlines() if x.startswith("{")]
        if line:
            print(line[-1], flush=True)
        else:
            err = [x for x in r.stderr.splitlines() if "Error" in x][-3:]
            print(json.dumps({"variant": name, "failed": err}), flush=True)


if __name__ == "__main__":
    main()
