"""Where RC Lighthouse time goes, and what the attention kernel can do at best.

On one real crop (default 240 px of sundarbans_tidal at ps1, the main run's chunk):

1. Model forwards: Lighthouse exact (``fov_quantum`` 1) and quantized (2, 4, 8) with
   per-phase profiles, and the stock tiled forward (16 px crops, no overlap,
   ``fast_pass``) on the same pixels, for a per-token reference.
2. Attention alone, at the crop's real token count (bf16, 12 heads, d64):
   * FlexAttention with the exact ViT plan (partial blocks through ``mask_mod``);
   * the same blocks with every block declared full (no mask: the kernel's cost
     for that sparsity pattern);
   * FlexAttention with quantized plans;
   * SDPA (flash) over the tiled windows: the dense kernel tiling uses.

Prints one JSON line per measurement and writes them to ``--out_json``.
"""

import argparse
import json
import sys
import time

import numpy as np
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
from olmoearth_pretrain.nn.joint_latent import flex_attention_cuda  # noqa: E402
from olmoearth_pretrain.nn.lighthouse_rc import (  # noqa: E402
    RCLighthouseSettings,
    _block_tables,
    _make_plan,
    _slot_layout,
)

RESULTS: list[dict] = []


def emit(**row: object) -> None:
    """Print and keep one measurement."""
    RESULTS.append(row)
    print(json.dumps(row), flush=True)


def bench(fn, reps: int = 3) -> float:
    """Median seconds of ``fn()`` after one warm-up call."""
    fn()
    times = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def forward(encoder, sample, settings=None, fast_pass=False):
    """Encoder outputs of one sample, under ``settings`` (stock when None)."""
    encoder.lighthouse = settings
    try:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            return encoder(sample, patch_size=1, input_res=10, fast_pass=fast_pass)
    finally:
        encoder.lighthouse = None


def main() -> None:
    """Run the measurements."""
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--name", default="sundarbans_tidal")
    p.add_argument("--side", type=int, default=240)
    p.add_argument("--out_json", default=None)
    a = p.parse_args()
    dev = torch.device("cuda")
    encoder = load_pretrain_checkpoint(a.checkpoint, device=dev).encoder.eval()
    ds = WindowSamples(build_window_dataset(a.dataset, "predict", [a.name]))
    _, sample, _ = ds[0]
    full = _batched(sample, dev)
    side = a.side
    crop_dict = {
        k: (v if k == "timestamps" else v[:, 400 : 400 + side, 400 : 400 + side])
        for k, v in full.items()
    }
    crop = _crop(crop_dict, slice(0, side), slice(0, side))
    km2 = side * side / 1e4

    # 1. Model forwards.
    ref = None
    for quantum, column in ((1, True), (1, False), (4, True), (8, True)):
        s = RCLighthouseSettings(fov_px=16, fov_quantum=quantum, column_mask=column)
        sec = bench(lambda: forward(encoder, crop, s), reps=2)
        prof = RCLighthouseSettings(
            16, fov_quantum=quantum, column_mask=column, profile=True
        )
        out = forward(encoder, crop, prof)
        stats = dict(encoder.last_lighthouse_stats)
        emb = out["student_registers"][0].float()
        if ref is None:
            ref = emb
        cos = torch.nn.functional.cosine_similarity(emb, ref, dim=-1)
        emit(
            what="lighthouse_forward",
            quantum=quantum,
            column_mask=column,
            nan_px=int(torch.isnan(emb).any(-1).sum()),
            seconds=sec,
            s_per_processed_km2=sec / km2,
            cos_vs_exact_mean=cos.mean().item(),
            cos_vs_exact_p01=cos.flatten().quantile(0.01).item(),
            **{k: round(v, 4) for k, v in stats.items()},
        )
    boxes = get_window_crop_options((16, 16), (0, 0), (0, 0, side, side))

    def tiled():
        for i in range(0, len(boxes), 64):
            forward(encoder, _stack_crops(crop_dict, boxes[i : i + 64]), fast_pass=True)

    sec = bench(tiled, reps=2)
    emit(what="tiled_ov0_fastpass", seconds=sec, s_per_processed_km2=sec / km2)

    def tiled_masked():
        for i in range(0, len(boxes), 64):
            forward(
                encoder, _stack_crops(crop_dict, boxes[i : i + 64]), fast_pass=False
            )

    sec = bench(tiled_masked, reps=1)
    emit(what="tiled_ov0_masked", seconds=sec, s_per_processed_km2=sec / km2)

    # 2. Attention alone at the real token count.
    n_tokens = int(encoder.last_lighthouse_stats["tokens"])
    per_cell = n_tokens // (side * side)
    cells = np.repeat(np.arange(side * side), per_cell)
    heads, d = 12, 64
    for quantum in (1, 2, 4, 8):
        tile = (1, side) if quantum == 1 else (quantum, quantum)
        lay = _slot_layout(cells // side, cells % side, side, tile, 128)
        lay.quantum = quantum
        settings = RCLighthouseSettings(fov_px=16, q_chunk=1 << 30)
        plan = _make_plan(lay, lay, 16, side, side, settings, dev)
        q = torch.randn(1, heads, lay.length, d, device=dev, dtype=torch.bfloat16)
        k, v = torch.randn_like(q), torch.randn_like(q)
        bm = plan.chunks[0][1]
        sec = bench(lambda: flex_attention_cuda(q, k, v, bm))
        emit(
            what="flex_attention",
            quantum=quantum,
            slots=lay.length,
            seconds=sec,
            **plan.stats,
        )
        if quantum == 1:
            from torch.nn.attention.flex_attention import BlockMask

            # Same sparsity, every block declared full: the kernel without mask_mod.
            tab = _block_tables(lay, lay, 16, side, side, 128)
            rows = [
                np.concatenate(
                    [
                        tab["part_idx"][i, : tab["part_num"][i]],
                        tab["full_idx"][i, : tab["full_num"][i]],
                    ]
                )
                for i in range(len(tab["part_num"]))
            ]
            num = np.array([r.size for r in rows], dtype=np.int32)
            idx = np.zeros((len(rows), int(num.max())), dtype=np.int32)
            for i, r in enumerate(rows):
                idx[i, : r.size] = np.sort(r)

            def t(x: np.ndarray) -> torch.Tensor:
                return torch.from_numpy(x).to(dev)[None, None]

            allfull = BlockMask.from_kv_blocks(
                t(np.zeros_like(num)),
                t(np.zeros_like(idx)),
                t(num),
                t(idx),
                BLOCK_SIZE=128,
                mask_mod=None,
                seq_lengths=(lay.length, lay.length),
                compute_q_blocks=False,
            )
            sec = bench(lambda: flex_attention_cuda(q, k, v, allfull))
            emit(what="flex_attention_same_blocks_no_mask", seconds=sec)
        torch.cuda.empty_cache()
    from torch.nn.attention import SDPBackend, sdpa_kernel

    n_win = (side // 16) ** 2
    qw = torch.randn(32, heads, 256 * per_cell, d, device=dev, dtype=torch.bfloat16)
    kw, vw = torch.randn_like(qw), torch.randn_like(qw)
    for backend in (SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION):
        try:
            with sdpa_kernel(backend):
                sec = bench(
                    lambda: torch.nn.functional.scaled_dot_product_attention(qw, kw, vw)
                )
            emit(
                what="sdpa_tiled_windows_ov0",
                backend=str(backend),
                windows=n_win,
                seconds=sec * n_win / 32,
            )
        except RuntimeError as e:
            emit(
                what="sdpa_tiled_windows_ov0", backend=str(backend), error=str(e)[:200]
            )
    if a.out_json:
        with open(a.out_json, "w") as f:
            json.dump(RESULTS, f, indent=1)


if __name__ == "__main__":
    main()
