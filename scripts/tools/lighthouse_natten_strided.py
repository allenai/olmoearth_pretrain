"""Lighthouse as STRIDED 2D neighbourhood attention in NATTEN.

Tokens on a 2D grid ``(rows, cols * K)`` (each cell's K tokens adjacent along the
column axis); ``na2d`` with kernel ``(W, W * K)`` and stride ``(1, K)`` makes the K
tokens of a cell one query group sharing one window. If NATTEN places that window
at whole cells, this is exactly the Lighthouse rule, with a 2D kernel that may tile
better than ``na3d``'s ``(W, W, K)``.

1. Parity vs our rule; if it fails, scan the column offset of the window (in
   tokens) to report what NATTEN computes instead.
2. Speed at ps1 / ps2 / ps4 chunk shapes (K = 36, 12 heads, d64, bf16): strided
   na2d vs na3d vs flash on the equivalent tiled windows.
"""

import json
import time

import torch
import torch.nn.functional as F


def emit(**row: object) -> None:
    """Print one result line."""
    print(json.dumps(row, default=str), flush=True)


def bench(fn, reps: int = 5) -> float:
    """Median seconds after one warm-up."""
    fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return sorted(ts)[len(ts) // 2]


def rule_mask(n: int, k: int, fov: int, col_offset: int, dev) -> torch.Tensor:
    """[L, L] mask on the (n, n*k) layout: rows = cell box; cols = token window.

    ``col_offset`` 0 is the Lighthouse rule (window starts at a cell boundary,
    ``k * clip(c - fov//2, 0, n - fov)``); other values shift the token window.
    """
    idx = torch.arange(n * n * k, device=dev)
    row = idx // (n * k)
    tcol = idx % (n * k)
    cell_col = tcol // k
    r0 = (row - fov // 2).clamp(0, n - fov)
    c0 = (k * (cell_col - fov // 2) + col_offset).clamp(0, n * k - fov * k)
    return (
        (row[None] >= r0[:, None])
        & (row[None] < r0[:, None] + fov)
        & (tcol[None] >= c0[:, None])
        & (tcol[None] < c0[:, None] + fov * k)
    )


def main() -> None:
    """Parity, then speed."""
    import natten

    dev = torch.device("cuda")
    emit(gpu=torch.cuda.get_device_name(), natten=natten.__version__)
    na2d, na3d = natten.na2d, natten.na3d

    # 1. Parity on a small grid.
    for fov, k in ((4, 6), (8, 6), (4, 36)):
        n, heads, d = fov + 6, 2, 64
        q, kk, v = (torch.randn(1, n, n * k, heads, d, device=dev) for _ in range(3))
        flat = [t.reshape(1, -1, heads, d).transpose(1, 2).double() for t in (q, kk, v)]
        try:
            out = na2d(
                q.bfloat16(),
                kk.bfloat16(),
                v.bfloat16(),
                kernel_size=(fov, fov * k),
                stride=(1, k),
            ).float()
        except Exception as e:  # noqa: BLE001
            emit(check="parity", fov=fov, k=k, error=repr(e)[:300])
            continue
        best = None
        for off in range(-k, k + 1):
            ref = F.scaled_dot_product_attention(
                *flat, attn_mask=rule_mask(n, k, fov, off, dev)
            )
            ref = ref.transpose(1, 2).reshape(1, n, n * k, heads, d).float()
            cos = F.cosine_similarity(out, ref, dim=-1).min().item()
            if off == 0:
                emit(check="parity_lighthouse_rule", fov=fov, k=k, cos_min=cos)
            if best is None or cos > best[1]:
                best = (off, cos)
        emit(
            check="best_matching_offset", fov=fov, k=k, offset=best[0], cos_min=best[1]
        )

    # 2. Speed at the chunk shapes.
    k, heads, d = 36, 12, 64
    for ps, n in ((1, 240), (2, 240), (4, 200)):
        fov = 16 // ps
        q, kk, v = (
            torch.randn(1, n, n * k, heads, d, device=dev, dtype=torch.bfloat16)
            for _ in range(3)
        )
        row = {"ps": ps, "cells": n, "tokens": n * n * k}
        try:
            row["na2d_strided_s"] = bench(
                lambda: na2d(q, kk, v, kernel_size=(fov, fov * k), stride=(1, k))
            )
        except Exception as e:  # noqa: BLE001
            row["na2d_strided_error"] = repr(e)[:200]
        q3, k3, v3 = (t.view(1, n, n, k, heads, d) for t in (q, kk, v))
        row["na3d_s"] = bench(lambda: na3d(q3, k3, v3, kernel_size=(fov, fov, k)))
        win = fov * fov * k
        n_win = (n // fov) ** 2
        qw = torch.randn(
            min(n_win, 64), heads, win, d, device=dev, dtype=torch.bfloat16
        )
        row["flash_tiled_ov0_s"] = (
            bench(lambda: F.scaled_dot_product_attention(qw, qw, qw))
            * n_win
            / qw.shape[0]
        )
        emit(check="speed", **row)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
