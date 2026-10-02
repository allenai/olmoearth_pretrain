"""Can NATTEN run exact Lighthouse attention for the ViT, and how fast?

Lighthouse's ViT rule on a complete token grid ``(rows, cols, K)`` (K = tokens per
cell) is 3D neighbourhood attention with kernel ``(W, W, K)``: all tokens of every
cell in the W x W box ``[clip(r - W//2, 0, n - W), + W)``, shifted inward at edges.

1. Parity: NATTEN ``na3d`` vs SDPA with that dense mask, on a small grid, for the
   even FOV we use (W = 16) and odd neighbours (15, 17), so we know which kernel
   reproduces our window exactly.
2. Speed at the Sundarbans chunk shape (240 x 240 cells, K = 36, 12 heads, d64,
   bf16): NATTEN vs cuDNN SDPA on the tiled windows (the dense lower bound).

Prints one JSON line per measurement.
"""

import json
import sys
import time

import torch
import torch.nn.functional as F


def emit(**row: object) -> None:
    """Print one result line."""
    print(json.dumps(row), flush=True)


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


def lighthouse_mask(n: int, k: int, fov: int, device: torch.device) -> torch.Tensor:
    """Dense [L, L] mask of the Lighthouse rule on an (n, n, k) row-major grid."""
    idx = torch.arange(n * n * k, device=device)
    cell = idx // k
    r, c = cell // n, cell % n
    r0 = (r - fov // 2).clamp(0, n - fov)
    c0 = (c - fov // 2).clamp(0, n - fov)
    return (
        (r[None] >= r0[:, None])
        & (r[None] < r0[:, None] + fov)
        & (c[None] >= c0[:, None])
        & (c[None] < c0[:, None] + fov)
    )


def main() -> None:
    """Run parity and speed checks."""
    dev = torch.device("cuda")
    emit(gpu=torch.cuda.get_device_name(), torch=torch.__version__)
    try:
        import natten
    except Exception as e:  # noqa: BLE001
        emit(natten_import="FAILED", error=repr(e)[:300])
        sys.exit(1)
    na3d = getattr(natten, "na3d", None)
    if na3d is None:
        from natten.functional import na3d  # type: ignore[no-redef]
    emit(natten=getattr(natten, "__version__", "?"))

    # 1. Parity on a small grid (fp32 reference, bf16 kernel).
    n, k, heads, d = 24, 6, 4, 64
    g = torch.Generator(device=dev).manual_seed(0)
    q, kk, v = (
        torch.randn(1, n, n, k, heads, d, device=dev, generator=g) for _ in range(3)
    )
    flat = [t.reshape(1, n * n * k, heads, d).transpose(1, 2) for t in (q, kk, v)]
    for fov in (16, 15, 17):
        ref = F.scaled_dot_product_attention(
            *[t.double() for t in flat], attn_mask=lighthouse_mask(n, k, 16, dev)
        )
        ref = ref.transpose(1, 2).reshape(1, n, n, k, heads, d).float()
        try:
            out = na3d(
                q.bfloat16(), kk.bfloat16(), v.bfloat16(), kernel_size=(fov, fov, k)
            ).float()
            cos = F.cosine_similarity(out, ref, dim=-1)
            emit(
                check="parity_vs_lighthouse16",
                kernel=fov,
                cos_min=cos.min().item(),
                cos_mean=cos.mean().item(),
                max_abs=(out - ref).abs().max().item(),
            )
        except Exception as e:  # noqa: BLE001
            emit(check="parity_vs_lighthouse16", kernel=fov, error=repr(e)[:300])

    # 2. Speed at the chunk shape.
    n, k, heads, d = 240, 36, 12, 64
    q, kk, v = (
        torch.randn(1, n, n, k, heads, d, device=dev, dtype=torch.bfloat16)
        for _ in range(3)
    )
    for fov in (16, 17):
        try:
            sec = bench(lambda: na3d(q, kk, v, kernel_size=(fov, fov, k)))
            emit(check="speed", kernel=fov, tokens=n * n * k, seconds_per_layer=sec)
        except Exception as e:  # noqa: BLE001
            emit(check="speed", kernel=fov, error=repr(e)[:300])
    from torch.nn.attention import SDPBackend, sdpa_kernel

    win = torch.randn(32, heads, 256 * k, d, device=dev, dtype=torch.bfloat16)
    n_win = (n // 16) ** 2
    for backend in (SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION):
        try:
            with sdpa_kernel(backend):
                sec = bench(lambda: F.scaled_dot_product_attention(win, win, win))
            emit(
                check="speed_tiled_windows",
                backend=str(backend),
                seconds_per_layer=sec * n_win / 32,
            )
        except Exception as e:  # noqa: BLE001
            emit(check="speed_tiled_windows", backend=str(backend), error=repr(e)[:200])


if __name__ == "__main__":
    main()
