"""NATTEN backend / tile-shape sweep for the Lighthouse ViT attention.

Shape: the Sundarbans chunk (240 x 240 cells, K = 36 tokens per cell, 12 heads,
d64, bf16), kernel (16, 16, 36) -- or (36, 16, 16) with the token axis first, since
the tile shapes that fit differ. Every config is checked against the default
backend's output (cos >= 0.9999) so a fast-but-wrong config cannot win.

Prints one JSON line per config.
"""

import inspect
import itertools
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


def main() -> None:
    """Sweep backends and tile shapes."""
    import natten

    na3d = natten.na3d
    emit(
        gpu=torch.cuda.get_device_name(),
        natten=natten.__version__,
        signature=str(inspect.signature(na3d)),
        config_helpers=[n for n in dir(natten) if "config" in n.lower()],
    )
    dev = torch.device("cuda")
    n, k, heads, d = 240, 36, 12, 64
    for layout in ("rck", "krc"):
        shape = (1, n, n, k, heads, d) if layout == "rck" else (1, k, n, n, heads, d)
        kernel = (16, 16, k) if layout == "rck" else (k, 16, 16)
        q, kk, v = (
            torch.randn(shape, device=dev, dtype=torch.bfloat16) for _ in range(3)
        )
        ref = na3d(q, kk, v, kernel_size=kernel)
        sec = bench(lambda: na3d(q, kk, v, kernel_size=kernel))
        emit(layout=layout, backend="default", seconds_per_layer=sec)
        for backend in ("hopper-fna", "cutlass-fna"):
            try:
                out = na3d(q, kk, v, kernel_size=kernel, backend=backend)
                cos = F.cosine_similarity(out.float(), ref.float(), dim=-1).min().item()
                sec = bench(lambda: na3d(q, kk, v, kernel_size=kernel, backend=backend))
                emit(layout=layout, backend=backend, seconds_per_layer=sec, cos_min=cos)
            except Exception as e:  # noqa: BLE001
                emit(layout=layout, backend=backend, error=repr(e)[:240])
                continue
            helper = {
                "cutlass-fna": "get_configs_for_cutlass_fna",
                "hopper-fna": "get_configs_for_cutlass_hopper_fna",
                "blackwell-fna": "get_configs_for_cutlass_blackwell_fna",
            }.get(backend)
            fn = getattr(natten, helper, None) if helper else None
            if fn is None:
                continue
            try:
                configs = list(fn(q, kk, v))
            except Exception as e:  # noqa: BLE001
                emit(layout=layout, backend=backend, configs_error=repr(e)[:240])
                continue
            emit(
                layout=layout,
                backend=backend,
                n_configs=len(configs),
                first=repr(configs[:2])[:300],
            )
            for cfg in itertools.islice(configs, 40):
                kwargs = {}
                if isinstance(cfg, dict):
                    kwargs = cfg
                elif isinstance(cfg, tuple) and len(cfg) >= 2:
                    kwargs = {"q_tile_shape": cfg[0], "kv_tile_shape": cfg[1]}
                    if len(cfg) > 2:
                        kwargs["kernel_schedule"] = cfg[2]
                try:
                    out = na3d(q, kk, v, kernel_size=kernel, backend=backend, **kwargs)
                    cos = (
                        F.cosine_similarity(out.float(), ref.float(), dim=-1)
                        .min()
                        .item()
                    )
                    sec = bench(
                        lambda kw=kwargs: na3d(
                            q, kk, v, kernel_size=kernel, backend=backend, **kw
                        ),
                        reps=3,
                    )
                    emit(
                        layout=layout,
                        backend=backend,
                        config=kwargs,
                        seconds_per_layer=sec,
                        cos_min=cos,
                        ok=cos >= 0.9999,
                    )
                except Exception as e:  # noqa: BLE001
                    emit(
                        layout=layout,
                        backend=backend,
                        config=kwargs,
                        error=repr(e)[:200],
                    )
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
