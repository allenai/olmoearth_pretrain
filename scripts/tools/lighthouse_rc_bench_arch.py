"""Tiled vs Lighthouse speed for an architecture, randomly initialized.

Builds the model from a run's ``config.json`` WITHOUT its weights (speed does not
depend on them), then on one real crop of an AOI window (default sundarbans_tidal,
S2 + S1 + Landsat, 12 months) and for each token patch size:

* tiled: 16 px crops at overlap 4, batch 64, the rslearn convention, with
  ``fast_pass`` (masks ignored) and masked (fixed eval path); s/km2 of output;
* Lighthouse exact (``nn/lighthouse_rc.py``, FlexAttention): one chunk of
  ``side + 2 * halo`` px; s/km2 of the ``side`` px core, with the per-phase split
  (ViT / Perceiver read / latent self-attention);
* NATTEN projection: the ViT attention (kernel (W, W, K)) and the latent
  self-attention (kernel (W, W, (ps/stride)^2)) timed with ``natten.na3d`` on the
  same shapes, swapped into the Lighthouse total in place of the flex times. Reads
  stay on flex (latent and token grids differ). Parity of the NATTEN kernels with
  the Lighthouse rule is checked on a small grid first.

Prints one JSON line per measurement; ``--out_json`` collects them.
"""

import argparse
import json
import sys
import time

import torch
import torch.nn.functional as F
from upath import UPath

sys.path.insert(0, "scripts/tools")
from lighthouse_aoi_inference import (  # noqa: E402
    WindowSamples,
    _batched,
    _crop,
    _stack_crops,
    build_window_dataset,
)
from rslearn.train.all_crops_dataset import get_window_crop_options  # noqa: E402

from olmoearth_pretrain.model_loader import _load_model_from_config  # noqa: E402
from olmoearth_pretrain.nn.lighthouse_rc import (  # noqa: E402
    RCLighthouseSettings,
    lighthouse_rc_reach_px,
)

ROWS: list[dict] = []


def emit(**row: object) -> None:
    """Print and keep one measurement."""
    ROWS.append(row)
    print(json.dumps(row, default=str), flush=True)


def timed(fn, reps: int = 2) -> tuple[object, float, float]:
    """(last result, median seconds, peak GiB) after one warm-up call."""
    out = fn()
    times = []
    torch.cuda.reset_peak_memory_stats()
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return (
        out,
        sorted(times)[len(times) // 2],
        torch.cuda.max_memory_allocated() / 2**30,
    )


def forward(encoder, sample, ps, settings=None, fast_pass=False):
    """Encoder output dict (stock forward when ``settings`` is None)."""
    encoder.lighthouse = settings
    try:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            return encoder(sample, patch_size=ps, input_res=10, fast_pass=fast_pass)
    finally:
        encoder.lighthouse = None


def natten_rule_mask(n: int, k: int, fov: int, dev: torch.device) -> torch.Tensor:
    """Dense Lighthouse mask on an (n, n, k) grid (all k tokens of FOV cells)."""
    idx = torch.arange(n * n * k, device=dev)
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


def natten_parity(na3d, fov: int, k: int, dev: torch.device) -> float:
    """Min cosine of na3d (kernel (fov, fov, k)) vs the dense Lighthouse rule."""
    n, heads, d = max(fov + 8, 12), 4, 64
    q, kk, v = (torch.randn(1, n, n, k, heads, d, device=dev) for _ in range(3))
    flat = [
        t.reshape(1, n * n * k, heads, d).transpose(1, 2).double() for t in (q, kk, v)
    ]
    ref = F.scaled_dot_product_attention(
        *flat, attn_mask=natten_rule_mask(n, k, fov, dev)
    )
    ref = ref.transpose(1, 2).reshape(1, n, n, k, heads, d).float()
    out = na3d(q.bfloat16(), kk.bfloat16(), v.bfloat16(), kernel_size=(fov, fov, k))
    return F.cosine_similarity(out.float(), ref, dim=-1).min().item()


def natten_time(na3d, n: int, k: int, fov: int, heads: int, d: int, dev) -> float:
    """Seconds per call of na3d on an (n, n, k) grid."""
    q, kk, v = (
        torch.randn(1, n, n, k, heads, d, device=dev, dtype=torch.bfloat16)
        for _ in range(3)
    )
    _, sec, _ = timed(lambda: na3d(q, kk, v, kernel_size=(fov, fov, k)), reps=5)
    return sec


def main() -> None:
    """Benchmark one architecture."""
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="a checkpoint's config.json")
    p.add_argument("--label", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--name", default="sundarbans_tidal")
    p.add_argument("--pss", type=int, nargs="+", default=[2, 4, 1])
    p.add_argument("--side", type=int, nargs="+", default=[448, 768, 224])
    p.add_argument("--halo", type=int, default=16)
    p.add_argument("--out_json", default=None)
    a = p.parse_args()
    dev = torch.device("cuda")
    torch.manual_seed(0)
    model = _load_model_from_config(UPath(a.config))
    encoder = model.encoder.to(dev).eval()
    perceiver = encoder.perceiver
    emit(
        label=a.label,
        gpu=torch.cuda.get_device_name(),
        vit_depth=len(encoder.blocks),
        perceiver_depth=len(perceiver.latent_blocks),
        pixel_latents=perceiver.pixel_latents,
        eval_latent_stride=perceiver.eval_latent_stride,
        weights="random init",
    )
    try:
        import natten

        na3d = natten.na3d
        emit(natten=natten.__version__)
    except Exception as e:  # noqa: BLE001
        na3d = None
        emit(natten="unavailable", error=repr(e)[:200])

    ds = WindowSamples(build_window_dataset(a.dataset, "predict", [a.name]))
    _, sample, _ = ds[0]
    full = _batched(sample, dev)
    heads = encoder.blocks[0].attn.num_heads
    d = encoder.blocks[0].attn.head_dim
    for ps, side in zip(a.pss, a.side):
        halo = a.halo
        chunk = side + 2 * halo
        crop_dict = {
            k: (v if k == "timestamps" else v[:, 64 : 64 + chunk, 64 : 64 + chunk])
            for k, v in full.items()
        }
        core_dict = {
            k: (
                v if k == "timestamps" else v[:, halo : halo + side, halo : halo + side]
            )
            for k, v in crop_dict.items()
        }
        km2 = side * side / 1e4
        row = {"label": a.label, "ps": ps, "core_px": side, "halo_px": halo}

        # Tiled on the core (16 px / overlap 4), fast_pass and masked.
        boxes = get_window_crop_options((16, 16), (4, 4), (0, 0, side, side))
        for fast in (True, False):

            def tiled(fast=fast):
                for i in range(0, len(boxes), 64):
                    forward(
                        encoder,
                        _stack_crops(core_dict, boxes[i : i + 64]),
                        ps,
                        fast_pass=fast,
                    )

            _, sec, _ = timed(tiled, reps=1)
            row[f"tiled_{'fastpass' if fast else 'masked'}_s_per_km2"] = sec / km2

        # Lighthouse exact on the chunk; per-phase profile from a separate call.
        crop = _crop(crop_dict, slice(0, chunk), slice(0, chunk))
        settings = RCLighthouseSettings(fov_px=16)
        try:
            _, sec, peak = timed(lambda: forward(encoder, crop, ps, settings))
            forward(encoder, crop, ps, RCLighthouseSettings(16, profile=True))
            stats = dict(encoder.last_lighthouse_stats)
        except torch.OutOfMemoryError:
            emit(**row, lighthouse="OOM", chunk_px=chunk)
            torch.cuda.empty_cache()
            continue
        row.update(
            lh_flex_s_per_km2=sec / km2,
            lh_peak_gib=peak,
            exact_reach_px=lighthouse_rc_reach_px(
                16, ps, len(encoder.blocks), len(perceiver.latent_blocks), 8
            ),
            **{f"prof_{k}": round(v, 4) for k, v in stats.items()},
        )

        # NATTEN projection for the ViT + latent self-attention.
        if na3d is not None:
            fov = 16 // ps
            k_tokens = int(round(stats["tokens"] / (chunk // ps) ** 2))
            stride = perceiver.eval_latent_stride if perceiver.pixel_latents else ps
            k_lat = (ps // stride) ** 2
            try:
                row["natten_parity_vit"] = natten_parity(
                    na3d, fov, min(k_tokens, 8), dev
                )
                row["natten_parity_lat"] = natten_parity(na3d, fov, k_lat, dev)
                vit_layer = natten_time(na3d, chunk // ps, k_tokens, fov, heads, d, dev)
                lat_layer = natten_time(na3d, chunk // ps, k_lat, fov, heads, d, dev)
                n_vit, n_lat = len(encoder.blocks), len(perceiver.latent_blocks)
                flex_vit = stats.get("vit_attention_s", 0.0)
                flex_lat = stats.get("latent_attention_s", 0.0)
                profiled = sum(
                    v
                    for k, v in stats.items()
                    if k.endswith("_s") and "plan" not in k and k != "layout_s"
                )
                # Scale the profiled phase split to the unprofiled wall time.
                scale = sec / max(profiled + stats.get("layout_s", 0.0), 1e-9)
                proj = (
                    sec
                    - scale * (flex_vit + flex_lat)
                    + n_vit * vit_layer
                    + n_lat * lat_layer
                )
                row.update(
                    k_tokens_per_cell=k_tokens,
                    natten_vit_s_per_layer=vit_layer,
                    natten_latent_s_per_layer=lat_layer,
                    lh_natten_projected_s_per_km2=proj / km2,
                )
            except Exception as e:  # noqa: BLE001
                row["natten_error"] = repr(e)[:200]
        emit(**row)
        torch.cuda.empty_cache()
    if a.out_json:
        with open(a.out_json, "w") as f:
            json.dump(ROWS, f, indent=1)


if __name__ == "__main__":
    main()
