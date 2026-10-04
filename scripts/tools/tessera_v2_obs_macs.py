"""Measured Tessera v2 inference cost on our eval windows.

Tessera v2 is a per-pixel transformer: each pixel's valid S2 and S1 observations are
padded to the next multiple of 8 (capped at 256) and run through two 4-layer
backbones. Its compute therefore depends on how many valid observations each pixel
has. This script samples fetch windows from each dataset we ran Tessera v2 inference
on, reads exactly the inputs inference saw (``build_dpixel_inputs``), counts valid
observations per pixel the way ``tessera_v2_infer`` does, and charges each pixel the
MACs of the real checkpoint at its padded (S2 bin, S1 bin) lengths, counted with
torch's FLOP counter.

Output: per dataset, mean valid observations, mean padded lengths and mean MACs per
pixel, scaled to a 16x16 window (256 pixels) for comparison with OlmoEarth.
"""

import argparse
import json
import random

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode
from upath import UPath

from olmoearth_pretrain.evals.datasets.tessera_v2_export import (
    DATASETS,
    build_dpixel_inputs,
    resolve_spec,
)
from olmoearth_pretrain.evals.embedding_materializer.providers import (
    RslearnWindowProvider,
)
from olmoearth_pretrain.evals.models.tessera.tessera_v2_infer import _vec_get_bin_size
from olmoearth_pretrain.evals.models.tessera.tessera_v2_model import load_model

STAGE_ROOT = "/weka/dfive-default/rslearn-eai/datasets/olmoearth_evals"
CANDIDATE_PATHS = {
    "pastis_year_aligned": [
        f"{STAGE_ROOT}/pastis_year_aligned",
        "/weka/dfive-default/rslearn-eai/datasets/pastis_rslearn",
    ],
}
CKPT = "/weka/dfive-default/helios/models/tessera_v2/ckpt/student_large.pt"


def pixel_macs(model, s2_len: int, s1_len: int, cache: dict) -> float:
    """MACs to encode one pixel with padded sequence lengths (s2_len, s1_len)."""
    key = (s2_len, s1_len)
    if key not in cache:
        s2 = torch.zeros(1, max(s2_len, 1), 11)
        s1 = torch.zeros(1, max(s1_len, 1), 3)
        with torch.no_grad(), FlopCounterMode(display=False) as fc:
            model.encode(s2, s1)
        cache[key] = fc.get_total_flops() / 2
    return cache[key]


def main() -> None:
    """Sample windows per dataset and print measured Tessera v2 MACs."""
    p = argparse.ArgumentParser()
    p.add_argument("--windows_per_dataset", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.backends.mha.set_fastpath_enabled(False)  # so the counter sees attention
    model = load_model(CKPT, device=torch.device("cpu"))
    print("tessera params", sum(x.numel() for x in model.parameters()), flush=True)
    cache: dict = {}
    results = {}
    for name in sorted(DATASETS):
        if name == "pastis_rslearn":
            continue
        spec = resolve_spec(name)
        paths = CANDIDATE_PATHS.get(name, [f"{STAGE_ROOT}/{name}"])
        windows = None
        for path in paths:
            try:
                provider = RslearnWindowProvider(UPath(path), groups=[spec.fetch_group])
                windows = provider.load_windows()
                if windows:
                    break
            except Exception as e:  # noqa: BLE001
                print(f"{name}: {path} unusable ({e})", flush=True)
        if not windows:
            print(f"{name}: no fetch windows found", flush=True)
            continue
        rng = random.Random(args.seed)
        sample = rng.sample(windows, min(args.windows_per_dataset, len(windows)))
        n2_all, n1_all, b2_all, b1_all, macs_all = [], [], [], [], []
        failed = 0
        for w in sample:
            try:
                x = build_dpixel_inputs(w, allow_unmaterialized_s1=True)
            except Exception as e:  # noqa: BLE001
                failed += 1
                print(f"{name}/{w.name}: {type(e).__name__}: {e}", flush=True)
                continue
            n2 = x["s2_masks"].astype(bool).sum(axis=0).reshape(-1)  # (H*W,)
            s1_valid = []
            for k in ("s1_asc_bands", "s1_desc_bands"):
                b = x[k]
                if b is not None and b.size:
                    s1_valid.append(np.any(b != 0, axis=-1))  # (T, H, W)
            n1 = (
                np.concatenate(s1_valid, axis=0).sum(axis=0).reshape(-1)
                if s1_valid
                else np.zeros_like(n2)
            )
            b2 = _vec_get_bin_size(n2)
            b1 = _vec_get_bin_size(n1)
            m = np.array(
                [
                    0.0
                    if (i == 0 and j == 0)
                    else pixel_macs(model, int(i), int(j), cache)
                    for i, j in zip(b2, b1)
                ]
            )
            n2_all.append(n2), n1_all.append(n1), b2_all.append(b2), b1_all.append(b1)
            macs_all.append(m)
        if not macs_all:
            continue
        n2, n1 = np.concatenate(n2_all), np.concatenate(n1_all)
        b2, b1, m = (
            np.concatenate(b2_all),
            np.concatenate(b1_all),
            np.concatenate(macs_all),
        )
        r = {
            "windows": len(macs_all),
            "failed": failed,
            "pixels": int(m.size),
            "n_s2_mean": float(n2.mean()),
            "n_s1_mean": float(n1.mean()),
            "bin_s2_mean": float(b2.mean()),
            "bin_s1_mean": float(b1.mean()),
            "macs_per_pixel_M": float(m.mean() / 1e6),
            "macs_per_16x16_window_G": float(m.mean() * 256 / 1e9),
            "macs_per_16x16_window_G_p10_p50_p90": [
                float(np.percentile(m, q) * 256 / 1e9) for q in (10, 50, 90)
            ],
        }
        results[name] = r
        print(name, json.dumps(r), flush=True)
    allm = [r["macs_per_16x16_window_G"] for r in results.values()]
    print(
        "SUMMARY",
        json.dumps({"mean_over_datasets_G": float(np.mean(allm)), **results}),
        flush=True,
    )


if __name__ == "__main__":
    main()
