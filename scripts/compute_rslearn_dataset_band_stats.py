"""Compute band stats from rslearn dataset."""

import argparse
import json

import torch
from einops import rearrange
from rslearn.dataset.dataset import Dataset as RslearnDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from upath import UPath

from helios.data.constants import YEAR_NUM_TIMESTEPS
from helios.data.constants import Modality as DataModality
from helios.data.utils import convert_to_db
from helios.evals.datasets.rslearn_dataset import (
    RSLEARN_TO_HELIOS,
    build_rslearn_model_dataset,
)


def get_bands_by_modality(input_layers: list[str]) -> dict[str, list[str]]:
    """Collect Helios band order for each requested modality."""
    bands_by_modality = {}
    for layer in input_layers:
        if layer not in RSLEARN_TO_HELIOS:
            raise ValueError(
                f"Unknown input layer '{layer}'. Allowed: {list(RSLEARN_TO_HELIOS)}"
            )
        modality, band_order = RSLEARN_TO_HELIOS[layer]
        bands_by_modality[modality] = band_order
    return bands_by_modality


def collate_inputs_only(batch):
    """Collate only the input dicts."""
    return [item[0] for item in batch]


def compute_band_stats(model_ds, bands_by_modality: dict[str, list[str]]) -> dict:
    """Compute mean/std/min/max for each band in each modality."""
    # Accumulators for band stats
    acc = {
        modality: {
            band: {
                "count": 0,
                "sum": 0.0,
                "sumsq": 0.0,
                "min": float("inf"),
                "max": float("-inf"),
            }
            for band in bands
        }
        for modality, bands in bands_by_modality.items()
    }

    loader = DataLoader(
        model_ds,
        batch_size=128,
        shuffle=False,
        num_workers=32,
        collate_fn=collate_inputs_only,
    )

    for batch_inputs in tqdm(loader, total=len(loader), desc="Computing stats"):
        for modality, bands in bands_by_modality.items():
            if modality not in batch_inputs[0]:
                continue
            cur = torch.stack([inp[modality] for inp in batch_inputs], dim=0)
            if cur.ndim == 3:
                cur = cur.unsqueeze(0)
            B, TC, H, W = cur.shape
            T, C = YEAR_NUM_TIMESTEPS, len(bands)

            if TC != T * C:
                raise ValueError(f"{modality}: expected T*C={T * C}, got {TC}")

            if modality == DataModality.SENTINEL1.name:
                cur = convert_to_db(cur)

            cur = rearrange(cur, "b (t c) h w -> (b h w t) c", t=T, c=C)
            finite = torch.isfinite(cur)

            for band_idx, band in enumerate(bands):
                vals = cur[:, band_idx][finite[:, band_idx]]
                if vals.numel() == 0:
                    continue
                s = acc[modality][band]
                s["count"] += vals.numel()
                s["sum"] += vals.sum().item()
                s["sumsq"] += (vals * vals).sum().item()
                vmin, vmax = vals.min().item(), vals.max().item()
                s["min"] = min(s["min"], vmin)
                s["max"] = max(s["max"], vmax)

    # Finalize stats
    out = {}
    for modality, bands in bands_by_modality.items():
        out[modality] = {}
        for band in bands:
            s = acc[modality][band]
            if s["count"] == 0:
                out[modality][band] = {
                    "mean": None,
                    "std": None,
                    "min": None,
                    "max": None,
                }
            else:
                mean = s["sum"] / s["count"]
                var = max(0.0, s["sumsq"] / s["count"] - mean * mean)
                out[modality][band] = {
                    "mean": mean,
                    "std": var**0.5,
                    "min": s["min"],
                    "max": s["max"],
                }
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ds_path", required=True)
    p.add_argument("--ds_group", action="append")
    p.add_argument("--input_layers", nargs="+", required=True)
    p.add_argument("--input_size", type=int, default=4)
    p.add_argument("--output_json", required=True)
    args = p.parse_args()

    base_ds = RslearnDataset(UPath(args.ds_path))
    model_ds = build_rslearn_model_dataset(
        rslearn_dataset=base_ds,
        layers=args.input_layers,
        rslearn_dataset_groups=args.ds_group,
        input_size=args.input_size,
        split="train",
        property_name="placeholder",
        classes=["cls0"],
        skip_targets=True,
    )

    bands_by_modality = get_bands_by_modality(args.input_layers)
    stats = compute_band_stats(model_ds, bands_by_modality)

    with UPath(args.output_json).open("w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved stats -> {args.output_json}")
