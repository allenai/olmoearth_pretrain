"""Compute band stats from rslearn dataset.

Moved from scripts/tools/compute_rslearn_dataset_band_stats.py.

This module computes per-band normalization statistics (mean, std, min, max)
for rslearn datasets. These stats are used for dataset-specific normalization
during evaluation.

Usage as CLI:
    uv run --group ingest python -m olmoearth_pretrain.evals.studio_ingest.band_stats \\
        --ds_path gs://bucket/dataset \\
        --input_layers sentinel2 sentinel1 \\
        --output_json /path/to/stats.json

Usage as library:
    from olmoearth_pretrain.evals.studio_ingest.band_stats import compute_band_stats

    stats = compute_band_stats(model_ds, bands_by_modality)

Todo:
-----
- [ ] Add percentile computation (p1, p99) for robust normalization
- [ ] Add sampling support for large datasets (currently processes all data)
- [ ] Integrate with ingestion workflow
"""

import argparse
import json

import torch
from einops import rearrange
from rslearn.dataset.dataset import Dataset as RslearnDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from upath import UPath

# NOTE: These imports use olmoearth_pretrain paths (was helios.* in original script)
from olmoearth_pretrain.data.constants import YEAR_NUM_TIMESTEPS
from olmoearth_pretrain.data.constants import Modality as DataModality
from olmoearth_pretrain.data.utils import convert_to_db
from olmoearth_pretrain.evals.datasets.rslearn_dataset import (
    RSLEARN_TO_HELIOS,
    build_rslearn_model_dataset,
)


def get_bands_by_modality(input_layers: list[str]) -> dict[str, list[str]]:
    """Collect Helios band order for each requested modality.

    Args:
        input_layers: List of rslearn layer names (e.g., ["sentinel2", "sentinel1"])

    Returns:
        Dict mapping modality name -> list of band names

    Raises:
        ValueError: If an unknown layer is provided
    """
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
    """Collate only the input dicts (ignore targets)."""
    return [item[0] for item in batch]


def compute_band_stats(model_ds, bands_by_modality: dict[str, list[str]]) -> dict:
    """Compute mean/std/min/max for each band in each modality.

    Uses Welford-style online accumulation for numerical stability.
    Processes all samples via DataLoader with multiprocessing.

    Args:
        model_ds: A torch Dataset that yields (inputs_dict, target) tuples
        bands_by_modality: Dict mapping modality name -> list of band names

    Returns:
        Nested dict: {modality: {band: {"mean": ..., "std": ..., "min": ..., "max": ...}}}

    Example output:
        {
            "sentinel2_l2a": {
                "B02": {"mean": 1234.5, "std": 456.7, "min": 0, "max": 10000},
                "B03": {"mean": 2345.6, "std": 567.8, "min": 0, "max": 10000},
                ...
            }
        }
    """
    # Initialize accumulators for online stats computation
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

            # Stack batch into tensor: (B, T*C, H, W)
            cur = torch.stack([inp[modality] for inp in batch_inputs], dim=0)
            if cur.ndim == 3:
                cur = cur.unsqueeze(0)
            B, TC, H, W = cur.shape
            T, C = YEAR_NUM_TIMESTEPS, len(bands)

            if TC != T * C:
                raise ValueError(f"{modality}: expected T*C={T * C}, got {TC}")

            # Convert Sentinel-1 to dB scale
            if modality == DataModality.SENTINEL1.name:
                cur = convert_to_db(cur)

            # Reshape to (B*H*W*T, C) for per-band processing
            cur = rearrange(cur, "b (t c) h w -> (b h w t) c", t=T, c=C)
            finite = torch.isfinite(cur)

            # Accumulate stats per band
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

    # Finalize: compute mean and std from accumulated sums
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


def main():
    """CLI entry point for computing band stats."""
    p = argparse.ArgumentParser(
        description="Compute per-band normalization stats from rslearn dataset"
    )
    p.add_argument("--ds_path", required=True, help="Path to rslearn dataset")
    p.add_argument(
        "--ds_group",
        action="append",
        help="Dataset group(s) to include (can specify multiple)",
    )
    p.add_argument(
        "--input_layers",
        nargs="+",
        required=True,
        help="Input layers (e.g., sentinel2 sentinel1)",
    )
    p.add_argument(
        "--input_size", type=int, default=4, help="Input patch size (default: 4)"
    )
    p.add_argument("--output_json", required=True, help="Output JSON path")
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


if __name__ == "__main__":
    main()
