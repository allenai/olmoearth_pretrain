"""Compute band stats from rslearn dataset.

This module computes per-band normalization statistics (mean, std, min, max)
for rslearn datasets. These stats are used for dataset-specific normalization
during evaluation.

Key features:
- Uses the same dataset builder as eval (rslearn_builder.py)
- Handles variable timesteps (infers from data shape)
- GPU acceleration when available
- Optional sampling for large datasets

Usage as library:
    from olmoearth_pretrain.evals.studio_ingest.band_stats import (
        compute_band_stats_from_model_config,
    )

    stats = compute_band_stats_from_model_config(
        model_config_path="/path/to/model.yaml",
        source_path="/path/to/dataset",
    )
"""

import os
import random

import torch
from einops import rearrange
from torch.utils.data import DataLoader
from tqdm import tqdm

from olmoearth_pretrain.data.constants import Modality as DataModality
from olmoearth_pretrain.evals.constants import RSLEARN_TO_OLMOEARTH
from olmoearth_pretrain.data.utils import convert_to_db
from olmoearth_pretrain.evals.datasets.rslearn_builder import (
    build_model_dataset_from_config,
    load_runtime_config,
)

# Default to 0 (no multiprocessing), but allow override via env var
_default_workers = 0
NUM_WORKERS = int(os.environ.get("OLMOEARTH_INGEST_WORKERS", _default_workers))


def _collate_inputs_only(batch):
    """Collate function that returns samples as-is (no stacking).

    This handles variable-shaped tensors by not attempting to stack them.
    """
    return batch


def _resolve_layer_name(layer: str) -> str | None:
    """Resolve an rslearn layer name to a key in RSLEARN_TO_OLMOEARTH.

    Also handles layer names prefixed with "pre_" or "post_" to make
    compatible with older datasets that use these prefixed layer names.
    """
    if layer in RSLEARN_TO_OLMOEARTH:
        return layer
    for prefix in ("pre_", "post_"):
        if layer.startswith(prefix):
            stripped = layer[len(prefix):]
            if stripped in RSLEARN_TO_OLMOEARTH:
                return stripped
    return None


def _get_bands_by_modality_from_runtime_config(runtime_config) -> dict[str, list[str]]:
    """Extract bands by modality from runtime config.

    Args:
        runtime_config: RuntimeConfig with parsed model.yaml.

    Returns:
        Dict mapping OlmoEarth modality name -> list of band names
    """
    bands_by_modality = {}
    modality_layers = runtime_config.get_modality_layers()

    for layer in modality_layers:
        resolved = _resolve_layer_name(layer)
        if resolved is not None:
            modality = RSLEARN_TO_OLMOEARTH[resolved]
            bands_by_modality[modality.name] = modality.band_order

    return bands_by_modality


def compute_band_stats(
    model_ds,
    bands_by_modality: dict[str, list[str]],
    batch_size: int = 8,
) -> dict:
    """Compute mean/std/min/max for each band in each modality.

    Key features:
    - Infers timesteps from actual data shape (handles variable timesteps)
    - Handles variable-shaped samples (no stacking failures)
    - GPU acceleration when available

    Args:
        model_ds: A torch Dataset that yields (inputs_dict, target) tuples
        bands_by_modality: Dict mapping modality name -> list of band names
        batch_size: Batch size for DataLoader

    Returns:
        Nested dict: {modality: {band: {"mean": ..., "std": ..., "min": ..., "max": ...}}}

    Example output:
        {
            "sentinel2_l2a": {
                "B02": {"mean": 1234.5, "std": 456.7, "min": 0, "max": 10000},
                ...
            }
        }
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

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

    num_workers = min(8, NUM_WORKERS) if NUM_WORKERS > 0 else 0
    loader = DataLoader(
        model_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate_inputs_only,
    )

    for batch in tqdm(loader, total=len(loader), desc="Computing stats"):
        for sample in batch:
            # Handle both (inputs_dict, target) tuples and raw dicts
            inputs_dict = sample[0] if isinstance(sample, tuple) else sample

            for modality, bands in bands_by_modality.items():
                if modality not in inputs_dict:
                    continue

                x = inputs_dict[modality]

                # Handle RasterImage objects (when passthrough=True in model.yaml)
                if hasattr(x, 'image'):
                    # RasterImage.image has shape (C, T, H, W)
                    x = x.image

                if not isinstance(x, torch.Tensor):
                    continue

                # Move to device for faster computation
                x = x.to(device, non_blocking=True)

                # Infer shape: expect (T*C, H, W) or (C, H, W) or (C, T, H, W)
                C = len(bands)

                if x.ndim == 4:
                    # Shape: (C, T, H, W) - rearrange to (T*C, H, W)
                    C_actual, T, H, W = x.shape
                    if C_actual == C:
                        x = rearrange(x, "c t h w -> (t c) h w")
                    else:
                        # Unexpected shape, skip
                        continue

                if x.ndim == 3:
                    TC, H, W = x.shape
                    # Infer T from channel count
                    if TC % C == 0:
                        T = TC // C
                    else:
                        # Can't determine structure, treat as single timestep
                        T = 1
                        C = TC
                else:
                    continue

                # Convert Sentinel-1 to dB scale
                if modality == DataModality.SENTINEL1.name:
                    x = convert_to_db(x)

                # Reshape to (T, C, H*W) for per-band processing
                try:
                    x = x.view(T, C, -1)  # (T, C, H*W)
                except RuntimeError:
                    # Shape mismatch, skip this sample
                    continue

                # Compute stats per band (vectorized across spatial dims)
                for band_idx, band in enumerate(bands):
                    if band_idx >= x.shape[1]:
                        continue

                    vals = x[:, band_idx, :].flatten()  # (T * H * W,)
                    finite_mask = torch.isfinite(vals)
                    vals = vals[finite_mask]

                    if vals.numel() == 0:
                        continue

                    s = acc[modality][band]

                    # Online accumulation
                    s["count"] += vals.numel()
                    s["sum"] += vals.sum().item()
                    s["sumsq"] += (vals ** 2).sum().item()
                    s["min"] = min(s["min"], vals.min().item())
                    s["max"] = max(s["max"], vals.max().item())

    # Finalize: compute mean and std from accumulated values
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
                    "std": var ** 0.5,
                    "min": s["min"],
                    "max": s["max"],
                }

    return out


def compute_band_stats_from_model_config(
    model_config_path: str,
    source_path: str,
    groups: list[str] | None = None,
    tags: dict[str, list[str]] | None = None,
    num_samples: int | None = None,
    seed: int = 42,
) -> dict:
    """Compute band statistics using the same dataset builder as eval.

    This uses build_model_dataset_from_config from rslearn_builder.py,
    ensuring the dataset is built with the correct configuration from model.yaml.

    Args:
        model_config_path: Path to model.yaml file.
        source_path: Path to the rslearn dataset.
        groups: Optional list of dataset group names to filter by.
        tags: Optional dict of tag filters (e.g., {"split": ["val"]}).
        num_samples: Number of samples to process. If None, processes all samples.
        seed: Random seed for reproducible sampling.

    Returns:
        Nested dict of band statistics per modality.
    """
    random.seed(seed)

    # Load runtime config from model.yaml
    runtime_config = load_runtime_config(model_config_path, source_path)

    if not runtime_config.model_config:
        raise ValueError(
            f"Failed to load model.yaml from {model_config_path}. "
            "Check that the file exists and is valid YAML."
        )

    # Build dataset using the same builder as eval
    model_ds = build_model_dataset_from_config(
        runtime_config=runtime_config,
        source_path=source_path,
        split="train",
        groups_override=groups,
        tags_override=tags,
    )

    total_samples = len(model_ds)

    # Apply sampling if requested
    if num_samples is not None and num_samples < total_samples:
        print(f"Sampling {num_samples} of {total_samples} samples for stats computation")
        indices = random.sample(range(total_samples), num_samples)
        model_ds = torch.utils.data.Subset(model_ds, indices)
    else:
        print(f"Processing all {total_samples} samples")

    # Get bands by modality from runtime config
    bands_by_modality = _get_bands_by_modality_from_runtime_config(runtime_config)

    if not bands_by_modality:
        raise ValueError("No modalities found in model config")

    band_stats = compute_band_stats(model_ds, bands_by_modality)
    # make sure none of the stats are None
    for modality, bands in band_stats.items():
        for band, stats in bands.items():
            if any(value is None for value in stats.values()):
                raise ValueError(f"Stats for {modality} {band} are None {stats}")

    # print the keys of the stats

    return band_stats
