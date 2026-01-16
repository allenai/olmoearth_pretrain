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

import os
from collections import defaultdict

import torch
from einops import rearrange
from rslearn.dataset.dataset import Dataset as RslearnDataset
from rslearn.train.dataset import DataInput as RsDataInput
from rslearn.train.dataset import ModelDataset as RsModelDataset
from rslearn.train.dataset import SplitConfig as RsSplitConfig
from rslearn.train.tasks.task import Task as RsTask
from torch.utils.data import DataLoader
from tqdm import tqdm
from upath import UPath

# NOTE: These imports use olmoearth_pretrain paths (was helios.* in original script)
from olmoearth_pretrain.data.constants import YEAR_NUM_TIMESTEPS
from olmoearth_pretrain.data.constants import Modality as DataModality
from olmoearth_pretrain.data.utils import convert_to_db
from olmoearth_pretrain.evals.datasets.rslearn_dataset import (
    RSLEARN_TO_OLMOEARTH,
)

# Default to cpu_count - 1, but allow override via env var
_default_workers = (os.cpu_count() or 1) - 1
NUM_WORKERS = int(os.environ.get("OLMOEARTH_INGEST_WORKERS", _default_workers))

# Optional sampling limits via env vars
MAX_SAMPLES = (
    int(os.environ["OLMOEARTH_MAX_SAMPLES"])
    if "OLMOEARTH_MAX_SAMPLES" in os.environ
    else None
)
SAMPLE_FRACTION = (
    float(os.environ["OLMOEARTH_SAMPLE_FRACTION"])
    if "OLMOEARTH_SAMPLE_FRACTION" in os.environ
    else None
)


def build_rslearn_model_dataset_for_band_stats(
    rslearn_dataset: RslearnDataset,
    layers: list[str],
    task: RsTask,
    rslearn_dataset_groups: list[str] | None = None,
    input_size: int | None = None,
    split: str = "train",
    skip_targets: bool = True,
) -> RsModelDataset:
    """Build an rslearn ModelDataset for computing band statistics.

    Args:
        rslearn_dataset: The source RslearnDataset.
        layers: List of rslearn layer names to use as model inputs.
            Example: "sentinel2". Only provide the base name, do not include
            layer names such as "sentinel2.1" or "sentinel2.n".
        task: The rslearn Task instance (e.g., SegmentationTask, ClassificationTask).
            Instantiated directly from model config.
        rslearn_dataset_groups: Optional list of dataset group names to include.
        input_size: Optional input patch size (pixels) to crop/resize samples to.
        split: Dataset split to use (e.g., "train", "val", "test").
        skip_targets: Whether or not to skip the target, if True the task is only
            used to satisfy the RsModelDataset interface.

    Returns:
        RsModelDataset: A dataset object ready for training or evaluation.
    """
    if not layers:
        raise ValueError(
            "`layers` must be a non-empty list of rslearn layer names, "
            f"allowed: {list(RSLEARN_TO_OLMOEARTH.keys())}"
        )
    if split not in ("train", "val", "test"):
        raise ValueError(f"Invalid split {split}, must be one of train/val/test")

    # Validate input layers
    unknown = [m for m in layers if m not in RSLEARN_TO_OLMOEARTH]
    if unknown:
        raise ValueError(
            f"Unknown rslearn layer(s): {unknown}. "
            f"Allowed: {list(RSLEARN_TO_OLMOEARTH.keys())}"
        )

    # Group rslearn layers by their OlmoEarth Pretrain modality key
    layers_by_olmoearth: dict[str, list[str]] = defaultdict(list)
    bands_by_olmoearth: dict[str, list[str]] = {}

    for rslearn_layer in layers:
        olmoearth_key, band_order = RSLEARN_TO_OLMOEARTH[rslearn_layer]
        layers_by_olmoearth[olmoearth_key].append(rslearn_layer)
        bands_by_olmoearth[olmoearth_key] = band_order

    transforms = []
    if input_size is not None:
        raise NotImplementedError("Input size is not supported for band stats")
        # transforms.append(
        #     RsPad(
        #         size=input_size,
        #         mode="center",
        #         image_selectors=list(layers_by_olmoearth.keys()),
        #     )
        # )

    inputs: dict[str, RsDataInput] = {}
    # Expand each rslearn layer name to time-indexed variants, keep the first *per base layer*
    for olmoearth_key, per_key_layers in layers_by_olmoearth.items():
        expanded: list[str] = []
        for base in per_key_layers:
            # convention: base, then base.1 ... base.(YEAR_NUM_TIMESTEPS-1)
            expanded.append(base)
            expanded.extend(f"{base}.{i}" for i in range(1, YEAR_NUM_TIMESTEPS))
        inputs[olmoearth_key] = RsDataInput(
            data_type="raster",
            layers=expanded,
            bands=bands_by_olmoearth[olmoearth_key],
            passthrough=True,
            load_all_layers=True,
            required=False,
        )

    split_config = RsSplitConfig(
        transforms=transforms,
        groups=rslearn_dataset_groups,
        skip_targets=skip_targets,
        # TODO: Either we need a canonical tag or a way to get the split from the dataset
        # tags={"helios_split": split}
        # if split
        # else {},  # must stay as helios because it is tagged that way in the dataset
    )

    return RsModelDataset(
        dataset=rslearn_dataset,
        split_config=split_config,
        inputs=inputs,
        task=task,
        workers=NUM_WORKERS,
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
        if layer not in RSLEARN_TO_OLMOEARTH:
            raise ValueError(
                f"Unknown input layer '{layer}'. Allowed: {list(RSLEARN_TO_OLMOEARTH)}"
            )
        modality, band_order = RSLEARN_TO_OLMOEARTH[layer]
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
        num_workers=NUM_WORKERS,
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


def compute_band_stats_from_rslearn_dataset(
    dataset_path: str,
    modalities: list[str],
    task: RsTask,
    groups: list[str] | None = None,
    max_samples: int | None = MAX_SAMPLES,
    sample_fraction: float | None = SAMPLE_FRACTION,
) -> dict:
    """Compute band statistics from an rslearn dataset.

    Args:
        dataset_path: Path to the rslearn dataset.
        modalities: List of rslearn layer names (e.g., ["sentinel2", "sentinel1"]).
        task: The rslearn Task instance, instantiated from model config.
        groups: Optional list of dataset group names to filter by.
        max_samples: Optional maximum number of samples to process.
            Defaults to OLMOEARTH_MAX_SAMPLES env var if set.
        sample_fraction: Optional fraction of samples to use (0.0-1.0).
            Defaults to OLMOEARTH_SAMPLE_FRACTION env var if set.
            If both max_samples and sample_fraction are set, the smaller limit is used.

    Returns:
        Nested dict of band statistics per modality.
    """
    base_ds = RslearnDataset(UPath(dataset_path))
    model_ds = build_rslearn_model_dataset_for_band_stats(
        rslearn_dataset=base_ds,
        layers=modalities,
        task=task,
        rslearn_dataset_groups=groups,
        split="train",
        skip_targets=True,
    )

    # Apply sampling if requested
    if max_samples is not None or sample_fraction is not None:
        import random

        total_samples = len(model_ds)
        target_samples = total_samples

        if sample_fraction is not None:
            target_samples = min(target_samples, int(total_samples * sample_fraction))
        if max_samples is not None:
            target_samples = min(target_samples, max_samples)

        if target_samples < total_samples:
            indices = random.sample(range(total_samples), target_samples)
            model_ds = torch.utils.data.Subset(model_ds, indices)

    bands_by_modality = get_bands_by_modality(modalities)
    stats = compute_band_stats(model_ds, bands_by_modality)

    return stats
