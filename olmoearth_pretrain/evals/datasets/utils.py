"""Shared functions across evaluation datasets."""

import json
from collections.abc import Sequence
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import TypeAlias

import numpy as np
import torch
from torch.utils.data import default_collate

from olmoearth_pretrain.datatypes import (
    MaskedOlmoEarthSample,
    MaskValue,
    OlmoEarthSample,
)

LabelFractionPartitionMap: TypeAlias = dict[float, str | None]

STANDARD_LABEL_FRACTION_PARTITIONS: LabelFractionPartitionMap = {
    0.01: "0.01x_train",
    0.02: "0.02x_train",
    0.05: "0.05x_train",
    0.10: "0.10x_train",
    0.20: "0.20x_train",
    0.50: "0.50x_train",
    1.00: None,
}


def resolve_label_fraction_partition(
    label_fraction: float,
    partitions: LabelFractionPartitionMap = STANDARD_LABEL_FRACTION_PARTITIONS,
) -> str | None:
    """Return the partition basename for a label fraction, validating support."""
    if label_fraction not in partitions:
        valid = ", ".join(f"{value:g}" for value in sorted(partitions))
        raise ValueError(
            f"Unsupported label_fraction {label_fraction}. Supported values are: {valid}"
        )
    return partitions[label_fraction]


def load_label_fraction_partition_indices(
    partition_dir: Path, partition: str
) -> list[int]:
    """Load train indices for a low-label partition basename."""
    with open(partition_dir / f"{partition}_partition.json") as json_file:
        return json.load(json_file)


def build_band_stat_arrays(
    band_stats: dict[str, dict[str, float]], band_names: Sequence[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build mean/std/min/max arrays ordered by band names."""
    means = []
    stds = []
    mins = []
    maxs = []
    for band_name in band_names:
        if band_name not in band_stats:
            raise KeyError(f"{band_name} not found in band_stats")
        stats = band_stats[band_name]
        means.append(stats["mean"])
        stds.append(stats["std"])
        mins.append(stats["min"])
        maxs.append(stats["max"])
    return np.array(means), np.array(stds), np.array(mins), np.array(maxs)


def build_masked_eval_sample(
    modality_tensors: dict[str, torch.Tensor],
    timestamps: torch.Tensor,
) -> MaskedOlmoEarthSample:
    """Build a masked eval sample from prepared modality tensors."""
    if not modality_tensors:
        raise ValueError("At least one modality tensor is required.")
    return MaskedOlmoEarthSample.from_olmoearthsample(
        OlmoEarthSample(**modality_tensors, timestamps=timestamps.long())
    )


def eval_collate_fn(
    batch: Sequence[tuple[MaskedOlmoEarthSample, torch.Tensor]],
) -> tuple[MaskedOlmoEarthSample, torch.Tensor]:
    """Collate function for DataLoaders."""
    samples, targets = zip(*batch)
    # we assume that the same values are consistently None
    collated_sample = default_collate([s.as_dict() for s in samples])
    collated_target = default_collate([t for t in targets])
    return MaskedOlmoEarthSample(**collated_sample), collated_target


def eval_collate_fn_variable_time(
    batch: Sequence[tuple[MaskedOlmoEarthSample, torch.Tensor]],
) -> tuple[MaskedOlmoEarthSample, torch.Tensor]:
    """Collate function for DataLoaders with variable temporal lengths.

    Pads modality tensors along the T dimension to the max in the batch.
    Expected tensor shape: (H, W, T, C) per sample, batched to (B, H, W, T, C).
    Padded timesteps get MaskValue.MISSING in their mask tensors.
    """
    samples, targets = zip(*batch)

    # Find max temporal length using sample.modalities property
    max_t = 0
    for s in samples:
        for modality in s.modalities:
            val = getattr(s, modality)
            if val is not None and val.ndim == 4:  # (H, W, T, C)
                max_t = max(max_t, val.shape[2])
    # Pad each sample
    padded_dicts = []
    for s in samples:
        padded = {}

        # Pad modalities and their masks
        for modality in s.modalities:
            val = getattr(s, modality)
            mask_key = MaskedOlmoEarthSample.get_masked_modality_name(modality)
            mask_val = getattr(s, mask_key, None)

            # Non-4D modalities (like latlon) - just copy as-is
            if val is None or val.ndim != 4:
                padded[modality] = val
                if mask_val is not None:
                    padded[mask_key] = mask_val
                continue

            h, w, t, c = val.shape
            pad_size = max_t - t

            # Pad data with zeros
            if pad_size > 0:
                padding = torch.zeros(
                    (h, w, pad_size, c), dtype=val.dtype, device=val.device
                )
                padded[modality] = torch.cat([val, padding], dim=2)
            else:
                padded[modality] = val

            # Pad mask with MISSING
            if mask_val is not None and pad_size > 0:
                mask_pad = torch.full(
                    (*mask_val.shape[:2], pad_size, mask_val.shape[-1]),
                    MaskValue.MISSING.value,
                    dtype=mask_val.dtype,
                    device=mask_val.device,
                )
                padded[mask_key] = torch.cat([mask_val, mask_pad], dim=2)
            elif mask_val is not None:
                padded[mask_key] = mask_val

        # Pad timestamps
        ts = s.timestamps
        if ts is not None:
            pad_size = max_t - ts.shape[0]
            if pad_size > 0:
                padding = torch.zeros(
                    (pad_size, ts.shape[1]), dtype=ts.dtype, device=ts.device
                )
                padded["timestamps"] = torch.cat([ts, padding], dim=0)
            else:
                padded["timestamps"] = ts

        padded_dicts.append(padded)

    collated_sample = default_collate(padded_dicts)
    collated_target = default_collate(list(targets))
    return MaskedOlmoEarthSample(**collated_sample), collated_target


@lru_cache(maxsize=1)
def load_min_max_stats() -> dict:
    """Load the min/max stats for a given dataset."""
    with (
        files("olmoearth_pretrain.evals.datasets.config") / "minmax_stats.json"
    ).open() as f:
        return json.load(f)
