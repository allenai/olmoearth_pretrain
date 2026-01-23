"""Shared functions across evaluation datasets."""

import json
from collections.abc import Sequence
from functools import lru_cache
from importlib.resources import files

import torch
from torch.utils.data import default_collate

from olmoearth_pretrain.train.masking import MaskedOlmoEarthSample


def eval_collate_fn(
    batch: Sequence[tuple[MaskedOlmoEarthSample, torch.Tensor]],
) -> tuple[MaskedOlmoEarthSample, torch.Tensor]:
    """Collate function for DataLoaders."""
    samples, targets = zip(*batch)
    # we assume that the same values are consistently None
    collated_sample = default_collate([s.as_dict(return_none=False) for s in samples])
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
    from olmoearth_pretrain.datatypes import MaskValue

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
                padding = torch.zeros((h, w, pad_size, c), dtype=val.dtype)
                padded[modality] = torch.cat([val, padding], dim=2)
            else:
                padded[modality] = val

            # Pad mask with MISSING
            if mask_val is not None and pad_size > 0:
                mask_pad = torch.full(
                    (h, w, pad_size, c), MaskValue.MISSING.value, dtype=mask_val.dtype
                )
                padded[mask_key] = torch.cat([mask_val, mask_pad], dim=2)
            elif mask_val is not None:
                padded[mask_key] = mask_val

        # Pad timestamps
        ts = s.timestamps
        if ts is not None:
            pad_size = max_t - ts.shape[0]
            if pad_size > 0:
                padding = torch.zeros((pad_size, ts.shape[1]), dtype=ts.dtype)
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
