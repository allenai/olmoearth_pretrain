"""Input corruption strategies for ERA5 reconstruction (objective B).

Two complementary masking strategies operate on normalized ERA5 input
``[B, T, V]``:

* **Short time masks** — contiguous spans of timesteps where *all* V
  variables are zeroed (forces the encoder to interpolate across time).
* **Variable-group masks** — entire variable groups dropped for *all* T
  timesteps (forces cross-variable reasoning).

Both are applied on-the-fly on the GPU and respect the existing
``ignore_mask`` (padding).  Masked positions are set to 0 (the
post-normalization mean).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default variable groups for ERA5L_DAY_10 (14 bands)
# ---------------------------------------------------------------------------

# Band order from Modality.ERA5L_DAY_10:
#   0: d2m, 1: e, 2: pev, 3: ro, 4: sp, 5: ssr, 6: ssrd, 7: str,
#   8: swvl1, 9: swvl2, 10: t2m, 11: tp, 12: u10, 13: v10
DEFAULT_VARIABLE_GROUPS: dict[str, list[int]] = {
    "temperature": [0, 10],  # d2m, t2m
    "wind": [12, 13],  # u10, v10
    "radiation": [5, 6, 7],  # ssr, ssrd, str
    "soil_moisture": [8, 9],  # swvl1, swvl2
    "water_flux": [1, 2, 3, 11],  # e, pev, ro, tp
    "pressure": [4],  # sp
}


@dataclass
class CorruptionConfig:
    """Configuration for ERA5 input corruption.

    Args:
        num_time_masks: Number of contiguous time spans to mask per sample.
        time_mask_min_len: Minimum span length (in timesteps).
        time_mask_max_len: Maximum span length (in timesteps).
        num_variable_group_masks: Number of variable groups to drop per
            sample (sampled without replacement from the group dict).
        variable_groups: Mapping from group name to list of band indices.
        min_mask_ratio: Ensure at least this fraction of (T*V) cells are
            masked (repeat time-mask sampling if needed).
    """

    num_time_masks: int = 3
    time_mask_min_len: int = 7
    time_mask_max_len: int = 30
    num_variable_group_masks: int = 1
    variable_groups: dict[str, list[int]] = field(
        default_factory=lambda: dict(DEFAULT_VARIABLE_GROUPS)
    )
    min_mask_ratio: float = 0.15


def corrupt_era5(
    era5: Tensor,
    ignore_mask: Tensor,
    config: CorruptionConfig,
) -> tuple[Tensor, Tensor]:
    """Apply corruption to a batch of ERA5 sequences.

    Args:
        era5: ``[B, T, V]`` normalized input.
        ignore_mask: ``[B, T]`` bool, True = padded / invalid timestep.
        config: Corruption settings.

    Returns:
        ``(era5_corrupted, mask)`` where *mask* is ``[B, T, V]`` bool
        (True = corrupted / zeroed position).  The corrupted tensor has
        masked positions set to 0.
    """
    b, t, v = era5.shape
    device = era5.device
    mask = torch.zeros(b, t, v, dtype=torch.bool, device=device)
    valid = ~ignore_mask  # [B, T]

    # --- Short time masks ---
    for _ in range(config.num_time_masks):
        lengths = torch.randint(
            config.time_mask_min_len,
            config.time_mask_max_len + 1,
            (b,),
            device=device,
        )
        max_start = (t - lengths).clamp(min=0)
        starts = (torch.rand(b, device=device) * (max_start.float() + 1)).long()
        # Build per-sample time ranges
        idx = torch.arange(t, device=device).unsqueeze(0)  # [1, T]
        span = (idx >= starts.unsqueeze(1)) & (idx < (starts + lengths).unsqueeze(1))
        span = span & valid  # don't mask padding
        mask = mask | span.unsqueeze(-1).expand_as(mask)

    # --- Variable-group masks ---
    group_names = list(config.variable_groups.keys())
    num_groups = len(group_names)
    if num_groups > 0 and config.num_variable_group_masks > 0:
        n_drop = min(config.num_variable_group_masks, num_groups)
        for i in range(b):
            chosen = torch.randperm(num_groups, device=device)[:n_drop]
            for g_idx in chosen:
                band_indices = config.variable_groups[group_names[int(g_idx)]]
                for bi in band_indices:
                    mask[i, :, bi] = mask[i, :, bi] | valid[i]

    # Apply mask: zero out corrupted positions
    era5_corrupted = era5.clone()
    era5_corrupted[mask] = 0.0

    return era5_corrupted, mask
