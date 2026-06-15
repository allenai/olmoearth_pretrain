"""Input corruption strategies for ERA5 reconstruction (objective B).

Masking operates on normalized ERA5 input ``[B, T, V]`` and combines:

* **Short time masks** — contiguous spans of timesteps where *all* V
  variables are masked (forces the encoder to interpolate across time).
* **Policy-driven variable masks** — a configurable mask *policy* that,
  per sample, samples one masking strategy from a weighted menu (see
  :data:`MASK_POLICY_V1`).  Strategies range from dropping a single
  variable within a group to dropping a whole physical group across a
  span of days or the full sequence.  This replaces the older fixed
  "drop N whole groups for all T" behaviour (still available as a legacy
  fallback when ``mask_policy`` is ``None``).

Note: ERA5L_DAY_10 has one timestep per day, so span lengths expressed
in *days* map directly onto timesteps.

Everything is applied on-the-fly on the GPU and respects the existing
``ignore_mask`` (padding).

:func:`corrupt_era5` returns a boolean corruption mask.  The encoder
replaces masked positions with a learned per-band embedding before
patchifying (see ``use_mask_embed`` in
:class:`~olmoearth_pretrain.nn.era5_encoder.Era5DailyEncoder`).
"""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default variable groups for ERA5L_DAY_10 (14 bands)
# ---------------------------------------------------------------------------

# Band order from Modality.ERA5L_DAY_10:
#   0: d2m, 1: e, 2: pev, 3: ro, 4: sp, 5: ssr, 6: ssrd, 7: str,
#   8: swvl1, 9: swvl2, 10: t2m, 11: tp, 12: u10, 13: v10

# radiation
# ssr — surface net solar (shortwave) radiation (J/m², accumulated). Incoming solar minus reflected.
# ssrd — surface solar radiation downwards (J/m²). Total incoming shortwave at the surface.
# str — surface net thermal (longwave) radiation (J/m²). Net longwave; typically negative (surface loses heat).
# ssr and ssrd differ only by surface albedo, so they're nearly collinear; str is the longwave counterpart. All driven by the same cloud/insolation regime.

# swvl1 — volumetric soil water, layer 1 (m³/m³), 0–7 cm depth.
# swvl2 — volumetric soil water, layer 2 (m³/m³), 7–28 cm depth.

# water_flux [1, 2, 3, 11]
# e — total evaporation (m of water equivalent; negative = upward flux from surface).
# pev — potential evaporation (m). Evaporation that would occur given unlimited water — an atmospheric demand proxy.
# ro — runoff (m). Surface + sub-surface water leaving the cell.
# tp — total precipitation (m). Rain + snow water equivalent.
# These are the components of the local surface water balance (precip in; evaporation and runoff out), so they're physically coupled.

# TODO: for V2
# add (or reconstruct?) derived channels
# wind_speed = sqrt(u10**2 + v10**2)
# vpd_proxy or rh_proxy from t2m, d2m, sp
# net_radiation = ssr + str
# albedo_proxy = 1 - ssr / ssrd
# Then groups could be:
# ERA5_GROUPS_WITH_DERIVED = {
#    "thermo_humidity": ["t2m", "d2m", "sp", "vpd_or_rh"],
#    "wind": ["u10", "v10", "wind_speed"],
#    "radiation": ["ssr", "ssrd", "str", "net_radiation", "albedo_proxy"],
#    "soil_moisture": ["swvl1", "swvl2"],
#    "evaporative_flux": ["e", "pev"],
#    "hydro_flux": ["tp", "ro"],
# }

DEFAULT_VARIABLE_GROUPS: dict[str, list[int]] = {
    # Near-surface thermodynamic state.
    "thermo": [0, 10],  # d2m, t2m
    # Wind vector
    "wind": [12, 13],  # u10, v10
    # Shortwave radiation pair: strongly related through albedo/cloud/insolation.
    "shortwave_radiation": [5, 6],  # ssr, ssrd
    # Longwave radiation is related, but not as redundant with shortwave.
    "longwave_radiation": [7],  # str
    # Land water storage / memory.
    "soil_moisture": [8, 9],  # swvl1, swvl2
    # Water input and output response.
    "hydro_flux": [3, 11],  # ro, tp
    # Evaporative demand / realized evaporation.
    "evaporation": [1, 2],  # e, pev
    # Synoptic/static-ish pressure signal.
    "pressure": [4],  # sp
}


# ---------------------------------------------------------------------------
# Per-group controls
# ---------------------------------------------------------------------------
#
# These tables describe, for each variable group, *how* it should be
# reconstructed and *how* it is allowed to be masked.  They are the single
# source of truth that :data:`MASK_POLICY_V1` is assembled from.
GROUP_RECON_MODE: dict[str, str] = {
    "thermo": "raw_plus_wavelet",
    "wind": "raw_plus_wavelet",
    "shortwave_radiation": "raw_plus_wavelet",
    "longwave_radiation": "raw_plus_wavelet",
    "soil_moisture": "raw_plus_slow_wavelet",
    "evaporation": "raw_plus_wavelet",
    "hydro_flux": "short_raw_plus_slow_wavelet",
    "pressure": "slow_wavelet",
}

# Groups that may be masked across the *entire* sequence (all T).  These are
# slow / persistent or strongly cross-correlated signals where dropping the
# whole record still leaves a learnable cross-variable reconstruction task.
FULL_T_GROUP_MASK_ALLOWED: set[str] = {
    "shortwave_radiation",
    "longwave_radiation",
    "soil_moisture",
    "evaporation",
    "pressure",
}

WITHIN_GROUP_SINGLE_VAR_GROUPS = {
    group for group, vars_ in DEFAULT_VARIABLE_GROUPS.items() if len(vars_) >= 2
}

# Groups that may *only* be masked over a contiguous span (never all T).
# These are fast / high-frequency signals (e.g. precipitation) where a
# full-record drop would be ill-posed; we only ever hide short windows.
SPAN_ONLY_GROUP_MASK: set[str] = {
    "thermo",
    "hydro_flux",
}

# Span length range (in days == timesteps for ERA5L_DAY_10) per group.
GROUP_SPAN_DAYS: dict[str, tuple[int, int]] = {
    "thermo": (1, 7),
    "wind": (3, 30),
    "shortwave_radiation": (3, 30),
    "longwave_radiation": (3, 30),
    "soil_moisture": (7, 90),
    "evaporation": (7, 30),
    "hydro_flux": (1, 7),
    "pressure": (7, 90),
}

# Long-span length range (in days) used by the single-variable masking
# strategy.  Group-specific because a long in-paint is only fair when the
# remaining channels of the group carry enough signal about the dropped one.
# A global (30, 180) is fine for highly redundant pairs (ssr/ssrd, swvl1/
# swvl2, e/pev) but far too hard for weakly mutually-predictive pairs such
# as tp from ro or u10 from v10, so those get shorter windows.
WITHIN_GROUP_LONG_SPAN_DAYS: dict[str, tuple[int, int]] = {
    "thermo": (7, 60),
    "wind": (3, 30),
    "shortwave_radiation": (30, 180),
    "soil_moisture": (30, 180),
    "evaporation": (14, 90),
    "hydro_flux": (1, 14),
}


# ---------------------------------------------------------------------------
# Mask policy
# ---------------------------------------------------------------------------
#
# A *mask policy* is a weighted menu of masking strategies.  For every
# sample in the batch we draw exactly one strategy (according to ``prob``)
# and apply it.  ``prob`` values are normalized, so they need not sum to 1.
#
# Supported strategies (keyed by name) and the fields they read:
#
#   within_group_single_var
#       Drop a single variable inside a randomly chosen group.  If the
#       group is in ``full_t_allowed`` then, with probability
#       ``all_T_prob``, the variable is masked across *all* timesteps;
#       otherwise (or for span-only groups) it is masked across a
#       contiguous span.  Full-T-allowed groups use ``long_span_days``;
#       span-only groups use their per-group ``span_days`` range.
#
#   whole_group_span
#       Drop an entire group across a contiguous span.  The span length
#       (in days) is sampled from the per-group ``span_days`` range.  The
#       group is chosen uniformly from the keys of ``span_days``.
#
#   whole_group_all_T
#       Drop an entire group across *all* timesteps.  The group is chosen
#       uniformly from ``allowed_groups`` (the full-T-allowed groups).
#
#   random_channel_time_mask
#       Drop a random set of channels (count sampled uniformly from
#       ``num_channels``) across a contiguous span sampled from
#       ``span_days``.  Channels are picked irrespective of group.
#
# Span lengths are clamped to the sequence length and masking always
# respects padding (never masks invalid timesteps).  The per-group tables
# above (``GROUP_SPAN_DAYS``, ``FULL_T_GROUP_MASK_ALLOWED``,
# ``SPAN_ONLY_GROUP_MASK``) are the source of truth referenced here.
MASK_POLICY_V1: dict = {
    "within_group_single_var": {
        "prob": 0.5,
        # "all_T_or_long_span": full sequence (full-T-allowed groups only)
        # or a group-specific long span.
        "all_T_prob": 0.25,
        # Candidate groups for single-variable masking: only multi-variable
        # groups (a single-var "subset" of a 1-var group is the whole group).
        "groups": sorted(WITHIN_GROUP_SINGLE_VAR_GROUPS),
        # Of those candidates, the ones that may also be masked across all T.
        "full_t_allowed": sorted(
            FULL_T_GROUP_MASK_ALLOWED & WITHIN_GROUP_SINGLE_VAR_GROUPS
        ),
        # Group-specific long-span ranges for the span branch.
        "long_span_days": dict(WITHIN_GROUP_LONG_SPAN_DAYS),
    },
    "whole_group_span": {
        "prob": 0.4,
        "span_days": dict(GROUP_SPAN_DAYS),
    },
    "whole_group_all_T": {
        "prob": 0.05,
        "allowed_groups": sorted(FULL_T_GROUP_MASK_ALLOWED),
    },
    "random_channel_time_mask": {
        "prob": 0.10,
        "span_days": (1, 7),
        "num_channels": (1, 4),
    },
}

# Fixed iteration order over strategies so multinomial indices are stable.
_POLICY_ORDER: list[str] = [
    "within_group_single_var",
    "whole_group_span",
    "whole_group_all_T",
    "random_channel_time_mask",
]


@dataclass
class CorruptionConfig:
    """Configuration for ERA5 input corruption.

    Args:
        num_time_masks: Number of contiguous time spans to mask per sample.
        time_mask_min_len: Minimum span length (in timesteps).
        time_mask_max_len: Maximum span length (in timesteps).
        variable_groups: Mapping from group name to list of band indices.
        mask_policy: Weighted menu of variable-masking strategies (see
            :data:`MASK_POLICY_V1`).  One strategy is sampled per sample.
            Required: :func:`corrupt_era5` raises if it is ``None``.
        group_recon_mode: Per-group reconstruction mode (see
            :data:`GROUP_RECON_MODE`).  Not used by :func:`corrupt_era5`;
            it is carried here as the single source of truth consumed by
            the reconstruction loss to weight per-group target terms.
        min_mask_ratio: Ensure at least this fraction of (T*V) cells are
            masked (repeat time-mask sampling if needed).
    """

    num_time_masks: int = 3
    time_mask_min_len: int = 7
    time_mask_max_len: int = 30
    variable_groups: dict[str, list[int]] = field(
        default_factory=lambda: dict(DEFAULT_VARIABLE_GROUPS)
    )
    mask_policy: dict | None = field(default_factory=lambda: deepcopy(MASK_POLICY_V1))
    group_recon_mode: dict[str, str] = field(
        default_factory=lambda: dict(GROUP_RECON_MODE)
    )
    min_mask_ratio: float = 0.15


def corrupt_era5(
    era5: Tensor,
    ignore_mask: Tensor,
    config: CorruptionConfig,
) -> Tensor:
    """Generate a corruption mask for a batch of ERA5 sequences.

    The mask indicates which ``(batch, timestep, variable)`` positions
    should be treated as corrupted.  The encoder applies a learned
    per-band embedding at masked positions (see ``use_mask_embed``).

    Args:
        era5: ``[B, T, V]`` normalized input (used only for shape/device).
        ignore_mask: ``[B, T]`` bool, True = padded / invalid timestep.
        config: Corruption settings.

    Returns:
        ``mask`` — ``[B, T, V]`` bool (True = corrupted position).
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

    # --- Policy-driven variable masks ---
    if config.mask_policy is None:
        raise ValueError(
            "CorruptionConfig.mask_policy must be set (e.g. to MASK_POLICY_V1); "
            "no masking policy was provided."
        )
    _apply_mask_policy(mask, valid, config.mask_policy, config.variable_groups)

    return mask


# ---------------------------------------------------------------------------
# Masking helpers
# ---------------------------------------------------------------------------


def _randint(lo: int, hi: int, device: torch.device) -> int:
    """Uniform integer in ``[lo, hi]`` (inclusive)."""
    if hi <= lo:
        return int(lo)
    return int(torch.randint(int(lo), int(hi) + 1, (1,), device=device))


def _random_choice(items: list[Any], device: torch.device) -> Any:
    """Uniformly pick one element from a non-empty list."""
    return items[int(torch.randint(0, len(items), (1,), device=device))]


def _as_span(span_days: Any) -> tuple[int, int]:
    """Coerce a ``(min, max)`` day range (tuple/list) into an int 2-tuple."""
    return (int(span_days[0]), int(span_days[1]))


def _random_span(span_days: tuple[int, int], t: int, device: torch.device) -> Tensor:
    """Return a ``[T]`` bool mask for a random contiguous span.

    ``span_days`` is a ``(min, max)`` length range in days (== timesteps
    for ERA5L_DAY_10); the length is clamped to the sequence length.
    """
    length = min(_randint(span_days[0], span_days[1], device), t)
    max_start = max(t - length, 0)
    start = _randint(0, max_start, device)
    seg = torch.zeros(t, dtype=torch.bool, device=device)
    seg[start : start + length] = True
    return seg


def _apply_mask_policy(
    mask: Tensor,
    valid: Tensor,
    policy: dict,
    variable_groups: dict[str, list[int]],
) -> None:
    """Sample and apply one masking strategy per sample (in place).

    Args:
        mask: ``[B, T, V]`` bool mask to update in place.
        valid: ``[B, T]`` bool, True = real (non-padded) timestep.
        policy: Mask policy dict (see :data:`MASK_POLICY_V1`).
        variable_groups: Group name -> band-index mapping.
    """
    b, t, v = mask.shape
    device = mask.device
    group_names = list(variable_groups.keys())
    if not group_names:
        return

    strat_names = [k for k in _POLICY_ORDER if k in policy]
    if not strat_names:
        return
    probs = torch.tensor(
        [float(policy[k].get("prob", 0.0)) for k in strat_names],
        dtype=torch.float,
        device=device,
    )
    if probs.sum() <= 0:
        return
    probs = probs / probs.sum()
    choices = torch.multinomial(probs, b, replacement=True)

    for i in range(b):
        valid_i = valid[i]  # [T]
        strat = strat_names[int(choices[i])]
        spec = policy[strat]

        if strat == "within_group_single_var":
            group = _random_choice(spec.get("groups") or group_names, device)
            bi = _random_choice(variable_groups[group], device)
            full_t_allowed = set(spec.get("full_t_allowed", []))
            long_span_days = spec.get("long_span_days", {})
            all_t_prob = float(spec.get("all_T_prob", 0.5))
            if (
                group in full_t_allowed
                and float(torch.rand(1, device=device)) < all_t_prob
            ):
                # Full-record drop, only for full-T-allowed groups.
                mask[i, :, bi] |= valid_i
            else:
                # Span drop with a group-specific long-span range.
                span_days = long_span_days.get(group, (30, 180))
                seg = _random_span(_as_span(span_days), t, device)
                mask[i, :, bi] |= seg & valid_i

        elif strat == "whole_group_span":
            span_days = spec["span_days"]
            # Only consider groups that both have a span range and exist.
            candidates = [g for g in span_days if g in variable_groups]
            if not candidates:
                continue
            group = _random_choice(candidates, device)
            seg = _random_span(_as_span(span_days[group]), t, device) & valid_i
            for bi in variable_groups[group]:
                mask[i, :, bi] |= seg

        elif strat == "whole_group_all_T":
            candidates = [
                g for g in spec.get("allowed_groups", []) if g in variable_groups
            ]
            if not candidates:
                continue
            group = _random_choice(candidates, device)
            for bi in variable_groups[group]:
                mask[i, :, bi] |= valid_i

        elif strat == "random_channel_time_mask":
            nc_range = spec.get("num_channels", (1, max(1, v // 4)))
            nc = min(_randint(nc_range[0], nc_range[1], device), v)
            channels = torch.randperm(v, device=device)[:nc]
            seg = _random_span(_as_span(spec["span_days"]), t, device) & valid_i
            for bi in channels.tolist():
                mask[i, :, bi] |= seg
