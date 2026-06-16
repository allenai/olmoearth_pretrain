"""Input corruption strategies for ERA5 reconstruction (objective B).

Masking operates on normalized ERA5 input ``[B, T, V]`` and combines:

* **Short time masks** — contiguous spans of timesteps where *all* V
  variables are masked (forces the encoder to interpolate across time).
* **Policy-driven variable masks** — a configurable :class:`MaskPolicy`
  that, per sample, samples one masking strategy from a weighted menu.
  Strategies range from dropping a single variable within a group to
  dropping a whole physical group across a span of days or the full
  sequence.

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
# source of truth that the :class:`MaskPolicy` defaults are assembled from.
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

# Groups where a *single variable* may be masked across the entire sequence
# (all T) while the remaining group member(s) stay visible.  This is safe for
# highly redundant pairs (ssr/ssrd, swvl1/swvl2, e/pev) because the visible
# partner carries strong signal about the hidden one.
SINGLE_VAR_ALL_T_ALLOWED: set[str] = {
    "shortwave_radiation",
    "soil_moisture",
    "evaporation",
}

# Groups where the *entire* group may be masked across all T.  Much riskier
# than single-var masking because no within-group signal remains — the model
# must reconstruct purely from cross-group context.  Start empty; consider
# adding "soil_moisture" later as a low-weight robustness signal.
WHOLE_GROUP_ALL_T_ALLOWED: set[str] = set()

GROUPS_WITH_MULT_VARIABLES = {
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
    "wind": (1, 3),
    "shortwave_radiation": (3, 30),
    "longwave_radiation": (3, 30),
    "soil_moisture": (7, 90),
    "evaporation": (7, 30),
    "hydro_flux": (1, 7),
    "pressure": (3, 30),
}

# Long-span length range (in days) used by the single-variable masking
# strategy.  Group-specific because a long in-paint is only fair when the
# remaining channels of the group carry enough signal about the dropped one.
# A global (30, 180) is fine for highly redundant pairs (ssr/ssrd, swvl1/
# swvl2, e/pev) but far too hard for weakly mutually-predictive pairs such
# as tp from ro or u10 from v10, so those get shorter windows.
WITHIN_GROUP_LONG_SPAN_DAYS: dict[str, tuple[int, int]] = {
    "thermo": (7, 60),
    "wind": (1, 3),
    "shortwave_radiation": (30, 180),
    "soil_moisture": (30, 180),
    "evaporation": (14, 90),
    "hydro_flux": (1, 14),
}


# ---------------------------------------------------------------------------
# Mask policy — per-strategy dataclasses
# ---------------------------------------------------------------------------


@dataclass
class WithinGroupSingleVarStrategy:
    """Drop a single variable inside a randomly chosen multi-variable group.

    If the group is in ``full_t_allowed``, the variable is masked across
    *all* timesteps with probability ``all_T_prob``; otherwise it is masked
    across a contiguous span drawn from ``long_span_days``.
    """

    prob: float = 0.5
    all_T_prob: float = 0.25
    groups: list[str] = field(
        default_factory=lambda: sorted(GROUPS_WITH_MULT_VARIABLES)
    )
    full_t_allowed: list[str] = field(
        default_factory=lambda: sorted(
            SINGLE_VAR_ALL_T_ALLOWED & GROUPS_WITH_MULT_VARIABLES
        )
    )
    long_span_days: dict[str, tuple[int, int]] = field(
        default_factory=lambda: dict(WITHIN_GROUP_LONG_SPAN_DAYS)
    )


@dataclass
class WholeGroupSpanStrategy:
    """Drop an entire group across a contiguous span.

    The span length (in days) is sampled from the per-group ``span_days``
    range.  The group is chosen uniformly from groups that have both a
    span range and exist in the variable groups.
    """

    prob: float = 0.4
    span_days: dict[str, tuple[int, int]] = field(
        default_factory=lambda: dict(GROUP_SPAN_DAYS)
    )


@dataclass
class WholeGroupAllTStrategy:
    """Drop an entire group across *all* timesteps.

    Much riskier than single-var masking — no within-group signal remains.
    The group is chosen uniformly from ``allowed_groups``.  Disabled by
    default (prob=0.0, empty allowed list).
    """

    prob: float = 0.0
    allowed_groups: list[str] = field(
        default_factory=lambda: sorted(WHOLE_GROUP_ALL_T_ALLOWED)
    )


@dataclass
class RandomChannelTimeMaskStrategy:
    """Drop a random set of channels across a contiguous span.

    Channel count is sampled uniformly from ``num_channels``; channels are
    picked irrespective of group boundaries.
    """

    prob: float = 0.10
    span_days: tuple[int, int] = (1, 7)
    num_channels: tuple[int, int] = (1, 4)


@dataclass
class MaskPolicy:
    """Weighted menu of variable-masking strategies.

    For every sample in the batch, exactly one strategy is drawn (according
    to its ``prob``).  Probabilities are normalized so they need not sum
    to 1.  Strategies with ``prob=0`` are skipped.
    """

    within_group_single_var: WithinGroupSingleVarStrategy = field(
        default_factory=WithinGroupSingleVarStrategy
    )
    whole_group_span: WholeGroupSpanStrategy = field(
        default_factory=WholeGroupSpanStrategy
    )
    whole_group_all_T: WholeGroupAllTStrategy = field(
        default_factory=WholeGroupAllTStrategy
    )
    random_channel_time_mask: RandomChannelTimeMaskStrategy = field(
        default_factory=RandomChannelTimeMaskStrategy
    )

    def strategies(self) -> list[tuple[str, Any]]:
        """Return ``(name, strategy)`` pairs in fixed iteration order."""
        return [
            ("within_group_single_var", self.within_group_single_var),
            ("whole_group_span", self.whole_group_span),
            ("whole_group_all_T", self.whole_group_all_T),
            ("random_channel_time_mask", self.random_channel_time_mask),
        ]


@dataclass
class CorruptionConfig:
    """Configuration for ERA5 input corruption.

    Args:
        num_time_masks: Number of contiguous time spans to mask per sample.
        time_mask_min_len: Minimum span length (in timesteps).
        time_mask_max_len: Maximum span length (in timesteps).
        variable_groups: Mapping from group name to list of band indices.
        mask_policy: Weighted menu of variable-masking strategies.  One
            strategy is sampled per sample.
        min_mask_ratio: Ensure at least this fraction of (T*V) cells are
            masked (repeat time-mask sampling if needed).
    """

    num_time_masks: int = 3
    time_mask_min_len: int = 7
    time_mask_max_len: int = 30
    variable_groups: dict[str, list[int]] = field(
        default_factory=lambda: dict(DEFAULT_VARIABLE_GROUPS)
    )
    mask_policy: MaskPolicy = field(default_factory=MaskPolicy)
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
        idx = torch.arange(t, device=device).unsqueeze(0)  # [1, T]
        span = (idx >= starts.unsqueeze(1)) & (idx < (starts + lengths).unsqueeze(1))
        span = span & valid
        mask = mask | span.unsqueeze(-1).expand_as(mask)

    # --- Policy-driven variable masks ---
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
    policy: MaskPolicy,
    variable_groups: dict[str, list[int]],
) -> None:
    """Sample and apply one masking strategy per sample (in place).

    Args:
        mask: ``[B, T, V]`` bool mask to update in place.
        valid: ``[B, T]`` bool, True = real (non-padded) timestep.
        policy: :class:`MaskPolicy` instance.
        variable_groups: Group name -> band-index mapping.
    """
    b, t, v = mask.shape
    device = mask.device
    group_names = list(variable_groups.keys())
    if not group_names:
        return

    strategies = policy.strategies()
    probs = torch.tensor(
        [s.prob for _, s in strategies],
        dtype=torch.float,
        device=device,
    )
    if probs.sum() <= 0:
        return
    probs = probs / probs.sum()
    choices = torch.multinomial(probs, b, replacement=True)

    for i in range(b):
        valid_i = valid[i]  # [T]
        strat_name, strat = strategies[int(choices[i])]

        if isinstance(strat, WithinGroupSingleVarStrategy):
            group = _random_choice(strat.groups or group_names, device)
            bi = _random_choice(variable_groups[group], device)
            full_t_allowed = set(strat.full_t_allowed)
            if (
                group in full_t_allowed
                and float(torch.rand(1, device=device)) < strat.all_T_prob
            ):
                mask[i, :, bi] |= valid_i
            else:
                span_days = strat.long_span_days.get(group, (30, 180))
                seg = _random_span(_as_span(span_days), t, device)
                mask[i, :, bi] |= seg & valid_i

        elif isinstance(strat, WholeGroupSpanStrategy):
            candidates = [g for g in strat.span_days if g in variable_groups]
            if not candidates:
                continue
            group = _random_choice(candidates, device)
            seg = _random_span(_as_span(strat.span_days[group]), t, device) & valid_i
            for bi in variable_groups[group]:
                mask[i, :, bi] |= seg

        elif isinstance(strat, WholeGroupAllTStrategy):
            candidates = [g for g in strat.allowed_groups if g in variable_groups]
            if not candidates:
                continue
            group = _random_choice(candidates, device)
            for bi in variable_groups[group]:
                mask[i, :, bi] |= valid_i

        elif isinstance(strat, RandomChannelTimeMaskStrategy):
            nc = min(_randint(strat.num_channels[0], strat.num_channels[1], device), v)
            channels = torch.randperm(v, device=device)[:nc]
            seg = _random_span(_as_span(strat.span_days), t, device) & valid_i
            for bi in channels.tolist():
                mask[i, :, bi] |= seg
