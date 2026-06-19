"""Input corruption strategies for ERA5 reconstruction (objective B).

Masking operates on normalized ERA5 input ``[B, T, V]`` via a two-stage
:class:`MaskPolicy`:

* **Stage 1 — Temporal interpolation** (:class:`TemporalInterpolationStrategy`):
  mask short contiguous time spans for eligible groups, forcing the
  encoder to interpolate across time.  Groups whose signals are too
  noisy or event-driven for temporal infilling (e.g. wind, hydro_flux)
  are excluded.

* **Stage 2 — Cross-variable reconstruction**: a weighted menu of
  strategies that, per sample, drops specific variables or groups to
  force reconstruction from cross-variable context.

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

# TODO: for ablations
# instead of reconstructing raw signals then swt transform to band signals,
# we can decode bands directly, then reconstruct raw/time signal from inverse swt
# This forces the reconstruction to pass through the multiscale representation

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
GROUP_SPAN_DAYS: dict[str, list[int]] = {
    "thermo": [1, 7],
    "wind": [1, 3],
    "shortwave_radiation": [3, 30],
    "longwave_radiation": [3, 30],
    "soil_moisture": [7, 90],
    "evaporation": [7, 30],
    "hydro_flux": [1, 3],
    "pressure": [3, 30],
}

# Long-span length range (in days) used by the single-variable masking
# strategy.  Group-specific because a long in-paint is only fair when the
# remaining channels of the group carry enough signal about the dropped one.
# A global (30, 180) is fine for highly redundant pairs (ssr/ssrd, swvl1/
# swvl2, e/pev) but far too hard for weakly mutually-predictive pairs such
# as tp from ro or u10 from v10, so those get shorter windows.
WITHIN_GROUP_LONG_SPAN_DAYS: dict[str, list[int]] = {
    "thermo": [7, 60],
    "wind": [1, 3],
    "shortwave_radiation": [30, 180],
    "soil_moisture": [30, 180],
    "evaporation": [14, 90],
    "hydro_flux": [1, 7],
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
    long_span_days: dict[str, list[int]] = field(
        default_factory=lambda: {
            k: list(v) for k, v in WITHIN_GROUP_LONG_SPAN_DAYS.items()
        }
    )


@dataclass
class WholeGroupSpanStrategy:
    """Drop an entire group across a contiguous span.

    The span length (in days) is sampled from the per-group ``span_days``
    range.  The group is chosen uniformly from groups that have both a
    span range and exist in the variable groups.
    """

    prob: float = 0.4
    span_days: dict[str, list[int]] = field(
        default_factory=lambda: {k: list(v) for k, v in GROUP_SPAN_DAYS.items()}
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
class TemporalInterpolationStrategy:
    """Stage 1: mask short time spans for eligible groups.

    Groups like wind and hydro_flux are excluded by default because their
    daily signals are too noisy or event-driven for meaningful temporal
    infilling.
    """

    num_masks: int = 3
    span_days: list[int] = field(default_factory=lambda: [1, 1])
    excluded_groups: list[str] = field(default_factory=lambda: ["wind", "hydro_flux"])


@dataclass
class MaskPolicy:
    """Two-stage masking policy with per-stage probabilities.

    **Stage 1** — :attr:`temporal_interpolation`

    **Stage 2** — cross-variable reconstruction: one strategy is drawn per
    sample (according to ``prob``).
    """

    temporal_interpolation_prob: float = 1.0
    cross_variable_prob: float = 1.0

    temporal_interpolation: TemporalInterpolationStrategy = field(
        default_factory=TemporalInterpolationStrategy
    )
    within_group_single_var: WithinGroupSingleVarStrategy = field(
        default_factory=WithinGroupSingleVarStrategy
    )
    whole_group_span: WholeGroupSpanStrategy = field(
        default_factory=WholeGroupSpanStrategy
    )
    whole_group_all_T: WholeGroupAllTStrategy = field(
        default_factory=WholeGroupAllTStrategy
    )

    def strategies(self) -> list[tuple[str, Any]]:
        """Return stage 2 ``(name, strategy)`` pairs in fixed order."""
        return [
            ("within_group_single_var", self.within_group_single_var),
            ("whole_group_span", self.whole_group_span),
            ("whole_group_all_T", self.whole_group_all_T),
        ]


@dataclass
class NaiveMaskPolicy(MaskPolicy):
    """Naive baseline: random contiguous spans x random channel subsets."""

    max_num_masks: int = 5
    span_days: tuple[int, int] = (1, 30)
    num_channels: tuple[int, int] = (1, 7)


def corrupt_era5(
    era5: Tensor,
    ignore_mask: Tensor,
    policy: MaskPolicy,
    variable_groups: dict[str, list[int]],
    target_start: int,
) -> Tensor:
    """Generate a corruption mask for a batch of ERA5 sequences.

    Dispatches to the appropriate masking logic based on the policy type:

    * :class:`MaskPolicy` — two-stage physics-aware masking.
    * :class:`NaiveMaskPolicy` — simple random span x channel baseline.

    Only timesteps at index ``target_start`` and beyond are eligible for
    masking; the buffer region ``[:target_start]`` is never corrupted.

    Args:
        era5: ``[B, T, V]`` normalized input (used only for shape/device).
        ignore_mask: ``[B, T]`` bool, True = padded / invalid timestep.
        policy: Masking policy (:class:`MaskPolicy` or :class:`NaiveMaskPolicy`).
        variable_groups: Group name -> band-index mapping.
        target_start: First index of the target window.  Indices before this
            are treated as a causal buffer and are never masked.

    Returns:
        ``mask`` — ``[B, T, V]`` bool (True = corrupted position).
    """
    if isinstance(policy, NaiveMaskPolicy):
        return _corrupt_naive(era5, ignore_mask, policy, target_start)
    return _corrupt_two_stage(era5, ignore_mask, policy, variable_groups, target_start)


def _corrupt_naive(
    era5: Tensor,
    ignore_mask: Tensor,
    policy: NaiveMaskPolicy,
    target_start: int,
) -> Tensor:
    """Naive masking: random spans x random channels, repeated ``num_masks`` times."""
    b, t, v = era5.shape
    device = era5.device
    mask = torch.zeros(b, t, v, dtype=torch.bool, device=device)
    valid = ~ignore_mask  # [B, T]
    span_range = _as_span(policy.span_days)
    t_win = t - target_start

    for i in range(b):
        n_masks = _randint(1, policy.max_num_masks, device)
        for _ in range(n_masks):
            seg = _random_span(span_range, t_win, device)  # [t_win]
            seg = seg & valid[i, target_start:]
            nc = _randint(
                policy.num_channels[0], min(policy.num_channels[1], v), device
            )
            channels = torch.randperm(v, device=device)[:nc]
            mask[i, target_start:, channels] |= seg.unsqueeze(1)

    return mask


def _corrupt_two_stage(
    era5: Tensor,
    ignore_mask: Tensor,
    policy: MaskPolicy,
    variable_groups: dict[str, list[int]],
    target_start: int,
) -> Tensor:
    """Two-stage physics-aware masking (temporal interpolation + cross-variable).

    All masking is restricted to the target window ``[target_start:]``.
    """
    b, t, v = era5.shape
    device = era5.device
    mask = torch.zeros(b, t, v, dtype=torch.bool, device=device)
    valid = ~ignore_mask  # [B, T]
    t_win = t - target_start

    # Per-sample coin flips for each stage
    do_stage1 = torch.rand(b, device=device) < policy.temporal_interpolation_prob
    do_stage2 = torch.rand(b, device=device) < policy.cross_variable_prob

    # --- Stage 1: Temporal interpolation (target window only) ---
    ti = policy.temporal_interpolation
    if ti.num_masks > 0 and do_stage1.any():
        excluded = set(ti.excluded_groups)
        eligible = torch.zeros(v, dtype=torch.bool, device=device)
        for gname, gidx in variable_groups.items():
            if gname not in excluded:
                eligible[gidx] = True

        if eligible.any():
            lo, hi = _as_span(ti.span_days)
            for _ in range(ti.num_masks):
                lengths = torch.randint(lo, hi + 1, (b,), device=device)
                max_start = (t_win - lengths).clamp(min=0)
                starts = (torch.rand(b, device=device) * (max_start.float() + 1)).long()
                idx = torch.arange(t_win, device=device).unsqueeze(0)
                span = (idx >= starts.unsqueeze(1)) & (
                    idx < (starts + lengths).unsqueeze(1)
                )
                span = span & valid[:, target_start:] & do_stage1.unsqueeze(1)
                mask[:, target_start:, :] = mask[:, target_start:, :] | (
                    span.unsqueeze(-1) & eligible[None, None, :]
                )

    # --- Stage 2: Cross-variable reconstruction (target window only) ---
    _apply_stage2_mask_policy(
        mask, valid, policy, variable_groups, do_stage2, target_start
    )

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


def _apply_stage2_mask_policy(
    mask: Tensor,
    valid: Tensor,
    policy: MaskPolicy,
    variable_groups: dict[str, list[int]],
    do_stage2: Tensor,
    target_start: int,
) -> None:
    """Sample and apply one masking strategy per sample (in place).

    All masking is restricted to the target window ``[target_start:]``.

    Args:
        mask: ``[B, T, V]`` bool mask to update in place.
        valid: ``[B, T]`` bool, True = real (non-padded) timestep.
        policy: :class:`MaskPolicy` instance.
        variable_groups: Group name -> band-index mapping.
        do_stage2: ``[B]`` bool, True = apply stage 2 to this sample.
        target_start: First index of the target window.
    """
    b, t, v = mask.shape
    device = mask.device
    t_win = t - target_start
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
        if not do_stage2[i]:
            continue
        valid_win = valid[i, target_start:]  # [t_win]
        strat_name, strat = strategies[int(choices[i])]

        if isinstance(strat, WithinGroupSingleVarStrategy):
            group = _random_choice(strat.groups or group_names, device)
            bi = _random_choice(variable_groups[group], device)
            full_t_allowed = set(strat.full_t_allowed)
            if (
                group in full_t_allowed
                and float(torch.rand(1, device=device)) < strat.all_T_prob
            ):
                mask[i, target_start:, bi] |= valid_win
            else:
                span_days = strat.long_span_days.get(group, [30, 180])
                seg = _random_span(_as_span(span_days), t_win, device)
                mask[i, target_start:, bi] |= seg & valid_win

        elif isinstance(strat, WholeGroupSpanStrategy):
            candidates = [g for g in strat.span_days if g in variable_groups]
            if not candidates:
                continue
            group = _random_choice(candidates, device)
            seg = (
                _random_span(_as_span(strat.span_days[group]), t_win, device)
                & valid_win
            )
            for bi in variable_groups[group]:
                mask[i, target_start:, bi] |= seg

        elif isinstance(strat, WholeGroupAllTStrategy):
            candidates = [g for g in strat.allowed_groups if g in variable_groups]
            if not candidates:
                continue
            group = _random_choice(candidates, device)
            for bi in variable_groups[group]:
                mask[i, target_start:, bi] |= valid_win
