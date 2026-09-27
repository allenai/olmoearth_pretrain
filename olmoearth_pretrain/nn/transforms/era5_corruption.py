"""Input corruption strategies for ERA5 reconstruction (objective B).

Masking operates in the **SWT band space** ``[B, T, V * n_bands]`` that the
encoder consumes when ``is_swt_input=True``.  Two policies are provided:

* :class:`SwtNaiveMaskPolicy` — the budget-only baseline.  Every band element
  ``(timestep, variable, swt_band)`` in the target window is masked
  independently with probability ``budget``.

* :class:`SwtHaloSpanMaskPolicy` — halo-corrected span masking (no-leak).  A
  few contiguous day spans are drawn per sample over a random subset of
  variables; each span is expanded across bands with a per-band causal
  right halo so the raw days in the span are *genuinely absent* from every
  scale the encoder sees.

Why halos are necessary
-----------------------
The undecimated (stationary) wavelet transform is ~``n_bands``x overcomplete:
every coefficient is a fixed linear functional of the same raw values.  As a
result, masking scattered band elements — or even all bands of a short span —
leaves the masked coefficients (and the raw values they encode) linearly
recoverable from the visible ones, with no weather prior required.  To hide a
contiguous raw span ``[s, s+L)`` from band ``s`` we must also mask every
coefficient whose causal support reaches back into the span, i.e. extend the
mask ``support_s - 1`` days to the *right* (the transform is causal, so no
left halo is needed).  See
:func:`~olmoearth_pretrain.nn.transforms.era5_swt.swt_band_supports`.

Because the halo positions are still (mostly) recoverable, the reconstruction
loss is supervised only on the raw days inside each span — the
``raw_loss_mask`` returned alongside the band-space ``band_mask``.

Note: ERA5L_DAY_10 has one timestep per day, so span lengths expressed in
*days* map directly onto timesteps.  Everything is applied on-the-fly on the
GPU and only timesteps at index ``target_start`` and beyond are eligible for
masking; ``[:target_start]`` is never corrupted. (The reconstruction
objective's ``mask_buffer`` option passes ``target_start=0`` so the SWT buffer
can be masked too, while losses still start after it.)

:func:`corrupt_era5_swt` returns an :class:`Era5CorruptionMasks` with both the
band-space corruption mask (fed to the encoder's learned ``mask_embed``) and
the raw ``[B, T, V]`` loss mask.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

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

# TODO: the loss-side tables below (DEFAULT_VARIABLE_GROUPS, GROUP_RECON_MODE,
# RECON_MODE_SPEC) are retained unchanged so baseline reconstruction losses are
# identical to prior runs.  They are no longer used by any masking policy and
# should be deleted (and the change A/B-tested) in a follow-up once the group
# recon-mode loss weighting is confirmed unnecessary.

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
# Per-group reconstruction-loss controls (loss side, not masking)
# ---------------------------------------------------------------------------
#
# These tables describe, for each variable group, *how* its reconstruction
# loss is weighted across raw vs. wavelet bands.  They feed the loss weighting
# in the reconstruction objective, not any masking policy.

SWT_DETAIL_LEVELS: list[int] = [0, 1, 2, 3, 4, 5]

RECON_MODE_SPEC: dict[str, dict] = {
    "raw_plus_all_swt": {
        "include_raw": True,
        "swt_detail_levels": [0, 1, 2, 3, 4, 5],
        "include_lowpass": False,
    },
    "raw_plus_no_fast_swt": {
        "include_raw": True,
        "swt_detail_levels": [1, 2, 3, 4, 5],
        "include_lowpass": False,
    },
    "raw_plus_slow_swt": {
        "include_raw": True,
        "swt_detail_levels": [2, 3, 4, 5],
        "include_lowpass": False,
    },
    "lowpass_plus_slow_swt": {
        "include_raw": False,
        "swt_detail_levels": [2, 3, 4, 5],
        "include_lowpass": True,
    },
}

GROUP_RECON_MODE: dict[str, str] = {
    # Exact weather state matters; fast variability is meaningful.
    "thermo": "raw_plus_all_swt",
    # Raw target useful, but no fastest detail band.
    "wind": "raw_plus_no_fast_swt",
    "shortwave_radiation": "raw_plus_no_fast_swt",
    "longwave_radiation": "raw_plus_no_fast_swt",
    "evaporation": "raw_plus_no_fast_swt",
    # Long-memory variables / difficult fluxes.
    "soil_moisture": "raw_plus_slow_swt",
    "hydro_flux": "raw_plus_slow_swt",
    # No pointwise reconstruction; only baseline + slow structure.
    "pressure": "lowpass_plus_slow_swt",
}


# ---------------------------------------------------------------------------
# Masking policies (SWT band space)
# ---------------------------------------------------------------------------


@dataclass
class SwtNaiveMaskPolicy:
    """Budget-only masking for SWT-input reconstruction (baseline).

    Every band element ``(timestep, variable, swt_band)`` in the target
    window is masked independently with probability ``budget``, so masking is
    spread uniformly at random across all three axes with no spans or
    per-group structure.

    The raw ``[B, T, V]`` loss mask is derived by reducing the band-space mask
    over the scale axis: ``"any"`` supervises a raw position where any band is
    masked, ``"all"`` only where every band is masked.
    """

    budget: float = 0.5
    raw_loss_mask_reduce: str = "any"


@dataclass
class SwtHaloSpanMaskPolicy:
    """Halo-corrected contiguous-span masking for SWT-input reconstruction.

    Per sample, ``num_spans`` spans are drawn (count sampled uniformly in the
    inclusive range).  Each span draws a length from ``span_days`` and a random
    subset of variables (size drawn uniformly from ``num_variables``).  Each
    span is expanded across bands with a per-band causal right halo of
    ``support_s - 1`` days so the span's raw days are genuinely hidden from
    every scale.  The loss is supervised only on the raw span days.

    ``placement`` controls where a span of length ``L`` can land in the
    maskable window ``[lo, t)``:

    * ``"inside"`` — start uniform over ``[lo, t - L]``. Days within ``L`` of
      either edge are rarely covered (coverage ramps from ~1/L at the edge).
    * ``"pin"`` — start uniform over ``[lo - L + 1, t - 1]``, then shifted
      inside so the span stays contiguous at full length (flush with the
      edge). Edge days get interior coverage; days just inside each edge get
      up to ~2x for a single span (a mild bump ``L`` days wide).
    """

    num_spans: tuple[int, int] = (1, 5)
    span_days: tuple[int, int] = (7, 60)
    num_variables: tuple[int, int] = (1, 14)
    placement: str = "inside"


MaskPolicy = SwtNaiveMaskPolicy | SwtHaloSpanMaskPolicy


@dataclass
class Era5CorruptionMasks:
    """Pair of masks produced by :func:`corrupt_era5_swt`.

    Attributes:
        band_mask: ``[B, T, V * n_bands]`` bool (True = corrupted band element).
            Fed to the encoder to replace positions with the learned mask
            embedding.
        raw_loss_mask: ``[B, T, V]`` bool (True = genuinely-hidden raw position
            that should be supervised).
    """

    band_mask: Tensor
    raw_loss_mask: Tensor


# ---------------------------------------------------------------------------
# Corruption entry point
# ---------------------------------------------------------------------------


def corrupt_era5_swt(
    b: int,
    t: int,
    v: int,
    n_bands: int,
    band_supports: list[int],
    target_start: int,
    device: torch.device,
    policy: MaskPolicy,
) -> Era5CorruptionMasks:
    """Generate SWT band-space corruption masks for a batch.

    Dispatches on the policy type:

    * :class:`SwtNaiveMaskPolicy` — per-element budget masking.
    * :class:`SwtHaloSpanMaskPolicy` — halo-corrected span masking.

    Args:
        b: Batch size.
        t: Sequence length (timesteps).
        v: Number of raw variables.
        n_bands: Number of SWT bands per variable (channels ``= v * n_bands``).
        band_supports: Per-band causal support in days, in band order
            ``[detail_0, ..., detail_{L-1}, (approx_deepest)]`` (see
            :func:`~olmoearth_pretrain.nn.transforms.era5_swt.swt_band_supports`).
            Only used by the halo span policy.
        target_start: First maskable timestep; ``[:target_start]`` is never
            masked.
        device: Device for the returned tensors.
        policy: Masking policy.

    Returns:
        :class:`Era5CorruptionMasks` with ``band_mask`` ``[B, T, V*n_bands]``
        and ``raw_loss_mask`` ``[B, T, V]``.
    """
    if isinstance(policy, SwtNaiveMaskPolicy):
        return _corrupt_swt_naive(b, t, v, n_bands, target_start, device, policy)
    if isinstance(policy, SwtHaloSpanMaskPolicy):
        return _corrupt_swt_halo_span(
            b, t, v, n_bands, band_supports, target_start, device, policy
        )
    raise TypeError(f"Unsupported mask policy: {type(policy).__name__}")


def _corrupt_swt_naive(
    b: int,
    t: int,
    v: int,
    n_bands: int,
    target_start: int,
    device: torch.device,
    policy: SwtNaiveMaskPolicy,
) -> Era5CorruptionMasks:
    """Per-element budget masking (baseline)."""
    if policy.raw_loss_mask_reduce not in ("any", "all"):
        raise ValueError(
            f"raw_loss_mask_reduce must be 'any' or 'all', got "
            f"{policy.raw_loss_mask_reduce!r}"
        )
    c = v * n_bands
    band_mask = torch.zeros(b, t, c, dtype=torch.bool, device=device)
    if policy.budget > 0.0:
        window = t - target_start
        band_mask[:, target_start:, :] = (
            torch.rand(b, window, c, device=device) < policy.budget
        )
    band4d = band_mask.view(b, t, v, n_bands)
    raw_loss_mask = (
        band4d.all(dim=-1)
        if policy.raw_loss_mask_reduce == "all"
        else band4d.any(dim=-1)
    )
    return Era5CorruptionMasks(band_mask=band_mask, raw_loss_mask=raw_loss_mask)


def _corrupt_swt_halo_span(
    b: int,
    t: int,
    v: int,
    n_bands: int,
    band_supports: list[int],
    target_start: int,
    device: torch.device,
    policy: SwtHaloSpanMaskPolicy,
) -> Era5CorruptionMasks:
    """Halo-corrected contiguous-span masking (no-leak).

    Vectorized over samples and spans: every draw stays on ``device`` (no
    host syncs) and each mask is the union over spans of day-range x
    variable-subset boxes. Samples draw ``n_spans`` of ``max(num_spans)`` span
    slots; the remaining slots are inactive.
    """
    if len(band_supports) != n_bands:
        raise ValueError(
            f"band_supports has {len(band_supports)} entries, expected "
            f"n_bands={n_bands}"
        )
    if policy.placement not in ("inside", "pin"):
        raise ValueError(
            f"placement must be 'inside' or 'pin', got {policy.placement!r}"
        )
    window = t - target_start
    if window <= 0:
        return Era5CorruptionMasks(
            band_mask=torch.zeros(b, t, v * n_bands, dtype=torch.bool, device=device),
            raw_loss_mask=torch.zeros(b, t, v, dtype=torch.bool, device=device),
        )

    span_lo, span_hi = int(policy.span_days[0]), int(policy.span_days[1])
    nspan_lo, nspan_hi = int(policy.num_spans[0]), int(policy.num_spans[1])
    nvar_lo, nvar_hi = int(policy.num_variables[0]), int(policy.num_variables[1])
    nvar_hi = min(nvar_hi, v)
    k = max(nspan_lo, nspan_hi)

    n_spans = _randint_tensor(nspan_lo, nspan_hi, (b, 1), device)
    active = torch.arange(k, device=device) < n_spans  # [B, K]
    length = _randint_tensor(span_lo, span_hi, (b, k), device).clamp(max=window)

    # Start drawn uniformly over [low, high] (inclusive): "inside" keeps the
    # span in the window; "pin" lets it overhang either edge by up to L - 1
    # days, then the clamp below shifts it flush with that edge at full length.
    if policy.placement == "inside":
        low = torch.full_like(length, target_start)
        high = t - length
    else:
        low = target_start - length + 1
        high = torch.full_like(length, t - 1)
    num_starts = high - low + 1
    offset = (torch.rand(b, k, device=device) * num_starts).long()
    start = low + torch.minimum(offset, num_starts - 1)
    start = torch.minimum(start.clamp(min=target_start), t - length)
    end = start + length

    # Uniform random subset of n_vars variables per span: ranks of iid keys
    # form a uniform permutation, and ranks below n_vars pick the subset.
    n_vars = _randint_tensor(nvar_lo, nvar_hi, (b, k, 1), device)
    ranks = torch.rand(b, k, v, device=device).argsort(dim=-1).argsort(dim=-1)
    var_sel = (ranks < n_vars) & active.unsqueeze(-1)  # [B, K, V]

    day = torch.arange(t, device=device)
    after_start = day >= start.unsqueeze(-1)  # [B, K, T]

    def cover(stop: Tensor) -> Tensor:
        """Union over spans of days ``[start, stop)`` x selected variables."""
        in_range = after_start & (day < stop.unsqueeze(-1))  # [B, K, T]
        return (in_range.unsqueeze(-1) & var_sel.unsqueeze(2)).any(dim=1)

    # Supervise only the genuinely-hidden raw span days; hide each band over
    # the span plus its causal right halo (days past t simply do not exist).
    raw_loss_mask = cover(end)
    band4d = torch.stack(
        [cover(end + int(support) - 1) for support in band_supports], dim=-1
    )

    return Era5CorruptionMasks(
        band_mask=band4d.reshape(b, t, v * n_bands),
        raw_loss_mask=raw_loss_mask,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _randint_tensor(
    lo: int, hi: int, shape: tuple[int, ...], device: torch.device
) -> Tensor:
    """Uniform integers in ``[lo, hi]`` (inclusive); constant ``lo`` if ``hi <= lo``."""
    if hi <= lo:
        return torch.full(shape, int(lo), dtype=torch.long, device=device)
    return torch.randint(int(lo), int(hi) + 1, shape, device=device)
