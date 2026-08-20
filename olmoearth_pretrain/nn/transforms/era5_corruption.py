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
masking; the causal buffer ``[:target_start]`` is never corrupted.

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
    """

    num_spans: tuple[int, int] = (1, 5)
    span_days: tuple[int, int] = (7, 60)
    num_variables: tuple[int, int] = (1, 14)


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
    """Halo-corrected contiguous-span masking (no-leak)."""
    if len(band_supports) != n_bands:
        raise ValueError(
            f"band_supports has {len(band_supports)} entries, expected "
            f"n_bands={n_bands}"
        )
    band4d = torch.zeros(b, t, v, n_bands, dtype=torch.bool, device=device)
    raw_loss_mask = torch.zeros(b, t, v, dtype=torch.bool, device=device)
    window = t - target_start
    if window <= 0:
        return Era5CorruptionMasks(
            band_mask=band4d.reshape(b, t, v * n_bands),
            raw_loss_mask=raw_loss_mask,
        )

    span_lo, span_hi = int(policy.span_days[0]), int(policy.span_days[1])
    nspan_lo, nspan_hi = int(policy.num_spans[0]), int(policy.num_spans[1])
    nvar_lo, nvar_hi = int(policy.num_variables[0]), int(policy.num_variables[1])
    nvar_hi = min(nvar_hi, v)

    for i in range(b):
        n_spans = _randint(nspan_lo, nspan_hi, device)
        for _ in range(n_spans):
            length = min(_randint(span_lo, span_hi, device), window)
            max_start = t - length
            start = _randint(target_start, max_start, device)
            end = start + length

            n_vars = _randint(nvar_lo, nvar_hi, device)
            var_idx = torch.randperm(v, device=device)[:n_vars]

            # Supervise only the genuinely-hidden raw span days.
            raw_loss_mask[i, start:end][:, var_idx] = True

            # Hide each band over the span plus its causal right halo.
            for s, support in enumerate(band_supports):
                halo_end = min(end + int(support) - 1, t)
                band4d[i, start:halo_end][:, var_idx, s] = True

    return Era5CorruptionMasks(
        band_mask=band4d.reshape(b, t, v * n_bands),
        raw_loss_mask=raw_loss_mask,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _randint(lo: int, hi: int, device: torch.device) -> int:
    """Uniform integer in ``[lo, hi]`` (inclusive)."""
    if hi <= lo:
        return int(lo)
    return int(torch.randint(int(lo), int(hi) + 1, (1,), device=device))
