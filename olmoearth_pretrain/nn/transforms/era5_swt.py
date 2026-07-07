"""Undecimated (stationary) wavelet transform and multiscale loss for ERA5.

Implements a dependency-free differentiable SWT along the time axis using
dilated depthwise ``F.conv1d``.  The transform is non-decimated (à trous):
each level ``j`` dilates the filters by ``2^j`` so every coefficient band
has the same length ``T`` as the input — no downsampling, no alignment
headaches.

The companion :func:`multiscale_swt_loss` computes band-normalized Huber
losses between the wavelet coefficients of a prediction and target,
weighted per level.
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
from torch import Tensor, nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Wavelet filter banks (hardcoded, no external dependency)
# ---------------------------------------------------------------------------

# Daubechies-2 (db2) decomposition filters
_DB2_LO: list[float] = [
    -0.12940952255092145,
    0.22414386804185735,
    0.836516303737469,
    0.48296291314469025,
]
_DB2_HI: list[float] = [
    -0.48296291314469025,
    0.836516303737469,
    -0.22414386804185735,
    -0.12940952255092145,
]

# Haar (db1)
_HAAR_LO: list[float] = [0.7071067811865476, 0.7071067811865476]
_HAAR_HI: list[float] = [-0.7071067811865476, 0.7071067811865476]

_FILTER_BANKS: dict[str, tuple[list[float], list[float]]] = {
    "db2": (_DB2_LO, _DB2_HI),
    "haar": (_HAAR_LO, _HAAR_HI),
    "db1": (_HAAR_LO, _HAAR_HI),
}


def _get_filters(name: str) -> tuple[list[float], list[float]]:
    key = name.lower()
    if key not in _FILTER_BANKS:
        raise ValueError(
            f"Unknown wavelet {name!r}; available: {sorted(_FILTER_BANKS)}"
        )
    return _FILTER_BANKS[key]


# ---------------------------------------------------------------------------
# SWT module
# ---------------------------------------------------------------------------


class StationaryWaveletTransform1d(nn.Module):
    """Causal undecimated (stationary) wavelet transform along the time axis.

    Input shape ``[B, V, T]`` (channels-first, matching conv1d).
    Output: list of ``(approx, detail)`` tuples per level, each ``[B, V, T']``
    where ``T' = T - target_start`` when cropping is active.

    Uses causal (left-only) zero-padding. When a ``target_start`` buffer is provided
    to :meth:`forward`, the first ``target_start`` coefficients are discarded
    so that every returned coefficient is free of boundary artifacts.
    """

    def __init__(
        self,
        num_channels: int,
        max_levels: int = 6,
        wavelet: str = "haar",
    ) -> None:
        """Initialize the stationary wavelet transform filters."""
        super().__init__()
        lo, hi = _get_filters(wavelet)
        self.filter_len = len(lo)
        self.max_levels = max_levels

        # Depthwise filters: [V, 1, K] repeated for groups=V conv
        lo_t = torch.tensor(lo, dtype=torch.float32).flip(0)
        hi_t = torch.tensor(hi, dtype=torch.float32).flip(0)
        # Shape [num_channels, 1, K]
        lo_w = lo_t.unsqueeze(0).unsqueeze(0).expand(num_channels, -1, -1).clone()
        hi_w = hi_t.unsqueeze(0).unsqueeze(0).expand(num_channels, -1, -1).clone()
        self.register_buffer("lo_filter", lo_w)
        self.register_buffer("hi_filter", hi_w)
        self.num_channels = num_channels

    def forward(
        self,
        x: Tensor,
        levels: list[int] | None = None,
        target_start: int = 83,
    ) -> list[tuple[Tensor, Tensor]]:
        """Compute the causal undecimated SWT.

        Args:
            x: ``[B, V, T]`` input signal.
            levels: Which decomposition levels to return (0-indexed).
                ``None`` returns all ``max_levels`` levels.
            target_start: If > 0, crop the first ``target_start`` timesteps
                from every returned band so that only the target window
                (free of boundary effects) is returned.

        Returns:
            List of ``(approx, detail)`` pairs, one per requested level.
            Each tensor has shape ``[B, V, T']`` where
            ``T' = T - target_start``.
        """
        if levels is None:
            levels = list(range(self.max_levels))
        results: list[tuple[Tensor, Tensor]] = []
        current = x
        for j in range(max(levels) + 1):
            dilation = 2**j
            pad = dilation * (self.filter_len - 1)
            padded = F.pad(current, (pad, 0), mode="constant", value=0.0)
            approx = F.conv1d(
                padded, self.lo_filter, groups=self.num_channels, dilation=dilation
            )
            detail = F.conv1d(
                padded, self.hi_filter, groups=self.num_channels, dilation=dilation
            )
            if j in levels:
                if target_start > 0:
                    results.append(
                        (approx[:, :, target_start:], detail[:, :, target_start:])
                    )
                else:
                    results.append((approx, detail))
            current = approx
        return results


# ---------------------------------------------------------------------------
# Band stacking helper (shared by the encoder input adapter)
# ---------------------------------------------------------------------------


def swt_bands_to_channels(
    bands: list[tuple[Tensor, Tensor]],
    include_approx: bool = True,
) -> Tensor:
    """Stack SWT bands into a var-major channel tensor.

    Turns the per-level ``(approx, detail)`` list returned by
    :meth:`StationaryWaveletTransform1d.forward` into a single dense tensor
    suitable as encoder input.

    Args:
        bands: List of ``(approx, detail)`` per level, each ``[B, V, T]``.
        include_approx: If True, append the *deepest* level's approximation
            band after the detail bands (making the representation complete).

    Returns:
        ``[B, T, V * n_bands]`` where ``n_bands = len(bands) + include_approx``.
        Channels are **var-major**: ``c = v * n_bands + s`` with scale order
        ``[detail_0, ..., detail_{L-1}, (approx_deepest)]``, so a
        ``view(B, T, V, n_bands)`` recovers per-variable scale groups.
    """
    if not bands:
        raise ValueError("swt_bands_to_channels requires at least one SWT level")
    band_list = [detail for _, detail in bands]  # detail_0 .. detail_{L-1}
    if include_approx:
        band_list.append(bands[-1][0])  # deepest-level approximation
    # Each band is [B, V, T]; stack along a new scale axis -> [B, n_bands, V, T].
    stacked = torch.stack(band_list, dim=1)
    b, n_bands, v, t = stacked.shape
    # var-major flatten: [B, V, n_bands, T] -> [B, V*n_bands, T] -> [B, T, C].
    stacked = stacked.permute(0, 2, 1, 3).reshape(b, v * n_bands, t)
    return stacked.transpose(1, 2).contiguous()


# ---------------------------------------------------------------------------
# Multiscale SWT loss
# ---------------------------------------------------------------------------


def multiscale_swt_loss(
    x_hat: Tensor,
    x: Tensor,
    swt: StationaryWaveletTransform1d,
    levels: list[int],
    huber_delta: float = 1.0,
    mask: Tensor | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Band-normalized multiscale Huber loss in the wavelet domain.

    Args:
        x_hat: ``[B, T, V]`` prediction.
        x: ``[B, T, V]`` target.
        swt: Pre-built :class:`StationaryWaveletTransform1d`.
        levels: Which SWT levels to include.
        huber_delta: Delta for :func:`F.huber_loss`.
        mask: Optional ``[B, T, V]`` bool mask (True = masked / corrupted).
            When provided, loss is computed only over masked positions.

    Returns:
        ``(total_swt_loss, per_level_metrics)``
    """
    # SWT expects [B, V, T]
    pred_vt = x_hat.transpose(1, 2)
    targ_vt = x.transpose(1, 2)

    pred_bands = swt(pred_vt, levels=levels)
    targ_bands = swt(targ_vt, levels=levels)

    if mask is not None:
        mask_vt = mask.transpose(1, 2)  # [B, V, T]

    total = torch.zeros((), device=x.device, dtype=x.dtype)
    metrics: dict[str, Tensor] = {}

    deepest_idx = len(levels) - 1

    for i, j in enumerate(levels):
        a_pred, d_pred = pred_bands[i]
        a_targ, d_targ = targ_bands[i]

        # --- Detail band loss (all levels) ---
        with torch.no_grad():
            std = d_targ.std(dim=(0, 2)).clamp(min=1e-6)  # [V]
        d_pred_n = d_pred / std.unsqueeze(0).unsqueeze(2)
        d_targ_n = d_targ / std.unsqueeze(0).unsqueeze(2)

        if mask is not None:
            flat_pred = d_pred_n[mask_vt]
            flat_targ = d_targ_n[mask_vt]
            if flat_pred.numel() == 0:
                level_loss = torch.zeros((), device=x.device, dtype=x.dtype)
            else:
                level_loss = F.huber_loss(
                    flat_pred, flat_targ, reduction="mean", delta=huber_delta
                )
        else:
            level_loss = F.huber_loss(
                d_pred_n, d_targ_n, reduction="mean", delta=huber_delta
            )

        total = total + level_loss
        metrics[f"swt_level_{j}_loss"] = level_loss.detach()

        # --- Deepest-level approximation loss (low-frequency residual) ---
        if i == deepest_idx:
            with torch.no_grad():
                a_std = a_targ.std(dim=(0, 2)).clamp(min=1e-6)
            a_pred_n = a_pred / a_std.unsqueeze(0).unsqueeze(2)
            a_targ_n = a_targ / a_std.unsqueeze(0).unsqueeze(2)

            if mask is not None:
                flat_a_pred = a_pred_n[mask_vt]
                flat_a_targ = a_targ_n[mask_vt]
                if flat_a_pred.numel() == 0:
                    approx_loss = torch.zeros((), device=x.device, dtype=x.dtype)
                else:
                    approx_loss = F.huber_loss(
                        flat_a_pred, flat_a_targ, reduction="mean", delta=huber_delta
                    )
            else:
                approx_loss = F.huber_loss(
                    a_pred_n, a_targ_n, reduction="mean", delta=huber_delta
                )

            total = total + approx_loss
            metrics[f"swt_level_{j}_approx_loss"] = approx_loss.detach()

    return total, metrics
