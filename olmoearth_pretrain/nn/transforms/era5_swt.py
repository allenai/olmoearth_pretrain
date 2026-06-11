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
    """Undecimated (stationary) wavelet transform along the time axis.

    Input shape ``[B, V, T]`` (channels-first, matching conv1d).
    Output: list of ``(approx, detail)`` tuples per level, each ``[B, V, T]``.
    """

    def __init__(
        self,
        num_channels: int,
        max_levels: int = 4,
        wavelet: str = "db2",
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
    ) -> list[tuple[Tensor, Tensor]]:
        """Compute the undecimated SWT.

        Args:
            x: ``[B, V, T]`` input signal.
            levels: Which decomposition levels to return (0-indexed).
                ``None`` returns all ``max_levels`` levels.

        Returns:
            List of ``(approx, detail)`` pairs, one per requested level.
            Each tensor has shape ``[B, V, T]``.
        """
        if levels is None:
            levels = list(range(self.max_levels))
        results: list[tuple[Tensor, Tensor]] = []
        current = x
        for j in range(max(levels) + 1):
            dilation = 2**j
            pad = dilation * (self.filter_len - 1)
            # Circular padding along time
            padded = F.pad(current, (pad, 0), mode="circular")
            approx = F.conv1d(
                padded, self.lo_filter, groups=self.num_channels, dilation=dilation
            )
            detail = F.conv1d(
                padded, self.hi_filter, groups=self.num_channels, dilation=dilation
            )
            if j in levels:
                results.append((approx, detail))
            current = approx
        return results


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

    for i, j in enumerate(levels):
        _, d_pred = pred_bands[i]
        _, d_targ = targ_bands[i]

        # Band-normalize: divide by per-channel coeff std of the *target*
        with torch.no_grad():
            # std per channel [V] computed over B and T
            std = d_targ.std(dim=(0, 2)).clamp(min=1e-6)  # [V]
        d_pred_n = d_pred / std.unsqueeze(0).unsqueeze(2)
        d_targ_n = d_targ / std.unsqueeze(0).unsqueeze(2)

        if mask is not None:
            # Flatten and select
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

    return total, metrics
