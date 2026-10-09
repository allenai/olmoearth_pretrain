"""Gaussian attention windows: every query predicts where, and how widely, to look.

Each head of each query token predicts a Gaussian over key coordinates -- ``(t, row,
col)`` in the encoder and the Perceiver reads, ``(row, col)`` on the latent grid and
in the decoder: a centre offset from the query's own position and a full precision
matrix (through a lower-triangular factor, so a window can be elongated and tilted in
space and in space-time). The attention logit is

    score_ij = CONTENT_SCALE * cos(q_i, k_j) - 1/2 (p_j - mu_i)^T Lambda_i (p_j - mu_i)

so the softmax weights are the content match times the Gaussian: the window says where
matches are likely and the content picks among the keys inside it. The content term is
bounded (``|.| <= CONTENT_SCALE``) while the Gaussian term is not, so a strong content
match far outside the window cannot win: at Mahalanobis distance ``m`` a key is
down-weighted by ``exp(-m^2 / 2)`` (90x at 3 sigma, 3000x at 4), and beyond
``m = 2 sqrt(CONTENT_SCALE)`` (~6.3) it loses to a key at the centre whatever the
content. The window replaces RoPE: it is the only positional signal in the attention.

**Exact on fused kernels.** The Gaussian term is a quadratic in ``p_j``, i.e. a dot
product between per-query coefficients and per-key monomials of ``p_j`` (6 for
``(row, col)``, 13 for ``(t, row, col)``). Appending them to q and k (with the content
scale folded into q and SDPA run at ``scale=1``) gives the exact logit inside an
ordinary fused SDPA kernel, with no N x N bias tensor and gradients to the window
through autograd. The expansion cancels large terms (``|p|^2 / sigma^2`` for a narrow
window far from the origin), so (1) coordinates are first centred per sample on the
keys' mean (exact: the logit depends on ``p_j - mu_i`` only), and (2) under bf16 each
monomial is split into hi + lo bf16 parts and the dot product taken as
``hi.hi + hi.lo + lo.hi``. Measured: a 2 px window on a 256 px crop is ~0.09 logits
off, against ~43 for a plain bf16 expansion. The error grows with the window's
precision and the crop, but key spacing grows with the crop (bounded token count), so
at the width floor it stays small next to the gap between neighbouring keys: 0.13
logits on a 256 px crop with keys 8 px apart (gap 128), 0.04 on a 22 px crop with
keys 1 px apart (gap 2).

**Time.** Tokens without time (static modalities, which the 3D positions put at
``t = 0``, i.e. 2000-01-01, a date no capture has) skip the temporal part of the
window: a pair where either side has no time uses the spatial block of the precision
only.

**Parametrization.** All learned quantities are dimensionless and zero at init: the
offset is in units of a fixed per-head scale ``sigma0``, and the raw precision is
``(A D^-1)^T (A D^-1)`` with ``D = diag(sigma0)`` and ``A`` lower-triangular with
``exp`` on its diagonal. Zero-initialized, every query starts centred on itself with an
axis-aligned window of size ``sigma0``, and weight decay pulls back toward that prior
rather than toward a unit-size window.

**Width floor.** The window is blurred by a fixed ``sigma_min`` Gaussian (half the
finest token spacing: 0.5 px, 0.5 month): covariance ``Lambda^-1 + S^2``,
``S = diag(sigma_min)``, so no window is sharper than ``sigma_min`` along any axis and
the predictor's pull saturates there. Without it, windows with nothing but position to
go on (static-modality decoder queries) shrank to ~0.0005 px; the nearest key, off the
query's grid, then sat at a Mahalanobis^2 in the millions and the gradient with it
(grad norm 2 -> inf in ~240 steps). ``sigma0`` is geometric across heads
(multi-scale, like ALiBi slopes); in 3D the temporal scale runs the opposite way, so
the most local spatial head is the broadest in time (same place, all dates) and vice
versa.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Fixed temperature of the cosine content term: bounds what content can contribute.
CONTENT_SCALE = 10.0
# Range of the per-head window scales sigma0, in the units of the RoPE coordinates:
# row/col in 10 m pixels (rope_coordinate_scale 1), time in days x
# rope_temporal_coordinate_scale (1/30 in v1.3, i.e. ~months).
SPATIAL_SIGMA0_RANGE = (2.0, 128.0)
TEMPORAL_SIGMA0_RANGE = (1.0, 24.0)
# Narrowest window (see "Width floor" in the module docstring), same units.
SPATIAL_SIGMA_MIN = 0.5
TEMPORAL_SIGMA_MIN = 0.5


def _geometric(lo: float, hi: float, n: int) -> torch.Tensor:
    return torch.exp(torch.linspace(math.log(lo), math.log(hi), n))


class GaussianWindow(nn.Module):
    """Per-query, per-head Gaussian windows over key coordinates.

    Holds the window predictor; :meth:`augment` turns content q/k into the augmented
    q/k whose plain dot product is the full logit (content + Gaussian).
    """

    def __init__(self, dim: int, num_heads: int, spatiotemporal: bool) -> None:
        """Initialize the window predictor.

        Args:
            dim: Width of the query input the windows are predicted from.
            num_heads: Number of attention heads (one window per head).
            spatiotemporal: ``(t, row, col)`` windows if True, ``(row, col)`` if False.
        """
        super().__init__()
        self.num_heads = num_heads
        self.ndim = 3 if spatiotemporal else 2
        n = self.ndim
        # Per head: centre offset (n), log of A's diagonal (n), A's strictly-lower part.
        self.num_params = 2 * n + n * (n - 1) // 2
        self.to_params = nn.Linear(dim, num_heads * self.num_params)
        # Zero init = every window starts at its prior (see the module docstring); the
        # encoder's xavier init must not overwrite it.
        nn.init.zeros_(self.to_params.weight)
        nn.init.zeros_(self.to_params.bias)
        self.to_params._skip_custom_init = True
        spatial = _geometric(*SPATIAL_SIGMA0_RANGE, num_heads)
        sigma0 = torch.stack([spatial, spatial], dim=-1)
        if spatiotemporal:
            temporal = _geometric(*TEMPORAL_SIGMA0_RANGE, num_heads).flip(0)
            sigma0 = torch.cat([temporal[:, None], sigma0], dim=-1)
        self.register_buffer("sigma0", sigma0, persistent=False)
        sigma_min = [TEMPORAL_SIGMA_MIN] * (n - 2) + [SPATIAL_SIGMA_MIN] * 2
        self.register_buffer("sigma_min", torch.tensor(sigma_min), persistent=False)
        rows, cols = torch.tril_indices(n, n, offset=-1)
        self.register_buffer("tril_flat", rows * n + cols, persistent=False)

    def windows(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict each query's window.

        Args:
            x: Query input ``[B, Nq, dim]``.

        Returns:
            offset ``[B, Nq, H, n]`` (coordinate units) and precision
            ``[B, Nq, H, n, n]``, both float32.
        """
        n = self.ndim
        params = self.to_params(x).float()
        params = params.unflatten(-1, (self.num_heads, self.num_params))
        offset = params[..., :n] * self.sigma0
        lower = params.new_zeros(*params.shape[:-1], n * n)
        lower[..., self.tril_flat] = params[..., 2 * n :]
        a = torch.diag_embed(params[..., n : 2 * n].exp()) + lower.unflatten(-1, (n, n))
        w = a / self.sigma0[:, None, :]
        # Width floor: (Lambda^-1 + S^2)^-1 = S^-1 (I - (I + S Lambda S)^-1) S^-1, which
        # stays finite as Lambda grows and saturates at S^-2. Tiny matrices, so float64.
        ws = (w * self.sigma_min).double()
        eye = torch.eye(n, dtype=torch.float64, device=x.device)
        inner = eye - torch.linalg.inv(eye + ws.transpose(-1, -2) @ ws)
        return offset, inner.float() / (self.sigma_min[:, None] * self.sigma_min)

    def features(
        self,
        x: torch.Tensor,
        q_positions: torch.Tensor,
        k_positions: torch.Tensor,
        key_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-query coefficients and per-key monomials of the Gaussian term.

        ``q_features[b, h, i] . k_features[b, j]`` is
        ``-1/2 (p_j - mu_i)^T Lambda_i (p_j - mu_i)`` (temporal part gated, see module
        docstring).

        Args:
            x: Query input ``[B, Nq, dim]``.
            q_positions: ``[B, Nq, n]`` query coordinates.
            k_positions: ``[B, Nk, n]`` key coordinates.
            key_mask: Optional bool ``[B, Nk]``, True for valid keys (sets the centring).

        Returns:
            ``q_features [B, H, Nq, F]`` and ``k_features [B, Nk, F]``, float32, with
            ``F = 6`` for ``(row, col)`` and 13 for ``(t, row, col)``.
        """
        n = self.ndim
        if q_positions.shape[-1] != n or k_positions.shape[-1] != n:
            raise ValueError(
                f"{n}D Gaussian windows need {n} coordinates, got query "
                f"{q_positions.shape[-1]} / key {k_positions.shape[-1]}"
            )
        qp, kp = q_positions.float(), k_positions.float()
        valid = (
            key_mask.float()
            if key_mask is not None
            else torch.ones(kp.shape[:2], device=kp.device)
        )
        if n == 3:
            q_time = (qp[..., 0] != 0).float()
            k_time = (kp[..., 0] != 0).float()
            # Centre time on the keys that have one, space on all valid keys.
            time_weight = valid * k_time
            t_ref = (kp[..., 0] * time_weight).sum(1) / time_weight.sum(1).clamp(min=1)
            s_ref = (kp[..., 1:] * valid[..., None]).sum(1) / valid.sum(1).clamp(min=1)[
                :, None
            ]
            ref = torch.cat([t_ref[:, None], s_ref], dim=-1)[:, None]
        else:
            ref = (kp * valid[..., None]).sum(1, keepdim=True) / valid.sum(1).clamp(
                min=1
            )[:, None, None]
        qp, kp = qp - ref, kp - ref

        offset, lam = self.windows(x)
        mu = qp[:, :, None, :] + offset  # [B, Nq, H, n]
        # Spatial block: the last two axes in both layouts.
        r, c = n - 2, n - 1
        lrr, lcc, lrc = lam[..., r, r], lam[..., c, c], lam[..., r, c]
        mr, mc = mu[..., r], mu[..., c]
        q_feats = [
            -0.5 * lrr,
            -0.5 * lcc,
            -lrc,
            lrr * mr + lrc * mc,
            lcc * mc + lrc * mr,
            -0.5 * (lrr * mr * mr + lcc * mc * mc) - lrc * mr * mc,
        ]
        kr, kc = kp[..., r], kp[..., c]
        k_feats = [kr * kr, kc * kc, kr * kc, kr, kc, torch.ones_like(kr)]
        if n == 3:
            ltt, ltr, ltc = lam[..., 0, 0], lam[..., 0, r], lam[..., 0, c]
            mt = mu[..., 0]
            gate = q_time[..., None]  # [B, Nq, 1] -> broadcasts over heads
            q_feats += [
                gate * f
                for f in (
                    -0.5 * ltt,
                    -ltr,
                    -ltc,
                    ltt * mt + ltr * mr + ltc * mc,
                    ltr * mt,
                    ltc * mt,
                    -0.5 * ltt * mt * mt - ltr * mt * mr - ltc * mt * mc,
                )
            ]
            kt = kp[..., 0]
            k_feats += [
                k_time * f
                for f in (kt * kt, kt * kr, kt * kc, kt, kr, kc, torch.ones_like(kt))
            ]
        q_features = torch.stack(q_feats, dim=-1).transpose(1, 2)  # [B, H, Nq, F]
        k_features = torch.stack(k_feats, dim=-1)  # [B, Nk, F]
        return q_features, k_features

    def augment(
        self,
        x: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        q_positions: torch.Tensor,
        k_positions: torch.Tensor,
        key_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Augment content q/k so that ``q_aug . k_aug`` is the full logit.

        Args:
            x: Query input ``[B, Nq, dim]`` (the windows are predicted from it).
            q: Content queries ``[B, H, Nq, D]``.
            k: Content keys ``[B, H, Nk, D]``.
            q_positions: ``[B, Nq, n]`` query coordinates.
            k_positions: ``[B, Nk, n]`` key coordinates.
            key_mask: Optional bool ``[B, Nk]``, True for valid keys.

        Returns:
            ``q_aug [B, H, Nq, D']`` and ``k_aug [B, H, Nk, D']`` in q's dtype, ``D'`` a
            multiple of 8; run SDPA on them with ``scale=1``.
        """
        q_features, k_features = self.features(x, q_positions, k_positions, key_mask)
        k_features = k_features[:, None].expand(-1, self.num_heads, -1, -1)
        dtype = q.dtype
        q = F.normalize(q, dim=-1) * CONTENT_SCALE
        k = F.normalize(k, dim=-1)
        if dtype == torch.float32:
            q_parts = [q, q_features]
            k_parts = [k, k_features]
        else:
            q_hi = q_features.to(dtype)
            q_lo = (q_features - q_hi.float()).to(dtype)
            k_hi = k_features.to(dtype)
            k_lo = (k_features - k_hi.float()).to(dtype)
            q_parts = [q, q_hi, q_hi, q_lo]
            k_parts = [k, k_hi, k_lo, k_hi]
        q_aug = torch.cat([p.to(dtype) for p in q_parts], dim=-1)
        k_aug = torch.cat([p.to(dtype) for p in k_parts], dim=-1)
        pad = -q_aug.shape[-1] % 8
        return F.pad(q_aug, (0, pad)), F.pad(k_aug, (0, pad))
