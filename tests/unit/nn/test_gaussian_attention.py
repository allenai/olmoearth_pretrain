"""Gaussian attention windows (``olmoearth_pretrain/nn/gaussian_attention.py``)."""

import math

import pytest
import torch
import torch.nn.functional as F

from olmoearth_pretrain.nn.attention import Attention
from olmoearth_pretrain.nn.gaussian_attention import (
    CONTENT_SCALE,
    SPATIAL_SIGMA0_RANGE,
    TEMPORAL_SIGMA0_RANGE,
    GaussianWindow,
)

B, H, DIM = 2, 4, 32


def _randomize(window: GaussianWindow, std: float = 0.3) -> None:
    """Give the (zero-initialized) window predictor random weights."""
    with torch.no_grad():
        window.to_params.weight.normal_(
            0, std / math.sqrt(window.to_params.in_features)
        )
        window.to_params.bias.normal_(0, std)


def _positions(
    n: int, spatiotemporal: bool, static: tuple[int, ...] = ()
) -> torch.Tensor:
    """Token-like coordinates: rows/cols on a grid in pixels, t in ~months since 2000."""
    rc = torch.randint(0, 16, (B, n, 2)).float() * 4.0
    if not spatiotemporal:
        return rc
    t = 9000 / 30 + torch.randint(0, 12, (B, n, 1)).float()
    t[:, list(static)] = 0.0  # static tokens sit at t = 0
    return torch.cat([t, rc], dim=-1)


def _reference_bias(
    window: GaussianWindow, x: torch.Tensor, qp: torch.Tensor, kp: torch.Tensor
) -> torch.Tensor:
    """-1/2 (p_j - mu_i)^T Lambda_i (p_j - mu_i), computed directly in float64.

    For a pair where either side has no time (t == 0), only the spatial block.
    """
    offset, lam = window.windows(x)
    offset, lam = offset.double(), lam.double()
    mu = qp.double()[:, :, None, :] + offset  # [B, Nq, H, n]
    d = kp.double()[:, None, None, :, :] - mu[:, :, :, None, :]  # [B, Nq, H, Nk, n]
    full = -0.5 * torch.einsum("bihjx,bihxy,bihjy->bhij", d, lam, d)
    if window.ndim == 2:
        return full
    ds, ls = d[..., 1:], lam[..., 1:, 1:]
    spatial = -0.5 * torch.einsum("bihjx,bihxy,bihjy->bhij", ds, ls, ds)
    both_timed = (qp[..., 0] != 0)[:, None, :, None] & (kp[..., 0] != 0)[:, None, None]
    return torch.where(both_timed, full, spatial)


@pytest.mark.parametrize("spatiotemporal", [True, False])
def test_features_are_the_gaussian_logit(spatiotemporal: bool) -> None:
    """q_features . k_features is exactly the (gated) Gaussian log-density."""
    torch.manual_seed(0)
    window = GaussianWindow(DIM, H, spatiotemporal)
    _randomize(window)
    nq, nk = 5, 7
    x = torch.randn(B, nq, DIM)
    qp = _positions(nq, spatiotemporal, static=(1,) if spatiotemporal else ())
    kp = _positions(nk, spatiotemporal, static=(2, 3) if spatiotemporal else ())
    qf, kf = window.features(x, qp, kp, key_mask=None)
    assert qf.shape == (B, H, nq, 13 if spatiotemporal else 6)
    got = torch.einsum("bhif,bjf->bhij", qf.double(), kf.double())
    torch.testing.assert_close(
        got, _reference_bias(window, x, qp, kp), atol=2e-3, rtol=1e-4
    )


def test_zero_init_is_the_prior() -> None:
    """At init every query sits on itself with an axis-aligned sigma0 window.

    sigma0 is geometric over heads; in 3D time runs the opposite way to space.
    """
    window = GaussianWindow(DIM, H, spatiotemporal=True)
    offset, lam = window.windows(torch.randn(B, 3, DIM))
    assert torch.equal(offset, torch.zeros_like(offset))
    expected = torch.diag_embed(window.sigma0**-2).expand_as(lam)
    torch.testing.assert_close(lam, expected)
    spatial, temporal = window.sigma0[:, 1], window.sigma0[:, 0]
    torch.testing.assert_close(spatial[[0, -1]], torch.tensor(SPATIAL_SIGMA0_RANGE))
    torch.testing.assert_close(temporal[[-1, 0]], torch.tensor(TEMPORAL_SIGMA0_RANGE))
    assert torch.equal(window.sigma0[:, 1], window.sigma0[:, 2])


@pytest.mark.parametrize("position_encoding", ["gaussian_3d", "gaussian"])
def test_attention_matches_dense_reference(position_encoding: str) -> None:
    """Attention through augmented q/k on SDPA == softmax(tau cos + bias) v."""
    torch.manual_seed(0)
    spatiotemporal = position_encoding == "gaussian_3d"
    attn = Attention(
        DIM, num_heads=H, qkv_bias=True, position_encoding=position_encoding
    )
    assert attn.gaussian is not None and attn.rope_mixed_freqs is None
    _randomize(attn.gaussian)
    n = 9
    x = torch.randn(B, n, DIM)
    pos = _positions(n, spatiotemporal, static=(4,) if spatiotemporal else ())
    mask = torch.ones(B, n, dtype=torch.bool)
    mask[0, -3:] = False  # padding keys
    out = attn(x, attn_mask=mask, rope_positions=pos)

    q = attn.q(x).unflatten(-1, (H, -1)).transpose(1, 2)
    k = attn.k(x).unflatten(-1, (H, -1)).transpose(1, 2)
    v = attn.v(x).unflatten(-1, (H, -1)).transpose(1, 2)
    content = CONTENT_SCALE * F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).mT
    logits = content.double() + _reference_bias(attn.gaussian, x, pos, pos)
    logits = logits.masked_fill(~mask[:, None, None], float("-inf"))
    ref = (logits.softmax(-1).float() @ v).transpose(1, 2).flatten(2)
    torch.testing.assert_close(out, attn.proj(ref), atol=1e-4, rtol=1e-4)


def test_cross_attention_uses_key_positions() -> None:
    """Cross-attention places the window over the keys' own coordinates."""
    torch.manual_seed(0)
    attn = Attention(
        DIM, num_heads=H, qkv_bias=True, cross_attn=True, position_encoding="gaussian"
    )
    x, y = torch.randn(B, 3, DIM), torch.randn(B, 6, DIM)
    qp, kp = _positions(3, False), _positions(6, False)
    out = attn(x, y=y, rope_positions=qp, rope_positions_y=kp)
    moved = attn(x, y=y, rope_positions=qp, rope_positions_y=kp + 40.0)
    assert not torch.allclose(out, moved)
    # Translating queries and keys together changes nothing (relative windows).
    shifted = attn(x, y=y, rope_positions=qp + 40.0, rope_positions_y=kp + 40.0)
    torch.testing.assert_close(out, shifted, atol=1e-5, rtol=1e-5)


def test_bf16_logits_are_precise_at_real_coordinates() -> None:
    """hi/lo bf16 split keeps the window logits precise at real coordinates.

    Worst case: the narrowest head (sigma0 = 2 px) on a 256 px crop, t in days since
    2000. Measured: hi/lo error ~0.09 logits there (0.009 at 64 px), plain bf16 ~43.
    """
    torch.manual_seed(0)
    window = GaussianWindow(DIM, H, spatiotemporal=True)
    n = 64
    rc = torch.rand(B, n, 2) * 256.0
    t = 9000 / 30 + torch.randint(0, 12, (B, n, 1)).float()
    pos = torch.cat([t, rc], dim=-1)
    x = torch.randn(B, n, DIM)
    q = torch.randn(B, H, n, 64, dtype=torch.bfloat16)
    k = torch.randn(B, H, n, 64, dtype=torch.bfloat16)
    q_aug, k_aug = window.augment(x, q, k, pos, pos, key_mask=None)
    assert q_aug.dtype == torch.bfloat16 and q_aug.shape[-1] == 104
    got = q_aug.float() @ k_aug.float().mT  # bf16 inputs, fp32 accumulation
    content = CONTENT_SCALE * (
        F.normalize(q.float(), dim=-1) @ F.normalize(k.float(), dim=-1).mT
    )
    gauss = _reference_bias(window, x, pos, pos)
    relevant = gauss > -30  # keys a softmax can still see
    err = (got.double() - content.double() - gauss).abs()[relevant]
    assert err.max() < 0.15  # content alone is ~0.015 off in bf16

    qf, kf = window.features(x, pos, pos, key_mask=None)
    single = (qf.bfloat16().float() @ kf[:, None].bfloat16().float().mT).double()
    assert (single - gauss).abs()[relevant].max() > 100 * err.max()


def test_static_keys_ignore_time() -> None:
    """A key without time (t = 0) is scored on space only, however far the date."""
    window = GaussianWindow(DIM, H, spatiotemporal=True)
    x = torch.randn(1, 1, DIM)
    q_pos = torch.tensor([[[300.0, 10.0, 10.0]]])
    k_pos = torch.tensor([[[0.0, 10.0, 10.0], [300.0, 10.0, 10.0]]])
    qf, kf = window.features(x, q_pos, k_pos, key_mask=None)
    bias = torch.einsum("bhif,bjf->bhij", qf, kf)
    torch.testing.assert_close(bias, torch.zeros_like(bias), atol=1e-4, rtol=0)


def test_strong_match_outside_the_window_loses_to_the_centre() -> None:
    """Bounded content cannot pull attention outside the window.

    Beyond Mahalanobis distance 2 sqrt(CONTENT_SCALE) (~6.3), the worst content match
    at the window centre beats a perfect one: here 7 sigma away.
    """
    attn = Attention(DIM, num_heads=1, cross_attn=True, position_encoding="gaussian")
    with torch.no_grad():
        e0 = torch.zeros(DIM, DIM)
        e0[0, 0] = 1.0
        attn.q.weight.copy_(e0)  # content lives in dim 0 ...
        attn.k.weight.copy_(e0)
        attn.v.weight.copy_(torch.eye(DIM))  # ... the value marker in dim 1
        attn.proj.weight.copy_(torch.eye(DIM))
        attn.proj.bias.zero_()
    assert attn.gaussian is not None
    sigma = attn.gaussian.sigma0[0, 0].item()
    query = torch.zeros(1, 1, DIM)
    query[..., 0] = 1.0
    keys = torch.zeros(1, 2, DIM)
    keys[0, :, 0] = torch.tensor([-1.0, 1.0])  # centre: cos -1; far: cos +1
    keys[0, :, 1] = torch.tensor([0.0, 1.0])  # marker: output dim 1 = weight on far
    out = attn(
        query,
        y=keys,
        rope_positions=torch.zeros(1, 1, 2),
        rope_positions_y=torch.tensor([[[0.0, 0.0], [7 * sigma, 0.0]]]),
    )
    # logits: centre -10, far +10 - 49/2 = -14.5 -> weight 1 / (1 + e^4.5)
    torch.testing.assert_close(
        out[0, 0, 1], torch.tensor(1 / (1 + math.exp(4.5))), atol=1e-5, rtol=1e-4
    )


def test_window_params_get_gradients() -> None:
    """Gradients reach the window predictor (zero init is not a dead end)."""
    torch.manual_seed(0)
    attn = Attention(DIM, num_heads=H, qkv_bias=True, position_encoding="gaussian_3d")
    pos = _positions(6, True)
    attn(torch.randn(B, 6, DIM), rope_positions=pos).square().sum().backward()
    assert attn.gaussian is not None
    grad = attn.gaussian.to_params.weight.grad
    assert grad is not None and grad.abs().sum() > 0
