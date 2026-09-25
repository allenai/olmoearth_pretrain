"""The sync-free rewrites and the compiled RoPE reproduce the code they replace."""

import pytest
import torch
import torch.nn.functional as F

from olmoearth_pretrain.data.constants import MISSING_VALUE
from olmoearth_pretrain.nn import encodings
from olmoearth_pretrain.nn.flexi_vit import Encoder
from olmoearth_pretrain.nn.supervision_head import (
    _binary_classification_loss,
    _classification_loss,
    _regression_loss,
)


def _old_add_removed_tokens(
    x: torch.Tensor, indices: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    masked_tokens = torch.zeros(x.shape[0], indices.shape[1], x.shape[2], dtype=x.dtype)
    full_mask = torch.cat(
        (
            mask,
            torch.zeros((x.shape[0], indices.shape[1] - x.shape[1]), dtype=mask.dtype),
        ),
        dim=-1,
    )
    out = masked_tokens.clone()
    out[full_mask] = x[mask]
    out = out.scatter(1, indices[:, :, None].expand_as(out), out)
    full_mask = full_mask.scatter(1, indices.expand_as(full_mask), full_mask)
    return out, full_mask


def test_add_removed_tokens_matches_boolean_assignment() -> None:
    """Token re-insertion is exactly the old boolean-mask assignment."""
    torch.manual_seed(0)
    B, N, D = 3, 11, 4
    tokens = torch.randn(B, N, D)
    keep = torch.rand(B, N) > 0.4
    keep[:, 0] = True
    x, indices, new_mask, _, _ = Encoder.remove_masked_tokens(tokens, keep)
    new_out, new_full = Encoder.add_removed_tokens(x, indices, new_mask)
    old_out, old_full = _old_add_removed_tokens(x, indices, new_mask)
    torch.testing.assert_close(new_out, old_out, rtol=0, atol=0)
    assert torch.equal(new_full, old_full)
    torch.testing.assert_close(new_out[keep], tokens[keep], rtol=0, atol=0)


def _targets(
    shape: tuple[int, ...],
    valid_frac: float,
    binary: bool = False,
    classes: torch.Tensor | None = None,
) -> torch.Tensor:
    t = torch.rand(shape) if not binary else (torch.rand(shape) > 0.7).float()
    if classes is not None:
        t = classes[torch.randint(len(classes), shape)]
    invalid = torch.rand(shape[:-1]) > valid_frac
    t[invalid] = MISSING_VALUE
    return t


@pytest.mark.parametrize("valid_frac", [0.6, 0.0])
def test_supervision_losses_match_boolean_indexing(valid_frac: float) -> None:
    """Masked means equal the old ``x[mask].mean()`` losses, values and gradients.

    With nothing valid the old code returned ``0 * pred.sum()``: the new losses must
    be 0 with zero gradients.
    """
    torch.manual_seed(1)
    B, H, W, T = 2, 5, 6, 1
    # --- classification
    classes = torch.tensor([0.1, 0.2, 0.5, 0.9])
    pred = torch.randn(B, H, W, 4, requires_grad=True)
    raw = _targets((B, H, W, 1), valid_frac, classes=classes)
    valid = (raw != MISSING_VALUE).all(dim=-1)
    new = _classification_loss(pred, raw, valid, classes)
    new.backward()
    g_new = pred.grad.clone()
    pred.grad = None
    if valid.any():
        idx = (raw[..., 0].unsqueeze(-1) - classes).abs().argmin(dim=-1)
        old = F.cross_entropy(pred[valid].float(), idx[valid])
        old.backward()
        torch.testing.assert_close(new, old)
        torch.testing.assert_close(g_new, pred.grad)
    else:
        assert new.item() == 0.0 and torch.count_nonzero(g_new) == 0
    # --- binary, with and without pos_weight
    for pw in (False, True):
        pred = torch.randn(B, H, W, T, 3, requires_grad=True)
        raw = _targets((B, H, W, T, 3), valid_frac, binary=True)
        valid = (raw != MISSING_VALUE).all(dim=-1)
        new = _binary_classification_loss(pred, raw, valid, pos_weight=pw)
        new.backward()
        g_new = pred.grad.clone()
        pred.grad = None
        if valid.any():
            ve = valid.unsqueeze(-1).expand_as(pred)
            if pw:
                vm = valid.float().unsqueeze(-1)
                p = (
                    (raw.float() * vm).sum(dim=(0, 1, 2, 3)) / vm.sum().clamp(min=1)
                ).clamp(1e-3, 1 - 1e-3)
                old = F.binary_cross_entropy_with_logits(
                    pred.float(), raw.float(), pos_weight=(1 - p) / p, reduction="none"
                )[ve].mean()
            else:
                old = F.binary_cross_entropy_with_logits(
                    pred[ve].float(), raw[ve].float()
                )
            old.backward()
            torch.testing.assert_close(new, old)
            torch.testing.assert_close(g_new, pred.grad)
        else:
            assert new.item() == 0.0 and torch.count_nonzero(g_new) == 0
    # --- regression, l1 / mse, with and without norm_pix
    for loss_type in ("l1", "mse"):
        for norm_pix in (False, True):
            h = w = 4
            pred = torch.randn(B, h, w, T, 2, requires_grad=True)
            raw = _targets((B, h, w, T, 2), valid_frac)
            valid = (raw != MISSING_VALUE).all(dim=-1)
            if norm_pix:
                valid = valid[..., 0]
            new = _regression_loss(
                pred,
                raw,
                valid,
                norm_pix_loss=norm_pix,
                max_patch_size=2,
                regression_loss_type=loss_type,
            )
            new.backward()
            g_new = pred.grad.clone()
            pred.grad = None
            if not valid.any():
                assert new.item() == 0.0 and torch.count_nonzero(g_new) == 0
                continue
            if not norm_pix:
                ve = valid.unsqueeze(-1).expand_as(pred)
                fn = F.l1_loss if loss_type == "l1" else F.mse_loss
                old = fn(pred[ve].float(), raw[ve].float())
                old.backward()
                torch.testing.assert_close(new, old)
                torch.testing.assert_close(g_new, pred.grad)
            else:
                assert torch.isfinite(new)


@pytest.mark.parametrize("with_extents", [False, True])
def test_compiled_mixed_rope_matches_eager(with_extents: bool) -> None:
    """``torch.compile`` of the mixed-RoPE op matches eager, forward and backward."""
    torch.manual_seed(0)
    B, H, N, D = 2, 3, 13, 16
    freqs = encodings.init_3d_mixed_rope_freqs(D, H, base=100.0).requires_grad_(True)
    x = torch.randn(B, H, N, D, requires_grad=True)
    pos = torch.rand(B, N, 3) * 5
    kwargs = {}
    if with_extents:
        t_w, s_w = torch.rand(B, N), torch.rand(B, N)
        t_w[:, :7] = 0
        s_w[:, :7] = 0
        kwargs = dict(extent=t_w, spatial_extent=s_w, extent_start=7)
    # A rotation preserves length, so a sum of squares would give the frequencies an
    # identically-zero gradient (pure rounding noise); weight the output instead.
    w = torch.randn(B, H, N, D)
    eager = encodings._apply_3d_mixed_rope_eager(x, pos, freqs, **kwargs)
    (eager * w).sum().backward()
    gx, gf = x.grad.clone(), freqs.grad.clone()
    x.grad = freqs.grad = None
    try:
        encodings.use_compiled_mixed_rope(True)
        compiled = encodings.apply_3d_mixed_rope(x, pos, freqs, **kwargs)
        (compiled * w).sum().backward()
    except Exception as e:  # pragma: no cover - no working compiler on this host
        pytest.skip(f"torch.compile unavailable here: {type(e).__name__}: {e}")
    finally:
        encodings.use_compiled_mixed_rope(False)
    torch.testing.assert_close(compiled, eager, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(x.grad, gx, atol=1e-5, rtol=1e-5)
    # The frequency gradient is a sum over every batch, position and head dim, so
    # compare it relative to its norm (fp32 reassociation of a long sum).
    rel = (freqs.grad - gf).norm() / gf.norm()
    assert rel < 1e-4, f"frequency gradient differs by {rel:.2e} of its norm"
