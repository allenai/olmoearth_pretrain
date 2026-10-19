"""Tests for the sub-token phase loss."""

import torch

from olmoearth_pretrain.train.phase_loss import phase_loss


def test_phase_loss_sees_a_column_offset_only() -> None:
    """Random tokens score near 0; an odd-column offset scores high."""
    torch.manual_seed(0)
    z = torch.randn(8, 16, 16, 32)
    assert phase_loss(z.flatten(1, 2), (8, 8), 2).item() < 0.01
    z[:, :, 1::2] += 1.0
    assert phase_loss(z.flatten(1, 2), (8, 8), 2).item() > 0.1


def test_phase_loss_gradient_shrinks_the_offset() -> None:
    """Descending the loss reduces the offset."""
    torch.manual_seed(0)
    offset = torch.ones(32, requires_grad=True)
    z = torch.randn(4, 8, 8, 32)
    odd = (torch.arange(8) % 2).float()[None, None, :, None]
    phase_loss((z + odd * offset).flatten(1, 2), (4, 4), 2).backward()
    assert offset.grad is not None and (offset.grad > 0).all()
