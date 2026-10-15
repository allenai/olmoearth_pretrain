"""Tests for the sub-token phase ratio."""

import torch

from olmoearth_pretrain.nn.latent_phase import latent_phase_stats


def _grid(batch: int, n_h: int, n_w: int, r: int, dim: int) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(batch, n_h * r, n_w * r, dim)


def test_none_at_token_resolution() -> None:
    """One latent per token has no phase."""
    z = torch.randn(2, 12, 8)
    assert latent_phase_stats(z, (3, 4)) is None


def test_unbiased_grid_is_near_zero() -> None:
    """Random latents sit at the noise floor."""
    z = _grid(8, 8, 8, 2, 16)
    stats = latent_phase_stats(z.flatten(1, 2), (8, 8))
    assert stats is not None
    for name in ("phase", "col", "row", "control"):
        assert stats[name].item() < 0.01


def test_column_offset_is_seen_by_col_not_row_or_control() -> None:
    """An odd-column offset shows in col only."""
    z = _grid(8, 8, 8, 2, 16)
    z[:, :, 1::2] += 1.0  # odd columns shifted by a constant
    stats = latent_phase_stats(z.flatten(1, 2), (8, 8))
    assert stats is not None
    assert stats["col"].item() > 0.1
    assert stats["phase"].item() > 0.1
    assert stats["row"].item() < 0.01
    assert stats["control"].item() < 0.01


def test_penalty_gradient_removes_offset() -> None:
    """The penalty gradient shrinks the offset."""
    offset = torch.zeros(16, requires_grad=True)
    z = _grid(4, 4, 4, 2, 16)
    mask = torch.zeros(1, 1, 8, 1)
    mask[..., 1::2, :] = 1.0
    shifted = z + mask * (1.0 + offset)
    stats = latent_phase_stats(shifted.flatten(1, 2), (4, 4))
    assert stats is not None
    stats["phase"].backward()
    assert offset.grad is not None and (offset.grad > 0).all()
