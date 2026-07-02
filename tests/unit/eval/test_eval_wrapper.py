"""Unit tests for eval wrapper."""

import torch

from olmoearth_pretrain.evals.eval_wrapper import EvalWrapper


class TestExtractCenterToken:
    """Tests for _extract_center_token static method."""

    def test_odd_spatial_dims(self) -> None:
        """Get center token for odd dimensions."""
        B, H, W, D = 2, 7, 7, 64
        x = torch.randn(B, H, W, D)
        result = EvalWrapper._extract_center_token(x)
        assert result.shape == (B, D)
        assert torch.equal(result, x[:, 3, 3, :])

    def test_even_spatial_dims(self) -> None:
        """Get bottom-right of center for even dimensions."""
        B, H, W, D = 2, 8, 8, 64
        x = torch.randn(B, H, W, D)
        result = EvalWrapper._extract_center_token(x)
        assert result.shape == (B, D)
        assert torch.equal(result, x[:, 4, 4, :])

    def test_non_square(self) -> None:
        """Correct center for non-square dimensions."""
        B, H, W, D = 3, 4, 6, 32
        x = torch.randn(B, H, W, D)
        result = EvalWrapper._extract_center_token(x)
        assert result.shape == (B, D)
        assert torch.equal(result, x[:, 2, 3, :])
