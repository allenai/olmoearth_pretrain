"""Smoke tests for the cross-attention decoder."""

from __future__ import annotations

import torch

from olmoearth_pretrain.open_set.model.cross_attn_decoder import (
    CrossAttnDecoder,
    CrossAttnDecoderConfig,
)


class TestCrossAttnDecoder:
    """Decoder must preserve the image-token sequence shape."""

    def test_output_shape_matches_input(self) -> None:
        """Refined image tokens have shape ``[B, N, dim]``."""
        cfg = CrossAttnDecoderConfig(dim=32, depth=2, num_heads=4, use_flash_attn=False)
        dec = cfg.build(image_dim=48, text_dim=64)
        b, n, ltext = 2, 16, 7
        img = torch.randn(b, n, 48)
        text = torch.randn(b, ltext, 64)
        out = dec(img, text)
        assert out.shape == (b, n, 32)

    def test_text_attn_mask_does_not_break_forward(self) -> None:
        """Passing a text padding mask runs without shape errors."""
        cfg = CrossAttnDecoderConfig(dim=16, depth=1, num_heads=2, use_flash_attn=False)
        dec = cfg.build(image_dim=16, text_dim=16)
        b, n, ltext = 2, 4, 3
        img = torch.randn(b, n, 16)
        text = torch.randn(b, ltext, 16)
        attn = torch.tensor([[True, True, False], [True, False, False]])
        out = dec(img, text, text_attn_mask=attn)
        assert out.shape == (b, n, 16)

    def test_identity_projection_when_dims_match(self) -> None:
        """No projection layer is added if dims already match."""
        cfg = CrossAttnDecoderConfig(dim=24, depth=1, num_heads=4)
        dec = CrossAttnDecoder(cfg, image_dim=24, text_dim=24)
        assert isinstance(dec.image_proj, torch.nn.Identity)
        assert isinstance(dec.text_proj, torch.nn.Identity)

    def test_gradients_flow(self) -> None:
        """Gradients reach the decoder's parameters from the output."""
        cfg = CrossAttnDecoderConfig(dim=16, depth=1, num_heads=2)
        dec = cfg.build(image_dim=16, text_dim=16)
        img = torch.randn(1, 4, 16, requires_grad=True)
        text = torch.randn(1, 3, 16)
        out = dec(img, text)
        out.sum().backward()
        # At least one decoder parameter must have a grad.
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in dec.parameters()
        )
