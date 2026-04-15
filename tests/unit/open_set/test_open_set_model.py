"""End-to-end shape tests for :class:`OpenSetSegmenter` with a stub encoder.

We do not load a real OlmoEarth checkpoint here — instead we substitute a
small ``nn.Module`` that mimics the encoder forward contract (returns flat
``[B, N, D]`` tokens, a context mask, and a per-modality shapes dict).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample
from olmoearth_pretrain.open_set.model.cross_attn_decoder import (
    CrossAttnDecoderConfig,
)
from olmoearth_pretrain.open_set.model.open_set_model import (
    OpenSetSegmenter,
    OpenSetSegmenterConfig,
)


class _StubEncoder(nn.Module):
    """Returns deterministic tokens shaped to mimic Sentinel-2-only output."""

    def __init__(
        self,
        embedding_dim: int = 32,
        p_h: int = 4,
        p_w: int = 4,
        time: int = 2,
        bandsets: int = 1,
        max_patch_size: int = 2,
    ) -> None:
        super().__init__()
        self._embedding_dim = embedding_dim
        self._p_h = p_h
        self._p_w = p_w
        self._time = time
        self._bandsets = bandsets
        self.max_patch_size = max_patch_size

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def forward(
        self,
        sample: MaskedOlmoEarthSample,
        patch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, tuple[int, ...]]]:
        # Use sample.batch_size to get B; for a stub we just look at any tensor.
        b = sample.timestamps.shape[0] if sample.timestamps is not None else 1
        n = self._p_h * self._p_w * self._time * self._bandsets
        tokens = torch.randn(b, n, self._embedding_dim)
        mask = torch.ones(b, n, dtype=torch.bool)
        shapes: dict[str, tuple[int, ...]] = {
            "sentinel2_l2a": (self._p_h, self._p_w, self._time, self._bandsets)
        }
        return tokens, mask, shapes


def _make_masked_sample(b: int = 2, t: int = 2) -> MaskedOlmoEarthSample:
    """Build a dummy ``MaskedOlmoEarthSample`` with an OSM-shaped raster."""
    return MaskedOlmoEarthSample(
        timestamps=torch.zeros(b, t, 3, dtype=torch.long),
        openstreetmap_raster=torch.zeros(b, 8, 8, 1, 30),
        openstreetmap_raster_mask=torch.zeros(b, 8, 8, 1, 1, dtype=torch.long),
    )


class TestOpenSetSegmenter:
    """The forward must produce ``[N_classes, B, H, W]`` logits."""

    def _build(
        self, encoder_dim: int = 32, dec_dim: int = 16, text_dim: int = 24
    ) -> OpenSetSegmenter:
        encoder = _StubEncoder(embedding_dim=encoder_dim)
        cfg = OpenSetSegmenterConfig(
            decoder_config=CrossAttnDecoderConfig(
                dim=dec_dim, depth=1, num_heads=4, use_flash_attn=False
            ),
            text_dim=text_dim,
        )
        return OpenSetSegmenter(cfg, encoder=encoder)

    def test_forward_produces_expected_shape(self) -> None:
        """Output is ``[C, B, H_out, W_out]`` after downsample to target_size."""
        model = self._build()  # native: 4x4 patches * mps=2 -> 8x8
        sample = _make_masked_sample(b=2, t=2)
        n_classes, ltext, text_dim = 3, 5, 24
        text_tokens = torch.randn(n_classes, ltext, text_dim)
        text_pooled = torch.randn(n_classes, text_dim)
        # Downsample 8x8 native -> 4x4 target.
        out = model(
            sample=sample,
            patch_size=4,
            text_tokens=text_tokens,
            text_pooled=text_pooled,
            target_size=(4, 4),
        )
        assert out.shape == (n_classes, 2, 4, 4)

    def test_forward_without_target_size_uses_patch_grid_times_max_patch_size(
        self,
    ) -> None:
        """Without ``target_size`` the output is at ``P_H * max_patch_size``."""
        model = self._build()  # stub encoder: 4x4 patch grid, max_patch_size=2
        sample = _make_masked_sample(b=1, t=2)
        text_tokens = torch.randn(2, 4, 24)
        text_pooled = torch.randn(2, 24)
        out = model(
            sample=sample,
            patch_size=4,
            text_tokens=text_tokens,
            text_pooled=text_pooled,
        )
        # 4 (patch grid) * 2 (max_patch_size) = 8
        assert out.shape == (2, 1, 8, 8)

    def test_target_size_larger_than_native_raises(self) -> None:
        """Asking for a larger ``target_size`` than the native output errors out."""
        import pytest

        model = self._build()  # native output: 4x4 * 2 = 8x8
        sample = _make_masked_sample(b=1, t=2)
        text_tokens = torch.randn(2, 4, 24)
        text_pooled = torch.randn(2, 24)
        with pytest.raises(ValueError, match="exceeds the model's native"):
            model(
                sample=sample,
                patch_size=4,
                text_tokens=text_tokens,
                text_pooled=text_pooled,
                target_size=(16, 16),
            )

    def test_target_size_at_native_resolution_skips_interp(self) -> None:
        """When ``target_size`` matches the native output, no interpolation runs."""
        model = self._build()  # native output: 4x4 * 2 = 8x8
        sample = _make_masked_sample(b=1, t=2)
        text_tokens = torch.randn(2, 4, 24)
        text_pooled = torch.randn(2, 24)
        out = model(
            sample=sample,
            patch_size=4,
            text_tokens=text_tokens,
            text_pooled=text_pooled,
            target_size=(8, 8),
        )
        assert out.shape == (2, 1, 8, 8)

    def test_grad_reaches_decoder_only(self) -> None:
        """Encoder parameters (none on the stub) and text projections all live in the model."""
        model = self._build()  # native output: 8x8
        sample = _make_masked_sample(b=1, t=1)
        text_tokens = torch.randn(1, 3, 24)
        text_pooled = torch.randn(1, 24)
        out = model(
            sample=sample,
            patch_size=4,
            text_tokens=text_tokens,
            text_pooled=text_pooled,
            target_size=(8, 8),
        )
        out.sum().backward()
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.decoder.parameters()
        )
        assert model.text_proj.weight.grad is not None
        assert model.pixel_proj.weight.grad is not None
