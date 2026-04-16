"""Unit tests for the NLP supervision decoder."""

from collections.abc import Mapping

import pytest
import torch

from olmoearth_pretrain.data.constants import MISSING_VALUE
from olmoearth_pretrain.datatypes import MaskValue, TokensAndMasks
from olmoearth_pretrain.nn.nlp_supervision import (
    NLPSupervisionDecoder,
    NLPSupervisionDecoderConfig,
    _compute_classification_loss,
    _compute_regression_loss,
    _select_reference_modality,
    _slice_modality,
    _to_pixel_grid,
    flatten_encoder_tokens,
)
from olmoearth_pretrain.open_set.catalog.registry import (
    ClassEntry,
    NormalizedValueEqExtractor,
    RegressionExtractor,
)
from olmoearth_pretrain.open_set.model.cross_attn_decoder import CrossAttnDecoderConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

B, P_H, P_W, T, BS, D = 2, 3, 3, 2, 1, 16
MPS = 2  # max_patch_size


def _make_tokens_and_masks() -> TokensAndMasks:
    """Create a minimal TokensAndMasks with S2 and S1 modalities."""
    s2_tokens = torch.randn(B, P_H, P_W, T, BS, D)
    s2_mask = torch.full((B, P_H, P_W, T, BS), MaskValue.ONLINE_ENCODER.value)
    s1_tokens = torch.randn(B, P_H, P_W, T, BS, D)
    s1_mask = torch.full((B, P_H, P_W, T, BS), MaskValue.ONLINE_ENCODER.value)
    return TokensAndMasks(
        sentinel2_l2a=s2_tokens,
        sentinel2_l2a_mask=s2_mask,
        sentinel1=s1_tokens,
        sentinel1_mask=s1_mask,
    )


# ---------------------------------------------------------------------------
# Tests: flatten_encoder_tokens
# ---------------------------------------------------------------------------


class TestFlattenEncoderTokens:
    """Tests for the flatten_encoder_tokens utility."""

    def test_output_shapes(self) -> None:
        """Flattened output has expected shape across two modalities."""
        tam = _make_tokens_and_masks()
        tokens, context_mask, shapes = flatten_encoder_tokens(tam)

        expected_n = P_H * P_W * T * BS * 2  # two modalities
        assert tokens.shape == (B, expected_n, D)
        assert context_mask.shape == (B, expected_n)
        assert "sentinel2_l2a" in shapes
        assert "sentinel1" in shapes

    def test_context_mask_all_true(self) -> None:
        """All ONLINE_ENCODER tokens produce True in context_mask."""
        tam = _make_tokens_and_masks()
        _, context_mask, _ = flatten_encoder_tokens(tam)
        assert context_mask.all()

    def test_context_mask_missing_tokens(self) -> None:
        """Tokens with DECODER or MISSING mask should not be in context_mask."""
        s2_tokens = torch.randn(B, P_H, P_W, T, BS, D)
        s2_mask = torch.full((B, P_H, P_W, T, BS), MaskValue.DECODER.value)
        tam = TokensAndMasks(sentinel2_l2a=s2_tokens, sentinel2_l2a_mask=s2_mask)
        _, context_mask, _ = flatten_encoder_tokens(tam)
        assert not context_mask.any()


# ---------------------------------------------------------------------------
# Tests: spatial helpers
# ---------------------------------------------------------------------------


class TestSpatialHelpers:
    """Tests for spatial token slicing and grid construction."""

    def test_select_reference_modality(self) -> None:
        """Selects sentinel2_l2a when both S2 and S1 are present."""
        shapes: Mapping[str, tuple[int, ...]] = {
            "sentinel2_l2a": (P_H, P_W, T, BS),
            "sentinel1": (P_H, P_W, T, BS),
        }
        ref = _select_reference_modality(dict(shapes))
        assert ref == "sentinel2_l2a"

    def test_slice_modality(self) -> None:
        """Slicing recovers the original modality shape."""
        tam = _make_tokens_and_masks()
        tokens, _, shapes = flatten_encoder_tokens(tam)
        s2 = _slice_modality(tokens, shapes, "sentinel2_l2a")
        assert s2.shape == (B, P_H, P_W, T, BS, D)

    def test_to_pixel_grid_6d(self) -> None:
        """6-D input collapses temporal and bandset dims."""
        x = torch.randn(B, P_H, P_W, T, BS, D)
        grid = _to_pixel_grid(x)
        assert grid.shape == (B, P_H, P_W, D)

    def test_to_pixel_grid_4d(self) -> None:
        """4-D input passes through unchanged."""
        x = torch.randn(B, P_H, P_W, D)
        grid = _to_pixel_grid(x)
        assert grid.shape == (B, P_H, P_W, D)


# ---------------------------------------------------------------------------
# Tests: loss helpers
# ---------------------------------------------------------------------------


class TestLossHelpers:
    """Tests for per-pixel loss functions."""

    def test_classification_loss_basic(self) -> None:
        """BCE loss is positive for non-trivial prediction."""
        pred = torch.randn(8, 8)
        target = torch.zeros(8, 8)
        target[0, 0] = 1.0
        loss = _compute_classification_loss(pred, target)
        assert loss.ndim == 0
        assert loss.item() > 0

    def test_classification_loss_missing_pixels(self) -> None:
        """All-missing target produces zero loss."""
        pred = torch.randn(8, 8)
        target = torch.full((8, 8), MISSING_VALUE, dtype=torch.float)
        loss = _compute_classification_loss(pred, target)
        assert loss.item() == 0.0

    def test_regression_loss_basic(self) -> None:
        """MSE loss is positive for non-trivial prediction."""
        pred = torch.randn(8, 8)
        target = torch.randn(8, 8)
        loss = _compute_regression_loss(pred, target)
        assert loss.ndim == 0
        assert loss.item() > 0

    def test_regression_loss_missing(self) -> None:
        """All-missing target produces zero loss."""
        pred = torch.randn(8, 8)
        target = torch.full((8, 8), MISSING_VALUE, dtype=torch.float)
        loss = _compute_regression_loss(pred, target)
        assert loss.item() == 0.0


# ---------------------------------------------------------------------------
# Tests: NLPSupervisionDecoderConfig
# ---------------------------------------------------------------------------


class TestNLPSupervisionDecoderConfig:
    """Tests for building the NLP supervision decoder from config."""

    def test_build_mode_b(self) -> None:
        """Mode (b) builds without regression heads."""
        config = NLPSupervisionDecoderConfig(
            decoder_config=CrossAttnDecoderConfig(dim=32, depth=1, num_heads=2),
            text_dim=32,
            text_condition_regression=True,
        )
        decoder = config.build(encoder_dim=D, max_patch_size=MPS)
        assert isinstance(decoder, NLPSupervisionDecoder)
        assert decoder.regression_heads is None

    def test_build_mode_a(self) -> None:
        """Mode (a) builds with per-modality regression heads."""
        config = NLPSupervisionDecoderConfig(
            decoder_config=CrossAttnDecoderConfig(dim=32, depth=1, num_heads=2),
            text_dim=32,
            text_condition_regression=False,
            regression_modality_names=["srtm", "wri_canopy_height_map"],
        )
        decoder = config.build(encoder_dim=D, max_patch_size=MPS)
        assert decoder.regression_heads is not None
        assert "srtm" in decoder.regression_heads
        assert "wri_canopy_height_map" in decoder.regression_heads


# ---------------------------------------------------------------------------
# Tests: NLPSupervisionDecoder._predict_text_conditioned
# ---------------------------------------------------------------------------


class TestPredictTextConditioned:
    """Tests for the text-conditioned prediction path."""

    @pytest.fixture
    def decoder(self) -> NLPSupervisionDecoder:
        """Build a small decoder for testing."""
        config = NLPSupervisionDecoderConfig(
            decoder_config=CrossAttnDecoderConfig(dim=32, depth=1, num_heads=2),
            text_dim=32,
        )
        return config.build(encoder_dim=D, max_patch_size=MPS)

    def test_output_shape(self, decoder: NLPSupervisionDecoder) -> None:
        """Output is [C, B, P_H*mps, P_W*mps] without target_size."""
        tam = _make_tokens_and_masks()
        tokens, mask, shapes = flatten_encoder_tokens(tam)
        n_classes = 3
        text_tokens = torch.randn(n_classes, 5, 32)  # [C, L, D_text]

        preds = decoder._predict_text_conditioned(
            encoder_tokens=tokens,
            context_mask=mask,
            shapes=shapes,
            text_tokens=text_tokens,
            text_attn_mask=None,
            n_classes=n_classes,
            target_size=None,
        )
        # Output: [C, B, P_H*mps, P_W*mps]
        assert preds.shape == (n_classes, B, P_H * MPS, P_W * MPS)

    def test_output_shape_with_target_size(
        self, decoder: NLPSupervisionDecoder
    ) -> None:
        """Output downsamples to target_size when specified."""
        tam = _make_tokens_and_masks()
        tokens, mask, shapes = flatten_encoder_tokens(tam)
        n_classes = 2
        text_tokens = torch.randn(n_classes, 5, 32)
        target_size = (P_H, P_W)  # smaller than native P_H*MPS

        preds = decoder._predict_text_conditioned(
            encoder_tokens=tokens,
            context_mask=mask,
            shapes=shapes,
            text_tokens=text_tokens,
            text_attn_mask=None,
            n_classes=n_classes,
            target_size=target_size,
        )
        assert preds.shape == (n_classes, B, P_H, P_W)


# ---------------------------------------------------------------------------
# Tests: catalog entries
# ---------------------------------------------------------------------------


class TestCatalogEntries:
    """Tests for class catalog entries and extractors."""

    def test_worldcover_entry(self) -> None:
        """NormalizedValueEqExtractor matches the expected class value."""
        entry = ClassEntry(
            text="tree cover",
            source="worldcover",
            extractor=NormalizedValueEqExtractor(class_value=0.1),
            task_type="classification",
        )
        tensor = torch.full((2, 8, 8, 1, 1), 0.1)
        mask = entry.extractor(tensor)
        assert mask.shape == (2, 8, 8)
        assert mask.sum() == 2 * 8 * 8

    def test_regression_extractor(self) -> None:
        """RegressionExtractor returns raw continuous values."""
        entry = ClassEntry(
            text="elevation",
            source="srtm",
            extractor=RegressionExtractor(band_index=0),
            task_type="regression",
        )
        assert entry.is_regression
        tensor = torch.randn(2, 8, 8, 1, 1)
        result = entry.extractor(tensor)
        assert result.shape == (2, 8, 8)
        assert torch.allclose(result, tensor[..., 0, 0])

    def test_normalized_value_eq_no_match(self) -> None:
        """NormalizedValueEqExtractor returns zero mask when no pixels match."""
        ext = NormalizedValueEqExtractor(class_value=0.5, tolerance=0.01)
        tensor = torch.full((1, 4, 4, 1, 1), 0.9)
        mask = ext(tensor)
        assert mask.sum() == 0

    def test_build_default_registry(self) -> None:
        """Default registry includes entries from all expected sources."""
        from olmoearth_pretrain.open_set.catalog import build_default_registry

        registry = build_default_registry()
        assert len(registry) > 0
        # Check we have entries from multiple sources.
        sources = registry.sources()
        assert "openstreetmap_raster" in sources
        assert "worldcover" in sources
        assert "srtm" in sources
