"""Unit tests for the channel attention module."""

import pytest
import torch

from olmoearth_pretrain.data.constants import Modality, ModalitySpec
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue
from olmoearth_pretrain.nn.channel_attention import (
    AggregatedPredictor,
    ChnAttn,
    ChnAttnConfig,
)
from olmoearth_pretrain.nn.flexi_vit import (
    Encoder,
    EncoderConfig,
    TokensAndMasks,
)


@pytest.fixture
def spatial_modalities() -> list[ModalitySpec]:
    """Modalities for testing channel attention (spatial only)."""
    return [Modality.SENTINEL2_L2A, Modality.SENTINEL1]


@pytest.fixture
def all_modalities() -> list[ModalitySpec]:
    """All modalities for testing."""
    return [Modality.SENTINEL2_L2A, Modality.LATLON]


class TestChnAttn:
    """Unit tests for the ChnAttn module."""

    def test_forward_all_mode_shapes(self) -> None:
        """Test ChnAttn 'all' mode produces correct output shapes."""
        B, H, W, T, D = 2, 3, 3, 2, 16
        modalities = [Modality.SENTINEL2_L2A, Modality.SENTINEL1]
        modality_names = [m.name for m in modalities]
        num_bandsets = {
            "sentinel2_l2a": 3,
            "sentinel1": 1,
        }

        chnattn = ChnAttn(
            embedding_size=D,
            attn_dim=D,
            num_heads=2,
            aggregation_mode="all",
            supported_modality_names=modality_names,
            num_bandsets_per_modality=num_bandsets,
        )

        tokens_dict = {
            "sentinel2_l2a": torch.randn(B, H, W, T, 3, D),
            "sentinel2_l2a_mask": torch.full(
                (B, H, W, T, 3), MaskValue.ONLINE_ENCODER.value
            ),
            "sentinel1": torch.randn(B, H, W, T, 1, D),
            "sentinel1_mask": torch.full(
                (B, H, W, T, 1), MaskValue.ONLINE_ENCODER.value
            ),
        }

        agg_tokens, agg_mask = chnattn(tokens_dict, modality_names)
        assert agg_tokens.shape == (B, H, W, T, 1, D)
        assert agg_mask.shape == (B, H, W, T, 1)

    def test_forward_per_modality_mode_shapes(self) -> None:
        """Test ChnAttn 'per_modality' mode produces correct output shapes."""
        B, H, W, T, D = 2, 3, 3, 2, 16
        modalities = [Modality.SENTINEL2_L2A, Modality.SENTINEL1]
        modality_names = [m.name for m in modalities]
        num_bandsets = {"sentinel2_l2a": 3, "sentinel1": 1}

        chnattn = ChnAttn(
            embedding_size=D,
            attn_dim=D,
            num_heads=2,
            aggregation_mode="per_modality",
            supported_modality_names=modality_names,
            num_bandsets_per_modality=num_bandsets,
        )

        tokens_dict = {
            "sentinel2_l2a": torch.randn(B, H, W, T, 3, D),
            "sentinel2_l2a_mask": torch.full(
                (B, H, W, T, 3), MaskValue.ONLINE_ENCODER.value
            ),
            "sentinel1": torch.randn(B, H, W, T, 1, D),
            "sentinel1_mask": torch.full(
                (B, H, W, T, 1), MaskValue.ONLINE_ENCODER.value
            ),
        }

        agg_tokens, agg_mask = chnattn(tokens_dict, modality_names)
        # 2 modalities -> 2 tokens per (h,w,t)
        assert agg_tokens.shape == (B, H, W, T, 2, D)
        assert agg_mask.shape == (B, H, W, T, 2)

    def test_mask_awareness(self) -> None:
        """Test that ChnAttn respects masks (MISSING tokens excluded)."""
        B, H, W, T, D = 1, 2, 2, 1, 8
        modality_names = ["sentinel2_l2a"]
        num_bandsets = {"sentinel2_l2a": 3}

        chnattn = ChnAttn(
            embedding_size=D,
            attn_dim=D,
            num_heads=2,
            aggregation_mode="all",
            supported_modality_names=modality_names,
            num_bandsets_per_modality=num_bandsets,
        )

        # All MISSING -> mask should be MISSING
        tokens_dict = {
            "sentinel2_l2a": torch.randn(B, H, W, T, 3, D),
            "sentinel2_l2a_mask": torch.full((B, H, W, T, 3), MaskValue.MISSING.value),
        }

        agg_tokens, agg_mask = chnattn(tokens_dict, modality_names)
        assert (agg_mask == MaskValue.MISSING.value).all()

        # Mix of ONLINE_ENCODER and MISSING -> mask should be ONLINE_ENCODER
        mask = torch.full((B, H, W, T, 3), MaskValue.MISSING.value)
        mask[..., 0] = MaskValue.ONLINE_ENCODER.value
        tokens_dict["sentinel2_l2a_mask"] = mask
        agg_tokens, agg_mask = chnattn(tokens_dict, modality_names)
        assert (agg_mask == MaskValue.ONLINE_ENCODER.value).all()

    def test_all_missing_no_nan(self) -> None:
        """Test that all-MISSING positions produce zeros, not NaN."""
        B, H, W, T, D = 1, 2, 2, 1, 8
        modality_names = ["sentinel2_l2a"]
        num_bandsets = {"sentinel2_l2a": 3}

        chnattn = ChnAttn(
            embedding_size=D,
            attn_dim=D,
            num_heads=2,
            aggregation_mode="all",
            supported_modality_names=modality_names,
            num_bandsets_per_modality=num_bandsets,
        )

        tokens_dict = {
            "sentinel2_l2a": torch.randn(B, H, W, T, 3, D),
            "sentinel2_l2a_mask": torch.full((B, H, W, T, 3), MaskValue.MISSING.value),
        }

        agg_tokens, agg_mask = chnattn(tokens_dict, modality_names)
        # All MISSING -> output should be zero, no NaN
        assert not torch.isnan(agg_tokens).any(), (
            "NaN found in output with all-MISSING inputs"
        )
        assert (agg_tokens == 0).all(), "Expected zeros for all-MISSING positions"
        assert (agg_mask == MaskValue.MISSING.value).all()

    def test_partial_missing_no_nan(self) -> None:
        """Test mixed MISSING/ONLINE_ENCODER across (h,w,t) positions."""
        B, H, W, T, D = 1, 2, 2, 1, 8
        modality_names = ["sentinel2_l2a"]
        num_bandsets = {"sentinel2_l2a": 3}

        chnattn = ChnAttn(
            embedding_size=D,
            attn_dim=D,
            num_heads=2,
            aggregation_mode="all",
            supported_modality_names=modality_names,
            num_bandsets_per_modality=num_bandsets,
        )

        mask = torch.full((B, H, W, T, 3), MaskValue.MISSING.value)
        # Only (0,0,0) has ONLINE_ENCODER tokens
        mask[0, 0, 0, 0, :] = MaskValue.ONLINE_ENCODER.value

        tokens_dict = {
            "sentinel2_l2a": torch.randn(B, H, W, T, 3, D),
            "sentinel2_l2a_mask": mask,
        }

        agg_tokens, agg_mask = chnattn(tokens_dict, modality_names)
        assert not torch.isnan(agg_tokens).any(), "NaN found in output"
        # Position (0,0,0) should be non-zero, others should be zero
        assert (agg_tokens[0, 0, 0, 0] != 0).any()
        assert (agg_tokens[0, 1, 0, 0] == 0).all()
        assert agg_mask[0, 0, 0, 0, 0] == MaskValue.ONLINE_ENCODER.value
        assert agg_mask[0, 1, 0, 0, 0] == MaskValue.MISSING.value

    def test_configurable_attn_dim(self) -> None:
        """Test that wider attn_dim works."""
        B, H, W, T, D = 2, 2, 2, 1, 8
        attn_dim = 32
        modality_names = ["sentinel2_l2a"]
        num_bandsets = {"sentinel2_l2a": 3}

        chnattn = ChnAttn(
            embedding_size=D,
            attn_dim=attn_dim,
            num_heads=4,
            aggregation_mode="all",
            supported_modality_names=modality_names,
            num_bandsets_per_modality=num_bandsets,
        )

        tokens_dict = {
            "sentinel2_l2a": torch.randn(B, H, W, T, 3, D),
            "sentinel2_l2a_mask": torch.full(
                (B, H, W, T, 3), MaskValue.ONLINE_ENCODER.value
            ),
        }

        agg_tokens, agg_mask = chnattn(tokens_dict, modality_names)
        # Output should be back in embedding_size, not attn_dim
        assert agg_tokens.shape == (B, H, W, T, 1, D)


class TestChnAttnConfig:
    """Tests for ChnAttnConfig."""

    def test_valid_config(self) -> None:
        """Test valid config passes validation."""
        config = ChnAttnConfig(enabled=True, aggregation_mode="all")
        config.validate()

    def test_invalid_aggregation_mode(self) -> None:
        """Test invalid aggregation_mode raises."""
        config = ChnAttnConfig(enabled=True, aggregation_mode="invalid")
        with pytest.raises(ValueError, match="aggregation_mode"):
            config.validate()


class TestAggregatedPredictor:
    """Unit tests for the AggregatedPredictor."""

    @pytest.fixture
    def predictor(self, all_modalities: list[ModalitySpec]) -> AggregatedPredictor:
        """Create AggregatedPredictor fixture."""
        return AggregatedPredictor(
            supported_modalities=all_modalities,
            encoder_embedding_size=16,
            decoder_embedding_size=16,
            depth=1,
            mlp_ratio=2.0,
            num_heads=2,
            max_sequence_length=12,
            output_embedding_size=16,
        )

    def test_forward_shapes(
        self, predictor: AggregatedPredictor, all_modalities: list[ModalitySpec]
    ) -> None:
        """Test AggregatedPredictor produces correct output shapes."""
        B, H, W, T, D = 2, 2, 2, 2, 16

        # Create aggregated encoder output
        agg_tokens = torch.randn(B, H, W, T, 1, D)
        agg_mask = torch.full(
            (B, H, W, T, 1), MaskValue.ONLINE_ENCODER.value, dtype=torch.long
        )
        encoder_output = TokensAndMasks(
            aggregated=agg_tokens,
            aggregated_mask=agg_mask,
        )

        # Create original masks (per-modality, patchified shapes)
        # S2 has 3 bandsets, latlon has 1
        original_masks = {
            "sentinel2_l2a_mask": torch.full(
                (B, H, W, T, 3), MaskValue.DECODER.value, dtype=torch.long
            ),
            "latlon_mask": torch.full(
                (B, 1), MaskValue.DECODER.value, dtype=torch.long
            ),
        }

        timestamps = torch.randint(0, 12, (B, T, 3), dtype=torch.long)

        output = predictor(
            encoder_output=encoder_output,
            original_masks=original_masks,
            timestamps=timestamps,
            patch_size=4,
        )

        # Check that output has per-modality structure
        assert output.sentinel2_l2a is not None
        assert output.sentinel2_l2a.shape == (B, H, W, T, 3, D)
        assert output.sentinel2_l2a_mask is not None
        assert output.latlon is not None


class TestEncoderWithChnAttn:
    """Test Encoder with ChnAttn enabled."""

    def test_encoder_with_chnattn_all_mode(
        self, supported_modalities: list[ModalitySpec]
    ) -> None:
        """Test encoder forward with ChnAttn in 'all' mode."""
        D = 16
        chnattn_config = ChnAttnConfig(
            enabled=True, attn_dim=D, num_heads=2, aggregation_mode="all"
        )
        encoder = Encoder(
            embedding_size=D,
            max_patch_size=8,
            min_patch_size=1,
            num_heads=2,
            mlp_ratio=4.0,
            depth=1,
            drop_path=0.0,
            supported_modalities=supported_modalities,
            max_sequence_length=12,
            chnattn_config=chnattn_config,
        )

        # Create a minimal MaskedOlmoEarthSample
        B, H, W, T = 2, 8, 8, 2
        s2_bands = Modality.SENTINEL2_L2A.num_bands
        sample = MaskedOlmoEarthSample(
            sentinel2_l2a=torch.randn(B, H, W, T, s2_bands),
            sentinel2_l2a_mask=torch.full(
                (B, H, W, T, s2_bands),
                MaskValue.ONLINE_ENCODER.value,
                dtype=torch.long,
            ),
            latlon=torch.randn(B, 2),
            latlon_mask=torch.full(
                (B, 2), MaskValue.ONLINE_ENCODER.value, dtype=torch.long
            ),
            timestamps=torch.randint(0, 12, (B, T, 3), dtype=torch.long),
        )

        output_dict = encoder(sample, patch_size=8)
        output = output_dict["tokens_and_masks"]

        # With ChnAttn, output should have aggregated field
        assert output.aggregated is not None
        assert output.aggregated_mask is not None
        # Should have original_masks
        assert "original_masks" in output_dict

    def test_encoder_skip_chnattn(
        self, supported_modalities: list[ModalitySpec]
    ) -> None:
        """Test encoder with skip_chnattn=True produces per-modality output."""
        D = 16
        chnattn_config = ChnAttnConfig(
            enabled=True, attn_dim=D, num_heads=2, aggregation_mode="all"
        )
        encoder = Encoder(
            embedding_size=D,
            max_patch_size=8,
            min_patch_size=1,
            num_heads=2,
            mlp_ratio=4.0,
            depth=1,
            drop_path=0.0,
            supported_modalities=supported_modalities,
            max_sequence_length=12,
            chnattn_config=chnattn_config,
        )

        B, H, W, T = 2, 8, 8, 2
        s2_bands = Modality.SENTINEL2_L2A.num_bands
        sample = MaskedOlmoEarthSample(
            sentinel2_l2a=torch.randn(B, H, W, T, s2_bands),
            sentinel2_l2a_mask=torch.full(
                (B, H, W, T, s2_bands),
                MaskValue.ONLINE_ENCODER.value,
                dtype=torch.long,
            ),
            latlon=torch.randn(B, 2),
            latlon_mask=torch.full(
                (B, 2), MaskValue.ONLINE_ENCODER.value, dtype=torch.long
            ),
            timestamps=torch.randint(0, 12, (B, T, 3), dtype=torch.long),
        )

        output_dict = encoder(sample, patch_size=8, skip_chnattn=True)
        output = output_dict["tokens_and_masks"]

        # With skip_chnattn, should have per-modality output, not aggregated
        assert output.aggregated is None
        assert output.sentinel2_l2a is not None
        assert "original_masks" not in output_dict

    def test_encoder_without_chnattn_backward_compat(
        self, supported_modalities: list[ModalitySpec]
    ) -> None:
        """Test encoder without ChnAttn works as before."""
        D = 16
        encoder = Encoder(
            embedding_size=D,
            max_patch_size=8,
            min_patch_size=1,
            num_heads=2,
            mlp_ratio=4.0,
            depth=1,
            drop_path=0.0,
            supported_modalities=supported_modalities,
            max_sequence_length=12,
        )

        B, H, W, T = 2, 8, 8, 2
        s2_bands = Modality.SENTINEL2_L2A.num_bands
        sample = MaskedOlmoEarthSample(
            sentinel2_l2a=torch.randn(B, H, W, T, s2_bands),
            sentinel2_l2a_mask=torch.full(
                (B, H, W, T, s2_bands),
                MaskValue.ONLINE_ENCODER.value,
                dtype=torch.long,
            ),
            latlon=torch.randn(B, 2),
            latlon_mask=torch.full(
                (B, 2), MaskValue.ONLINE_ENCODER.value, dtype=torch.long
            ),
            timestamps=torch.randint(0, 12, (B, T, 3), dtype=torch.long),
        )

        output_dict = encoder(sample, patch_size=8)
        output = output_dict["tokens_and_masks"]

        # No ChnAttn -> per-modality output
        assert output.aggregated is None
        assert output.sentinel2_l2a is not None
        assert "original_masks" not in output_dict

    def test_encoder_config_with_chnattn(
        self, supported_modalities: list[ModalitySpec]
    ) -> None:
        """Test EncoderConfig with chnattn_config builds correctly."""
        modality_names = [m.name for m in supported_modalities]
        config = EncoderConfig(
            supported_modality_names=modality_names,
            embedding_size=16,
            chnattn_config=ChnAttnConfig(
                enabled=True, attn_dim=16, num_heads=2, aggregation_mode="all"
            ),
        )
        encoder = config.build()
        assert encoder.uses_chnattn
