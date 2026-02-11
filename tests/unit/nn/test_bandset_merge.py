"""Unit tests for bandset merge/unmerge modules."""

import pytest
import torch

from olmoearth_pretrain.data.constants import Modality, ModalitySpec
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue
from olmoearth_pretrain.nn.bandset_merge import BandsetMerge, BandsetUnmerge
from olmoearth_pretrain.nn.flexi_vit import (
    EncoderConfig,
    PredictorConfig,
)
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig


class TestBandsetMerge:
    """Tests for the BandsetMerge module."""

    def test_output_shape(self) -> None:
        """Merge should reduce bandset dim from num_bandsets to 1."""
        D, num_bs = 16, 3
        merge = BandsetMerge(D, num_bs)

        tokens = torch.randn(2, 4, 4, 3, num_bs, D)  # [B, H, W, T, bs, D]
        mask = torch.zeros(2, 4, 4, 3, num_bs, dtype=torch.long)

        merged_tokens, merged_mask = merge(tokens, mask)
        assert merged_tokens.shape == (2, 4, 4, 3, 1, D)
        assert merged_mask.shape == (2, 4, 4, 3, 1)

    def test_mean_pool_init(self) -> None:
        """After init, merge should approximate mean pooling."""
        D, num_bs = 8, 3
        merge = BandsetMerge(D, num_bs)

        # Create tokens where each bandset has distinct values
        tokens = torch.zeros(1, 1, 1, 1, num_bs, D)
        tokens[0, 0, 0, 0, 0] = torch.ones(D) * 3.0
        tokens[0, 0, 0, 0, 1] = torch.ones(D) * 6.0
        tokens[0, 0, 0, 0, 2] = torch.ones(D) * 9.0
        mask = torch.zeros(1, 1, 1, 1, num_bs, dtype=torch.long)

        merged, _ = merge(tokens, mask)
        expected_mean = torch.ones(D) * 6.0  # (3+6+9)/3
        torch.testing.assert_close(
            merged.squeeze(), expected_mean, atol=1e-5, rtol=1e-5
        )

    def test_mask_takes_first_bandset(self) -> None:
        """Merged mask should equal the first bandset's mask (uniform assumption)."""
        D, num_bs = 8, 3
        merge = BandsetMerge(D, num_bs)

        tokens = torch.randn(2, 2, 2, 1, num_bs, D)
        mask = torch.full((2, 2, 2, 1, num_bs), MaskValue.ONLINE_ENCODER.value)
        mask[0, 0, 0, 0, :] = MaskValue.DECODER.value

        _, merged_mask = merge(tokens, mask)
        assert merged_mask[0, 0, 0, 0, 0].item() == MaskValue.DECODER.value
        assert merged_mask[0, 1, 0, 0, 0].item() == MaskValue.ONLINE_ENCODER.value

    def test_gradient_flows(self) -> None:
        """Gradients should flow through the merge projection."""
        D, num_bs = 8, 3
        merge = BandsetMerge(D, num_bs)

        tokens = torch.randn(1, 2, 2, 1, num_bs, D, requires_grad=True)
        mask = torch.zeros(1, 2, 2, 1, num_bs, dtype=torch.long)

        merged, _ = merge(tokens, mask)
        loss = merged.sum()
        loss.backward()
        assert tokens.grad is not None
        assert merge.proj.weight.grad is not None


class TestBandsetUnmerge:
    """Tests for the BandsetUnmerge module."""

    def test_output_shape(self) -> None:
        """Unmerge should expand bandset dim from 1 to num_bandsets."""
        D, num_bs = 16, 3
        unmerge = BandsetUnmerge(D, num_bs)

        tokens = torch.randn(2, 4, 4, 3, 1, D)
        mask = torch.zeros(2, 4, 4, 3, 1, dtype=torch.long)

        unmerged_tokens, unmerged_mask = unmerge(tokens, mask)
        assert unmerged_tokens.shape == (2, 4, 4, 3, num_bs, D)
        assert unmerged_mask.shape == (2, 4, 4, 3, num_bs)

    def test_mask_expansion(self) -> None:
        """Unmerged mask should broadcast the single mask to all bandsets."""
        D, num_bs = 8, 3
        unmerge = BandsetUnmerge(D, num_bs)

        tokens = torch.randn(2, 2, 2, 1, 1, D)
        mask = torch.zeros(2, 2, 2, 1, 1, dtype=torch.long)
        mask[0, 0, 0, 0, 0] = MaskValue.DECODER.value

        _, unmerged_mask = unmerge(tokens, mask)
        # All bandsets at (0,0,0,0) should be DECODER
        for bs in range(num_bs):
            assert unmerged_mask[0, 0, 0, 0, bs].item() == MaskValue.DECODER.value
        # All bandsets at (0,1,0,0) should be ONLINE_ENCODER (0)
        for bs in range(num_bs):
            assert (
                unmerged_mask[0, 1, 0, 0, bs].item() == MaskValue.ONLINE_ENCODER.value
            )

    def test_gradient_flows(self) -> None:
        """Gradients should flow through the unmerge projection."""
        D, num_bs = 8, 3
        unmerge = BandsetUnmerge(D, num_bs)

        tokens = torch.randn(1, 2, 2, 1, 1, D, requires_grad=True)
        mask = torch.zeros(1, 2, 2, 1, 1, dtype=torch.long)

        unmerged, _ = unmerge(tokens, mask)
        loss = unmerged.sum()
        loss.backward()
        assert tokens.grad is not None
        assert unmerge.proj.weight.grad is not None


class TestMergeUnmergeRoundTrip:
    """Test merge + unmerge preserve shapes."""

    def test_roundtrip_shapes(self) -> None:
        """Merge then unmerge should restore the original shape."""
        D, num_bs = 16, 3
        merge = BandsetMerge(D, num_bs)
        unmerge = BandsetUnmerge(D, num_bs)

        tokens = torch.randn(2, 4, 4, 3, num_bs, D)
        mask = torch.zeros(2, 4, 4, 3, num_bs, dtype=torch.long)

        merged_tokens, merged_mask = merge(tokens, mask)
        restored_tokens, restored_mask = unmerge(merged_tokens, merged_mask)

        assert restored_tokens.shape == tokens.shape
        assert restored_mask.shape == mask.shape


class TestEncoderWithMerge:
    """Test Encoder with merge_bandsets=True."""

    @pytest.fixture
    def supported_modalities(self) -> list[ModalitySpec]:
        """Create a list of supported modalities for testing."""
        return [Modality.SENTINEL2_L2A, Modality.LATLON]

    def test_encoder_config_build(
        self, supported_modalities: list[ModalitySpec]
    ) -> None:
        """EncoderConfig with merge_bandsets=True should build successfully."""
        names = [m.name for m in supported_modalities]
        config = EncoderConfig(
            supported_modality_names=names,
            embedding_size=16,
            max_patch_size=8,
            num_heads=2,
            depth=2,
            mlp_ratio=2.0,
            merge_bandsets=True,
        )
        encoder = config.build()
        assert encoder.merge_enabled is True
        assert encoder.bandset_merge_modules is not None
        assert "sentinel2_l2a" in encoder.bandset_merge_modules

    def test_encoder_config_build_no_merge(
        self, supported_modalities: list[ModalitySpec]
    ) -> None:
        """EncoderConfig with merge_bandsets=False should have no merge modules."""
        names = [m.name for m in supported_modalities]
        config = EncoderConfig(
            supported_modality_names=names,
            embedding_size=16,
            max_patch_size=8,
            num_heads=2,
            depth=2,
            mlp_ratio=2.0,
            merge_bandsets=False,
        )
        encoder = config.build()
        assert encoder.merge_enabled is False
        assert encoder.bandset_merge_modules is None

    def test_encoder_forward_merged_output_shape(
        self, supported_modalities: list[ModalitySpec]
    ) -> None:
        """Encoder output should have 1 bandset for S2 when merge is enabled."""
        B, H, W, T = 2, 16, 16, 2
        names = [m.name for m in supported_modalities]
        config = EncoderConfig(
            supported_modality_names=names,
            embedding_size=16,
            max_patch_size=8,
            num_heads=2,
            depth=2,
            mlp_ratio=2.0,
            merge_bandsets=True,
        )
        encoder = config.build()

        # Build input
        s2_data = torch.randn(B, H, W, T, 12)
        s2_mask = torch.zeros(B, H, W, T, 3, dtype=torch.long)
        # Uniform mask: all bandsets same value
        s2_mask[:, :8, :8, :, :] = MaskValue.ONLINE_ENCODER.value
        s2_mask[:, 8:, 8:, :, :] = MaskValue.DECODER.value

        latlon = torch.randn(B, 2)
        latlon_mask = torch.zeros(B, 1, dtype=torch.long)

        timestamps = (
            torch.tensor([[15, 1, 2023], [15, 6, 2023]], dtype=torch.long)
            .unsqueeze(0)
            .expand(B, -1, -1)
        )

        sample = MaskedOlmoEarthSample(
            sentinel2_l2a=s2_data,
            sentinel2_l2a_mask=s2_mask,
            latlon=latlon,
            latlon_mask=latlon_mask,
            timestamps=timestamps,
        )

        output = encoder(sample, patch_size=2)
        tokens_and_masks = output["tokens_and_masks"]

        # S2 should have 1 bandset (merged from 3)
        s2_out = tokens_and_masks.sentinel2_l2a
        assert s2_out is not None
        # bandset dim should be 1
        assert s2_out.shape[-2] == 1


class TestPredictorWithUnmerge:
    """Test Predictor with unmerge_bandsets=True."""

    @pytest.fixture
    def supported_modalities(self) -> list[ModalitySpec]:
        """Create a list of supported modalities for testing."""
        return [Modality.SENTINEL2_L2A, Modality.LATLON]

    def test_predictor_config_build(
        self, supported_modalities: list[ModalitySpec]
    ) -> None:
        """PredictorConfig with unmerge_bandsets=True should build successfully."""
        names = [m.name for m in supported_modalities]
        config = PredictorConfig(
            supported_modality_names=names,
            encoder_embedding_size=16,
            decoder_embedding_size=16,
            depth=2,
            num_heads=2,
            mlp_ratio=2.0,
            unmerge_bandsets=True,
        )
        decoder = config.build()
        assert decoder.bandset_unmerge_modules is not None
        assert "sentinel2_l2a" in decoder.bandset_unmerge_modules


class TestLatentMIMWithMerge:
    """Test LatentMIM with merge/unmerge."""

    @pytest.fixture
    def supported_modalities(self) -> list[ModalitySpec]:
        """Create a list of supported modalities for testing."""
        return [Modality.SENTINEL2_L2A, Modality.LATLON]

    def test_target_encoder_merge_disabled(
        self, supported_modalities: list[ModalitySpec]
    ) -> None:
        """Target encoder should have merge_enabled=False."""
        names = [m.name for m in supported_modalities]
        config = LatentMIMConfig(
            encoder_config=EncoderConfig(
                supported_modality_names=names,
                embedding_size=16,
                max_patch_size=8,
                num_heads=2,
                depth=2,
                mlp_ratio=2.0,
                merge_bandsets=True,
            ),
            decoder_config=PredictorConfig(
                supported_modality_names=names,
                encoder_embedding_size=16,
                decoder_embedding_size=16,
                depth=2,
                num_heads=2,
                mlp_ratio=2.0,
                unmerge_bandsets=True,
            ),
        )
        model = config.build()

        # Online encoder should merge
        assert model.encoder.merge_enabled is True
        # Target encoder should NOT merge
        assert model.target_encoder.merge_enabled is False
        # But target encoder should still have the modules (for EMA alignment)
        assert model.target_encoder.bandset_merge_modules is not None

    def test_param_count_alignment(
        self, supported_modalities: list[ModalitySpec]
    ) -> None:
        """Online and target encoder should have same number of parameters (for EMA)."""
        names = [m.name for m in supported_modalities]
        config = LatentMIMConfig(
            encoder_config=EncoderConfig(
                supported_modality_names=names,
                embedding_size=16,
                max_patch_size=8,
                num_heads=2,
                depth=2,
                mlp_ratio=2.0,
                merge_bandsets=True,
            ),
            decoder_config=PredictorConfig(
                supported_modality_names=names,
                encoder_embedding_size=16,
                decoder_embedding_size=16,
                depth=2,
                num_heads=2,
                mlp_ratio=2.0,
                unmerge_bandsets=True,
            ),
        )
        model = config.build()

        enc_params = list(model.encoder.parameters())
        target_params = list(model.target_encoder.parameters())
        assert len(enc_params) == len(target_params), (
            f"Parameter count mismatch: encoder={len(enc_params)}, "
            f"target={len(target_params)}"
        )
