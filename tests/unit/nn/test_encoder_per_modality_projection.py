"""Unit tests for EncoderWithPerModalityProjection."""

import torch

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.nn.flexi_vit import (
    EncoderWithPerModalityProjection,
    EncoderWithPerModalityProjectionConfig,
)


class TestEncoderWithPerModalityProjection:
    """Test the encoder with per-modality projections."""

    def test_encoder_initialization(self) -> None:
        """Test that encoder initializes with per-modality transforms."""
        D = 16
        encoder = EncoderWithPerModalityProjection(
            embedding_size=D,
            max_patch_size=8,
            min_patch_size=1,
            num_heads=2,
            mlp_ratio=4.0,
            depth=2,
            drop_path=0.0,
            supported_modalities=[Modality.SENTINEL2_L2A, Modality.SENTINEL1],
            max_sequence_length=12,
        )

        # Check that per-modality transforms exist
        assert hasattr(encoder, "per_modality_transforms")
        assert len(encoder.per_modality_transforms) == 2
        assert "sentinel2_l2a" in encoder.per_modality_transforms
        assert "sentinel1" in encoder.per_modality_transforms

        # Check that each transform is a Linear layer with correct dimensions
        for modality in ["sentinel2_l2a", "sentinel1"]:
            transform = encoder.per_modality_transforms[modality]
            assert isinstance(transform, torch.nn.Linear)
            assert transform.in_features == D
            assert transform.out_features == D

        # Check that project_and_aggregate still exists (shared projection unchanged)
        assert hasattr(encoder, "project_and_aggregate")

    def test_encoder_config_builds_correctly(self) -> None:
        """Test that the config builds the right encoder class."""
        supported_modality_names = ["sentinel2_l2a", "sentinel1"]

        config = EncoderWithPerModalityProjectionConfig(
            supported_modality_names=supported_modality_names,
            embedding_size=16,
            max_patch_size=8,
            min_patch_size=1,
            num_heads=2,
            depth=2,
        )

        encoder = config.build()
        assert isinstance(encoder, EncoderWithPerModalityProjection)
        assert hasattr(encoder, "per_modality_transforms")
        assert hasattr(encoder, "project_and_aggregate")

    def test_per_modality_transforms_are_learnable(self) -> None:
        """Test that per-modality transforms have parameters and can compute gradients."""
        D = 16
        encoder = EncoderWithPerModalityProjection(
            embedding_size=D,
            max_patch_size=8,
            min_patch_size=1,
            num_heads=2,
            mlp_ratio=4.0,
            depth=2,
            drop_path=0.0,
            supported_modalities=[Modality.SENTINEL2_L2A, Modality.SENTINEL1],
            max_sequence_length=12,
        )

        # Test that transforms have parameters
        for modality in ["sentinel2_l2a", "sentinel1"]:
            transform = encoder.per_modality_transforms[modality]
            assert transform.weight.requires_grad
            assert transform.bias.requires_grad

        # Test that gradients can be computed
        B, H, W, T, C = 2, 4, 4, 3, 3
        sentinel2_tokens = torch.randn(B, H, W, T, C, D, requires_grad=True)

        # Apply the transform
        transformed = encoder.per_modality_transforms["sentinel2_l2a"](sentinel2_tokens)
        loss = transformed.sum()
        loss.backward()

        # Check that gradients exist
        assert encoder.per_modality_transforms["sentinel2_l2a"].weight.grad is not None
        assert encoder.per_modality_transforms["sentinel2_l2a"].bias.grad is not None
        assert sentinel2_tokens.grad is not None

    def test_per_modality_transforms_produce_different_outputs(self) -> None:
        """Test that different modalities get different transformations."""
        D = 16
        encoder = EncoderWithPerModalityProjection(
            embedding_size=D,
            max_patch_size=8,
            min_patch_size=1,
            num_heads=2,
            mlp_ratio=4.0,
            depth=2,
            drop_path=0.0,
            supported_modalities=[Modality.SENTINEL2_L2A, Modality.SENTINEL1],
            max_sequence_length=12,
        )

        # Same input to both transforms
        B, H, W, T, C = 2, 4, 4, 3, 2
        input_tokens = torch.randn(B, H, W, T, C, D)

        # Apply both transforms
        s2_output = encoder.per_modality_transforms["sentinel2_l2a"](input_tokens)
        s1_output = encoder.per_modality_transforms["sentinel1"](input_tokens)

        # Outputs should be different (unless by extremely rare chance)
        assert not torch.allclose(s2_output, s1_output)

        # But shapes should match
        assert s2_output.shape == s1_output.shape == input_tokens.shape
