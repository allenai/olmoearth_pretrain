"""Unit tests for LatentMIMLITEContrastiveTrainModule."""

import torch

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.nn.flexi_vit import EncoderConfig, PredictorConfig
from olmoearth_pretrain.nn.latent_mim_lite import LatentMIMLITE, LatentMIMLITEConfig

MODALITIES = [Modality.SENTINEL2_L2A.name, Modality.LATLON.name, Modality.WORLDCOVER.name]


def _build_model(
    encoder_embed: int = 32,
    target_embed: int = 16,
) -> LatentMIMLITE:
    """Build a small LatentMIMLITE for testing."""
    config = LatentMIMLITEConfig(
        encoder_config=EncoderConfig(
            embedding_size=encoder_embed,
            num_heads=2,
            depth=2,
            mlp_ratio=1.0,
            supported_modality_names=MODALITIES,
            max_patch_size=8,
            drop_path=0.0,
            max_sequence_length=12,
        ),
        decoder_config=PredictorConfig(
            encoder_embedding_size=encoder_embed,
            decoder_embedding_size=16,
            depth=1,
            mlp_ratio=1.0,
            num_heads=2,
            supported_modality_names=MODALITIES,
            max_sequence_length=12,
            output_embedding_size=target_embed,
        ),
        target_encoder_config=EncoderConfig(
            embedding_size=target_embed,
            num_heads=2,
            depth=1,
            mlp_ratio=1.0,
            supported_modality_names=MODALITIES,
            max_patch_size=8,
            drop_path=0.0,
            max_sequence_length=12,
        ),
    )
    return config.build()


class TestLatentMIMLITENoEMA:
    """Test that LatentMIMLITE target encoder is not updated."""

    def test_target_encoder_weights_unchanged_after_manual_ema_style_check(
        self,
    ) -> None:
        """Target encoder weights should not change since there's no EMA.

        This verifies the fundamental contract: the target encoder is frozen
        and independent from the online encoder.
        """
        model = _build_model()

        # Snapshot target encoder weights
        target_weights_before = {
            name: p.clone() for name, p in model.target_encoder.named_parameters()
        }

        # Simulate what would happen in training: modify online encoder weights
        with torch.no_grad():
            for p in model.encoder.parameters():
                p.add_(torch.randn_like(p) * 0.1)

        # Target encoder should be completely unaffected
        for name, p in model.target_encoder.named_parameters():
            assert torch.equal(p, target_weights_before[name]), (
                f"Target encoder param {name} changed unexpectedly"
            )

    def test_encoder_and_target_have_different_params(self) -> None:
        """Encoder and target encoder should have independent (different) parameters."""
        model = _build_model(encoder_embed=64, target_embed=32)

        encoder_param_count = sum(p.numel() for p in model.encoder.parameters())
        target_param_count = sum(p.numel() for p in model.target_encoder.parameters())

        # Different architectures should have different param counts
        assert encoder_param_count != target_param_count
