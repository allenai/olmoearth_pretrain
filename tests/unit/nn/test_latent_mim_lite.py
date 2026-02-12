"""Unit tests for LatentMIMLITE."""

import pytest
import torch

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample
from olmoearth_pretrain.nn.flexi_vit import EncoderConfig, PredictorConfig
from olmoearth_pretrain.nn.latent_mim_lite import LatentMIMLITE, LatentMIMLITEConfig
from olmoearth_pretrain.train.masking import MaskValue

MODALITIES = [
    Modality.SENTINEL2_L2A.name,
    Modality.LATLON.name,
    Modality.WORLDCOVER.name,
]
MAX_SEQ_LEN = 12
MAX_PATCH_SIZE = 8


def _make_config(
    encoder_embed: int = 16,
    encoder_depth: int = 2,
    encoder_heads: int = 2,
    target_embed: int = 16,
    target_depth: int = 1,
    target_heads: int = 2,
    decoder_embed: int = 16,
    decoder_depth: int = 1,
    decoder_heads: int = 2,
) -> LatentMIMLITEConfig:
    """Helper to build a LatentMIMLITEConfig with small sizes for testing."""
    encoder_config = EncoderConfig(
        embedding_size=encoder_embed,
        num_heads=encoder_heads,
        depth=encoder_depth,
        mlp_ratio=1.0,
        supported_modality_names=MODALITIES,
        max_patch_size=MAX_PATCH_SIZE,
        drop_path=0.0,
        max_sequence_length=MAX_SEQ_LEN,
    )
    target_encoder_config = EncoderConfig(
        embedding_size=target_embed,
        num_heads=target_heads,
        depth=target_depth,
        mlp_ratio=1.0,
        supported_modality_names=MODALITIES,
        max_patch_size=MAX_PATCH_SIZE,
        drop_path=0.0,
        max_sequence_length=MAX_SEQ_LEN,
    )
    decoder_config = PredictorConfig(
        encoder_embedding_size=encoder_embed,
        decoder_embedding_size=decoder_embed,
        depth=decoder_depth,
        mlp_ratio=1.0,
        num_heads=decoder_heads,
        supported_modality_names=MODALITIES,
        max_sequence_length=MAX_SEQ_LEN,
        output_embedding_size=target_embed,
    )
    return LatentMIMLITEConfig(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        target_encoder_config=target_encoder_config,
    )


def _make_masked_batch(batch_size: int = 2) -> MaskedOlmoEarthSample:
    """Create a minimal masked batch for testing.

    Follows the pattern from conftest.py: S2 tokens are ONLINE_ENCODER,
    latlon and worldcover are DECODER.
    """
    H, W, T = 4, 4, 2
    s2_bands = Modality.SENTINEL2_L2A.num_bands
    s2 = torch.randn(batch_size, H, W, T, s2_bands)
    s2_mask = torch.full(
        (batch_size, H, W, T, s2_bands),
        fill_value=MaskValue.ONLINE_ENCODER.value,
        dtype=torch.long,
    )

    latlon = torch.randn(batch_size, 2)
    latlon_mask = torch.full(
        (batch_size, 2),
        fill_value=MaskValue.DECODER.value,
        dtype=torch.float32,
    )

    worldcover = torch.randn(batch_size, H, W, 1, 1)
    worldcover_mask = torch.full(
        (batch_size, H, W, 1, 1),
        fill_value=MaskValue.DECODER.value,
        dtype=torch.float32,
    )

    days = torch.randint(0, 25, (batch_size, T, 1), dtype=torch.long)
    months = torch.randint(0, 12, (batch_size, T, 1), dtype=torch.long)
    years = torch.randint(2018, 2020, (batch_size, T, 1), dtype=torch.long)
    timestamps = torch.cat([days, months, years], dim=-1)

    return MaskedOlmoEarthSample(
        sentinel2_l2a=s2,
        sentinel2_l2a_mask=s2_mask,
        latlon=latlon,
        latlon_mask=latlon_mask,
        worldcover=worldcover,
        worldcover_mask=worldcover_mask,
        timestamps=timestamps,
    )


class TestLatentMIMLITEConfig:
    """Tests for LatentMIMLITEConfig validation."""

    def test_valid_config_builds(self) -> None:
        """A valid config should build without errors."""
        config = _make_config()
        model = config.build()
        assert isinstance(model, LatentMIMLITE)

    def test_mismatched_output_embedding_raises(self) -> None:
        """Decoder output_embedding_size must match target encoder embedding_size."""
        config = _make_config(target_embed=32)
        # Manually break the alignment
        config.decoder_config.output_embedding_size = 999
        with pytest.raises(ValueError, match="output_embedding_size"):
            config.build()

    def test_mismatched_encoder_decoder_embedding_raises(self) -> None:
        """Encoder embedding_size must match decoder encoder_embedding_size."""
        config = _make_config(encoder_embed=64)
        config.decoder_config.encoder_embedding_size = 999
        with pytest.raises(ValueError, match="Encoder embedding size"):
            config.build()

    def test_different_encoder_and_target_sizes(self) -> None:
        """Encoder and target encoder can have different embedding sizes."""
        config = _make_config(
            encoder_embed=32, encoder_heads=2,
            target_embed=16, target_heads=2,
            decoder_embed=16, decoder_heads=2,
        )
        model = config.build()
        assert model.encoder.embedding_size == 32
        assert model.target_encoder.embedding_size == 16


class TestLatentMIMLITE:
    """Tests for LatentMIMLITE model."""

    def test_target_encoder_frozen(self) -> None:
        """Target encoder params should have requires_grad=False."""
        model = _make_config().build()
        for p in model.target_encoder.parameters():
            assert not p.requires_grad

    def test_encoder_has_trainable_params(self) -> None:
        """Online encoder should have at least some trainable params."""
        model = _make_config().build()
        trainable = [p for p in model.encoder.parameters() if p.requires_grad]
        assert len(trainable) > 0

    def test_decoder_has_trainable_params(self) -> None:
        """Decoder should have at least some trainable params."""
        model = _make_config().build()
        trainable = [p for p in model.decoder.parameters() if p.requires_grad]
        assert len(trainable) > 0

    def test_forward_returns_correct_tuple(self) -> None:
        """Forward should return (latent, decoded, pooled, extra_metrics)."""
        model = _make_config().build()
        batch = _make_masked_batch()
        result = model(batch, patch_size=4)
        assert len(result) == 4
        latent, decoded, pooled, extra_metrics = result
        assert isinstance(extra_metrics, dict)

    def test_forward_with_different_architectures(self) -> None:
        """Forward pass works when encoder and target have different sizes."""
        model = _make_config(
            encoder_embed=32, encoder_depth=2, encoder_heads=2,
            target_embed=16, target_depth=1, target_heads=2,
            decoder_embed=16, decoder_depth=1, decoder_heads=2,
        ).build()
        batch = _make_masked_batch()

        # Online path (encoder + decoder)
        latent, decoded, pooled, _ = model(batch, patch_size=4)
        assert pooled is not None

        # Target encoder path (separate forward)
        with torch.no_grad():
            output_dict = model.target_encoder.forward(
                batch.unmask(),
                patch_size=4,
                token_exit_cfg={
                    m: 0 for m in model.target_encoder.supported_modality_names
                },
            )
        assert "sentinel2_l2a" in output_dict or len(output_dict) > 0

    def test_no_reconstructor_attribute(self) -> None:
        """LatentMIMLITE should not have a reconstructor."""
        model = _make_config().build()
        assert not hasattr(model, "reconstructor")
