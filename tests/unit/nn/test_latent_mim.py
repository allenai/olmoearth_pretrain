"""Unit tests for the latent_mim module."""

import pytest
import torch

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue
from olmoearth_pretrain.nn.flexi_vit import EncoderConfig, PredictorConfig
from olmoearth_pretrain.nn.latent_mim import LatentMIM, LatentMIMConfig
from olmoearth_pretrain.nn.retokenizing_predictor import RetokenizingPredictorConfig
from olmoearth_pretrain.nn.tokenization import (
    ModalityTokenization,
    TokenizationConfig,
)


def _collapsed_tokenization() -> TokenizationConfig:
    return TokenizationConfig(
        overrides={
            "sentinel2_l2a": ModalityTokenization(
                band_groups=[
                    [
                        "B02",
                        "B03",
                        "B04",
                        "B08",
                        "B05",
                        "B06",
                        "B07",
                        "B8A",
                        "B11",
                        "B12",
                        "B01",
                        "B09",
                    ],
                ]
            ),
        }
    )


def _default_tokenization() -> TokenizationConfig:
    return TokenizationConfig()


class TestLatentMIMWithSeparateTargetEncoder:
    """Tests for LatentMIM with a separate target encoder."""

    @pytest.fixture
    def supported_modality_names(self) -> list[str]:
        """Supported modality names for testing."""
        return [Modality.SENTINEL2_L2A.name, Modality.LATLON.name]

    @pytest.fixture
    def embedding_size(self) -> int:
        """Embedding size for testing."""
        return 8

    def test_has_separate_target_encoder(
        self, supported_modality_names: list[str], embedding_size: int
    ) -> None:
        """Test that has_separate_target_encoder is set correctly."""
        encoder_config = EncoderConfig(
            supported_modality_names=supported_modality_names,
            embedding_size=embedding_size,
            num_heads=2,
            depth=1,
            tokenization_config=_collapsed_tokenization(),
        )
        target_encoder_config = EncoderConfig(
            supported_modality_names=supported_modality_names,
            embedding_size=embedding_size,
            num_heads=2,
            depth=1,
            tokenization_config=_default_tokenization(),
        )
        decoder_config = RetokenizingPredictorConfig(
            predictor_config=PredictorConfig(
                supported_modality_names=supported_modality_names,
                encoder_embedding_size=embedding_size,
                decoder_embedding_size=embedding_size,
                depth=1,
                num_heads=2,
                max_sequence_length=12,
                tokenization_config=_default_tokenization(),
            ),
            input_tokenization_config=_collapsed_tokenization(),
        )

        config = LatentMIMConfig(
            encoder_config=encoder_config,
            decoder_config=decoder_config,
            target_encoder_config=target_encoder_config,
        )
        model = config.build()
        assert model.has_separate_target_encoder is True

    def test_no_separate_target_encoder(
        self, supported_modality_names: list[str], embedding_size: int
    ) -> None:
        """Test default behavior: target encoder is a deepcopy."""
        config = LatentMIMConfig(
            encoder_config=EncoderConfig(
                supported_modality_names=supported_modality_names,
                embedding_size=embedding_size,
                num_heads=2,
                depth=1,
            ),
            decoder_config=PredictorConfig(
                supported_modality_names=supported_modality_names,
                encoder_embedding_size=embedding_size,
                decoder_embedding_size=embedding_size,
                depth=1,
                num_heads=2,
                max_sequence_length=12,
            ),
        )
        model = config.build()
        assert model.has_separate_target_encoder is False

    def test_target_encoder_params_frozen(
        self, supported_modality_names: list[str], embedding_size: int
    ) -> None:
        """Test target encoder parameters are frozen."""
        encoder_config = EncoderConfig(
            supported_modality_names=supported_modality_names,
            embedding_size=embedding_size,
            num_heads=2,
            depth=1,
            tokenization_config=_collapsed_tokenization(),
        )
        target_encoder_config = EncoderConfig(
            supported_modality_names=supported_modality_names,
            embedding_size=embedding_size,
            num_heads=2,
            depth=1,
        )
        decoder_config = PredictorConfig(
            supported_modality_names=supported_modality_names,
            encoder_embedding_size=embedding_size,
            decoder_embedding_size=embedding_size,
            depth=1,
            num_heads=2,
            max_sequence_length=12,
        )
        config = LatentMIMConfig(
            encoder_config=encoder_config,
            decoder_config=decoder_config,
            target_encoder_config=target_encoder_config,
        )
        model = config.build()
        for p in model.target_encoder.parameters():
            assert not p.requires_grad


class TestPrepareTargetBatch:
    """Tests for the prepare_target_batch method."""

    @pytest.fixture
    def model_with_separate_target(self) -> LatentMIM:
        """Build a LatentMIM with separate target encoder."""
        modality_names = [Modality.SENTINEL2_L2A.name, Modality.LATLON.name]
        emb = 8
        config = LatentMIMConfig(
            encoder_config=EncoderConfig(
                supported_modality_names=modality_names,
                embedding_size=emb,
                num_heads=2,
                depth=1,
                tokenization_config=_collapsed_tokenization(),
            ),
            decoder_config=RetokenizingPredictorConfig(
                predictor_config=PredictorConfig(
                    supported_modality_names=modality_names,
                    encoder_embedding_size=emb,
                    decoder_embedding_size=emb,
                    depth=1,
                    num_heads=2,
                    max_sequence_length=12,
                    tokenization_config=_default_tokenization(),
                ),
                input_tokenization_config=_collapsed_tokenization(),
            ),
            target_encoder_config=EncoderConfig(
                supported_modality_names=modality_names,
                embedding_size=emb,
                num_heads=2,
                depth=1,
            ),
        )
        return config.build()

    @pytest.fixture
    def model_without_separate_target(self) -> LatentMIM:
        """Build a default LatentMIM (deepcopy target)."""
        modality_names = [Modality.SENTINEL2_L2A.name, Modality.LATLON.name]
        emb = 8
        config = LatentMIMConfig(
            encoder_config=EncoderConfig(
                supported_modality_names=modality_names,
                embedding_size=emb,
                num_heads=2,
                depth=1,
            ),
            decoder_config=PredictorConfig(
                supported_modality_names=modality_names,
                encoder_embedding_size=emb,
                decoder_embedding_size=emb,
                depth=1,
                num_heads=2,
                max_sequence_length=12,
            ),
        )
        return config.build()

    def test_prepare_target_batch_adapts_mask_shape(
        self, model_with_separate_target: LatentMIM
    ) -> None:
        """Test masks are reshaped from collapsed (1) to default (3) bandsets."""
        B, H, W, T = 2, 8, 8, 2
        batch = MaskedOlmoEarthSample(
            timestamps=torch.randint(0, 30, (B, T, 3)),
            sentinel2_l2a=torch.randn(B, H, W, T, 12),
            sentinel2_l2a_mask=torch.full(
                (B, H, W, T, 1), MaskValue.DECODER.value, dtype=torch.float
            ),
            latlon=torch.randn(B, 2),
            latlon_mask=torch.full((B, 1), MaskValue.DECODER.value, dtype=torch.float),
        )
        target_batch = model_with_separate_target.prepare_target_batch(batch)

        # S2 mask should now have 3 bandsets (default)
        assert target_batch.sentinel2_l2a_mask is not None
        assert target_batch.sentinel2_l2a_mask.shape == (B, H, W, T, 3)
        # All should be ONLINE_ENCODER (unmasked)
        assert (target_batch.sentinel2_l2a_mask == MaskValue.ONLINE_ENCODER.value).all()

    def test_prepare_target_batch_preserves_missing(
        self, model_with_separate_target: LatentMIM
    ) -> None:
        """Test MISSING values are preserved and expanded."""
        B, H, W, T = 1, 4, 4, 1
        mask = torch.full((B, H, W, T, 1), MaskValue.DECODER.value, dtype=torch.float)
        # Mark one position as MISSING
        mask[0, 0, 0, 0, 0] = MaskValue.MISSING.value

        batch = MaskedOlmoEarthSample(
            timestamps=torch.randint(0, 30, (B, T, 3)),
            sentinel2_l2a=torch.randn(B, H, W, T, 12),
            sentinel2_l2a_mask=mask,
            latlon=torch.randn(B, 2),
            latlon_mask=torch.zeros(B, 1),
        )
        target_batch = model_with_separate_target.prepare_target_batch(batch)

        assert target_batch.sentinel2_l2a_mask is not None
        # All 3 bandsets at the MISSING position should be MISSING
        assert (
            target_batch.sentinel2_l2a_mask[0, 0, 0, 0, :] == MaskValue.MISSING.value
        ).all()
        # Other positions should be ONLINE_ENCODER
        assert (
            target_batch.sentinel2_l2a_mask[0, 1, 0, 0, :]
            == MaskValue.ONLINE_ENCODER.value
        ).all()

    def test_prepare_target_batch_same_tokenization(
        self, model_without_separate_target: LatentMIM
    ) -> None:
        """Test prepare_target_batch with same tokenization is equivalent to unmask."""
        B, H, W, T = 2, 4, 4, 1
        mask = torch.full((B, H, W, T, 3), MaskValue.DECODER.value, dtype=torch.float)
        mask[0, 0, 0, 0, 0] = MaskValue.MISSING.value

        batch = MaskedOlmoEarthSample(
            timestamps=torch.randint(0, 30, (B, T, 3)),
            sentinel2_l2a=torch.randn(B, H, W, T, 12),
            sentinel2_l2a_mask=mask,
            latlon=torch.randn(B, 2),
            latlon_mask=torch.zeros(B, 1),
        )

        target_batch = model_without_separate_target.prepare_target_batch(batch)
        expected = batch.unmask()

        assert torch.equal(target_batch.sentinel2_l2a_mask, expected.sentinel2_l2a_mask)

    def test_prepare_target_batch_raw_data_unchanged(
        self, model_with_separate_target: LatentMIM
    ) -> None:
        """Test that raw modality data is not modified."""
        B, H, W, T = 1, 4, 4, 1
        s2_data = torch.randn(B, H, W, T, 12)
        batch = MaskedOlmoEarthSample(
            timestamps=torch.randint(0, 30, (B, T, 3)),
            sentinel2_l2a=s2_data,
            sentinel2_l2a_mask=torch.zeros(B, H, W, T, 1),
            latlon=torch.randn(B, 2),
            latlon_mask=torch.zeros(B, 1),
        )
        target_batch = model_with_separate_target.prepare_target_batch(batch)

        # Raw data should be the same tensor
        assert torch.equal(target_batch.sentinel2_l2a, s2_data)


class TestLatentMIMConfigValidation:
    """Tests for LatentMIMConfig validation with target encoder."""

    def test_validate_mismatched_embedding_size(self) -> None:
        """Test validation catches mismatched embedding sizes."""
        modality_names = [Modality.SENTINEL2_L2A.name, Modality.LATLON.name]
        config = LatentMIMConfig(
            encoder_config=EncoderConfig(
                supported_modality_names=modality_names,
                embedding_size=16,
                num_heads=2,
                depth=1,
            ),
            decoder_config=PredictorConfig(
                supported_modality_names=modality_names,
                encoder_embedding_size=16,
                decoder_embedding_size=8,
                depth=1,
                num_heads=2,
                max_sequence_length=12,
            ),
            target_encoder_config=EncoderConfig(
                supported_modality_names=modality_names,
                embedding_size=32,  # Different from encoder
                num_heads=2,
                depth=1,
            ),
        )
        with pytest.raises(ValueError, match="same embedding size"):
            config.validate()

    def test_validate_mismatched_modalities(self) -> None:
        """Test validation catches mismatched modalities."""
        config = LatentMIMConfig(
            encoder_config=EncoderConfig(
                supported_modality_names=[
                    Modality.SENTINEL2_L2A.name,
                    Modality.LATLON.name,
                ],
                embedding_size=8,
                num_heads=2,
                depth=1,
            ),
            decoder_config=PredictorConfig(
                supported_modality_names=[
                    Modality.SENTINEL2_L2A.name,
                    Modality.LATLON.name,
                ],
                encoder_embedding_size=8,
                decoder_embedding_size=8,
                depth=1,
                num_heads=2,
                max_sequence_length=12,
            ),
            target_encoder_config=EncoderConfig(
                supported_modality_names=[
                    Modality.SENTINEL2_L2A.name
                ],  # Missing latlon
                embedding_size=8,
                num_heads=2,
                depth=1,
            ),
        )
        with pytest.raises(ValueError, match="same modalities"):
            config.validate()
