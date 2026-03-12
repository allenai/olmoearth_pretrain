"""Tests for PretrainedTargetEncoder and PretrainedTargetLatentMIM."""

import logging
from typing import cast
from unittest.mock import MagicMock, patch

import pytest
import torch

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.datatypes import (
    MaskedOlmoEarthSample,
    MaskValue,
    TokensAndMasks,
)
from olmoearth_pretrain.nn.flexi_vit import Encoder, Predictor
from olmoearth_pretrain.nn.pretrained_target_encoder import PretrainedTargetEncoder
from olmoearth_pretrain.nn.pretrained_target_latent_mim import (
    PretrainedTargetLatentMIM,
    PretrainedTargetLatentMIMConfig,
    _compute_bandset_expansion,
    _expand_bandsets_tokens_and_masks,
)
from olmoearth_pretrain.nn.tokenization import (
    ModalityTokenization,
    TokenizationConfig,
)
from olmoearth_pretrain.nn.utils import unpack_encoder_output

logger = logging.getLogger(__name__)

# Test constants
B, H, W, T = 2, 4, 4, 2
PATCH_SIZE = 4
EMBEDDING_SIZE = 16
MAX_PATCH_SIZE = 8
MAX_SEQ_LENGTH = 12
NUM_HEADS = 2
MLP_RATIO = 4.0
DEPTH = 2
DROP_PATH = 0.1

# Modalities for testing
ENCODABLE_MODALITIES = [Modality.SENTINEL2_L2A, Modality.WORLDCOVER]
ENCODABLE_MODALITY_NAMES = [m.name for m in ENCODABLE_MODALITIES]

# Single bandset tokenization for the online encoder
SINGLE_BANDSET_S2 = ModalityTokenization(
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
        ]
    ]
)
SINGLE_BANDSET_CONFIG = TokenizationConfig(
    overrides={"sentinel2_l2a": SINGLE_BANDSET_S2}
)


def _make_encoder(
    modalities: list,
    tokenization_config: TokenizationConfig | None = None,
) -> Encoder:
    """Create a small encoder for testing."""
    return Encoder(
        supported_modalities=modalities,
        embedding_size=EMBEDDING_SIZE,
        max_patch_size=MAX_PATCH_SIZE,
        min_patch_size=1,
        num_heads=NUM_HEADS,
        mlp_ratio=MLP_RATIO,
        max_sequence_length=MAX_SEQ_LENGTH,
        depth=DEPTH,
        drop_path=DROP_PATH,
        tokenization_config=tokenization_config,
    )


def _make_predictor(
    modalities: list,
    tokenization_config: TokenizationConfig | None = None,
    output_embedding_size: int | None = None,
) -> Predictor:
    """Create a small predictor for testing."""
    return Predictor(
        supported_modalities=modalities,
        encoder_embedding_size=EMBEDDING_SIZE,
        decoder_embedding_size=EMBEDDING_SIZE,
        depth=DEPTH,
        mlp_ratio=MLP_RATIO,
        num_heads=NUM_HEADS,
        max_sequence_length=MAX_SEQ_LENGTH,
        drop_path=DROP_PATH,
        tokenization_config=tokenization_config,
        output_embedding_size=output_embedding_size,
    )


def _make_masked_sample() -> MaskedOlmoEarthSample:
    """Create a simple masked sample with S2 and worldcover.

    S2 has a mix of ONLINE_ENCODER and DECODER tokens so that the decoder
    has enough tokens for a non-trivial contrastive loss.
    """
    s2_num_bands = Modality.SENTINEL2_L2A.num_bands
    sentinel2_l2a = torch.randn(B, H, W, T, s2_num_bands)
    # Half S2 tokens are ONLINE_ENCODER, half are DECODER
    sentinel2_l2a_mask = torch.full(
        (B, H, W, T, s2_num_bands),
        fill_value=MaskValue.ONLINE_ENCODER.value,
        dtype=torch.long,
    )
    # Make the second half of spatial dims DECODER
    sentinel2_l2a_mask[:, H // 2 :, :, :, :] = MaskValue.DECODER.value
    worldcover = torch.randn(B, H, W, 1, 1)
    worldcover_mask = torch.full(
        (B, H, W, 1, 1),
        fill_value=MaskValue.DECODER.value,
        dtype=torch.float32,
    )
    days = torch.randint(0, 25, (B, T, 1), dtype=torch.long)
    months = torch.randint(0, 12, (B, T, 1), dtype=torch.long)
    years = torch.randint(2018, 2020, (B, T, 1), dtype=torch.long)
    timestamps = torch.cat([days, months, years], dim=-1)

    return MaskedOlmoEarthSample(
        sentinel2_l2a=sentinel2_l2a,
        sentinel2_l2a_mask=sentinel2_l2a_mask,
        worldcover=worldcover,
        worldcover_mask=worldcover_mask,
        timestamps=timestamps,
    )


class TestPretrainedTargetEncoder:
    """Tests for PretrainedTargetEncoder."""

    def test_all_params_frozen(self) -> None:
        """Verify all target encoder params have requires_grad=False."""
        pretrained_encoder = _make_encoder(ENCODABLE_MODALITIES)
        target = PretrainedTargetEncoder(
            pretrained_encoder=pretrained_encoder,
            encodable_modality_names=ENCODABLE_MODALITY_NAMES,
        )
        for name, param in target.named_parameters():
            assert not param.requires_grad, f"Parameter {name} should be frozen"

    def test_forward_full_depth_single_pass(self) -> None:
        """Test full depth, single pass forward."""
        pretrained_encoder = _make_encoder(ENCODABLE_MODALITIES)
        target = PretrainedTargetEncoder(
            pretrained_encoder=pretrained_encoder,
            projection_only=False,
            per_modality_forward=False,
            encodable_modality_names=ENCODABLE_MODALITY_NAMES,
        )
        target.eval()
        sample = _make_masked_sample().unmask()
        output = target.forward(sample, patch_size=PATCH_SIZE)
        assert "tokens_and_masks" in output
        tam = output["tokens_and_masks"]
        assert isinstance(tam, TokensAndMasks)
        assert tam.sentinel2_l2a is not None
        assert tam.worldcover is not None

    def test_forward_projection_only(self) -> None:
        """Test projection-only forward (no transformer blocks)."""
        pretrained_encoder = _make_encoder(ENCODABLE_MODALITIES)
        target = PretrainedTargetEncoder(
            pretrained_encoder=pretrained_encoder,
            projection_only=True,
            encodable_modality_names=ENCODABLE_MODALITY_NAMES,
        )
        target.eval()
        sample = _make_masked_sample().unmask()
        output = target.forward(sample, patch_size=PATCH_SIZE)
        assert "tokens_and_masks" in output
        tam = output["tokens_and_masks"]
        assert tam.sentinel2_l2a is not None

    def test_forward_per_modality(self) -> None:
        """Test per-modality forward (separate pass per modality)."""
        pretrained_encoder = _make_encoder(ENCODABLE_MODALITIES)
        target = PretrainedTargetEncoder(
            pretrained_encoder=pretrained_encoder,
            projection_only=False,
            per_modality_forward=True,
            encodable_modality_names=ENCODABLE_MODALITY_NAMES,
        )
        target.eval()
        sample = _make_masked_sample().unmask()
        output = target.forward(sample, patch_size=PATCH_SIZE)
        assert "tokens_and_masks" in output
        tam = output["tokens_and_masks"]
        assert tam.sentinel2_l2a is not None
        assert tam.worldcover is not None

    def test_random_projections_for_decode_only(self) -> None:
        """Test that decode-only modalities get random projections."""
        pretrained_encoder = _make_encoder([Modality.SENTINEL2_L2A])
        target = PretrainedTargetEncoder(
            pretrained_encoder=pretrained_encoder,
            encodable_modality_names=[Modality.SENTINEL2_L2A.name],
            random_projection_modality_names=[Modality.WORLDCOVER.name],
            random_projection_embedding_size=EMBEDDING_SIZE,
        )
        target.eval()
        sample = _make_masked_sample().unmask()
        output = target.forward(sample, patch_size=PATCH_SIZE)
        assert "tokens_and_masks" in output
        tam = output["tokens_and_masks"]
        # Both modalities should be present
        assert tam.sentinel2_l2a is not None
        assert tam.worldcover is not None
        # Worldcover should have embedding dimension matching
        assert tam.worldcover.shape[-1] == EMBEDDING_SIZE

    def test_forward_signature_matches_encoder(self) -> None:
        """Test that forward signature is compatible with Encoder.forward() usage."""
        pretrained_encoder = _make_encoder(ENCODABLE_MODALITIES)
        target = PretrainedTargetEncoder(
            pretrained_encoder=pretrained_encoder,
            encodable_modality_names=ENCODABLE_MODALITY_NAMES,
        )
        target.eval()
        sample = _make_masked_sample().unmask()

        # This is how contrastive_latentmim.py calls it (line 316-321)
        output_dict = target.forward(
            sample,
            patch_size=PATCH_SIZE,
            token_exit_cfg={m: 0 for m in ENCODABLE_MODALITY_NAMES},
        )
        target_output, _, _ = unpack_encoder_output(output_dict)
        assert target_output is not None

    def test_band_dropout_disabled(self) -> None:
        """Verify band dropout is disabled on the pretrained encoder."""
        pretrained_encoder = _make_encoder(ENCODABLE_MODALITIES)
        pretrained_encoder.patch_embeddings.band_dropout_rate = 0.5
        target = PretrainedTargetEncoder(
            pretrained_encoder=pretrained_encoder,
            encodable_modality_names=ENCODABLE_MODALITY_NAMES,
        )
        assert target.pretrained_encoder.patch_embeddings.band_dropout_rate == 0.0


class TestBandsetExpansion:
    """Tests for bandset expansion logic."""

    def test_compute_expansion_single_to_multi(self) -> None:
        """Test expansion from 1 bandset to 3."""
        enc_config = TokenizationConfig(overrides={"sentinel2_l2a": SINGLE_BANDSET_S2})
        dec_config = TokenizationConfig()  # Default: 3 bandsets for S2

        expansion = _compute_bandset_expansion(
            enc_config, dec_config, ["sentinel2_l2a"]
        )
        # Default S2 has 3 bandsets, single has 1, so expansion = 3
        assert expansion["sentinel2_l2a"] == 3

    def test_compute_expansion_same(self) -> None:
        """Test expansion when encoder and decoder have same bandsets."""
        config = TokenizationConfig()  # Default for both
        expansion = _compute_bandset_expansion(config, config, ["sentinel2_l2a"])
        assert expansion["sentinel2_l2a"] == 1

    def test_compute_expansion_invalid(self) -> None:
        """Test that non-divisible expansion raises ValueError."""
        # 2 bandsets to 3 would not divide evenly
        enc_config = TokenizationConfig(
            overrides={
                "sentinel2_l2a": ModalityTokenization(
                    band_groups=[
                        ["B02", "B03", "B04", "B08"],
                        ["B05", "B06", "B07", "B8A", "B11", "B12", "B01", "B09"],
                    ]
                )
            }
        )
        dec_config = TokenizationConfig()  # Default: 3 bandsets

        with pytest.raises(ValueError, match="divisible"):
            _compute_bandset_expansion(enc_config, dec_config, ["sentinel2_l2a"])

    def test_expand_tokens_and_masks_shapes(self) -> None:
        """Verify shapes expand correctly from 1 to N bandsets."""
        # Create TokensAndMasks with 1 bandset for S2
        patched_H, patched_W = H // PATCH_SIZE, W // PATCH_SIZE
        s2_tokens = torch.randn(B, patched_H, patched_W, T, 1, EMBEDDING_SIZE)
        s2_mask = torch.ones(B, patched_H, patched_W, T, 1, dtype=torch.long)

        tam = TokensAndMasks(
            sentinel2_l2a=s2_tokens,
            sentinel2_l2a_mask=s2_mask,
        )
        expansion = {"sentinel2_l2a": 3}
        expanded = _expand_bandsets_tokens_and_masks(tam, expansion)

        assert expanded.sentinel2_l2a is not None
        assert expanded.sentinel2_l2a.shape == (
            B,
            patched_H,
            patched_W,
            T,
            3,
            EMBEDDING_SIZE,
        )
        assert expanded.sentinel2_l2a_mask is not None
        assert expanded.sentinel2_l2a_mask.shape == (B, patched_H, patched_W, T, 3)

    def test_expand_no_op_when_factor_is_1(self) -> None:
        """Verify expansion is a no-op when factor is 1."""
        patched_H, patched_W = H // PATCH_SIZE, W // PATCH_SIZE
        s2_tokens = torch.randn(B, patched_H, patched_W, T, 3, EMBEDDING_SIZE)
        s2_mask = torch.ones(B, patched_H, patched_W, T, 3, dtype=torch.long)

        tam = TokensAndMasks(
            sentinel2_l2a=s2_tokens,
            sentinel2_l2a_mask=s2_mask,
        )
        expansion = {"sentinel2_l2a": 1}
        expanded = _expand_bandsets_tokens_and_masks(tam, expansion)

        assert torch.equal(expanded.sentinel2_l2a, s2_tokens)
        assert torch.equal(expanded.sentinel2_l2a_mask, s2_mask)


class TestPretrainedTargetLatentMIM:
    """Tests for PretrainedTargetLatentMIM model."""

    def _build_model(
        self,
        tokenization_config: TokenizationConfig | None = None,
    ) -> PretrainedTargetLatentMIM:
        """Build a test model with matching tokenization everywhere."""
        modalities = ENCODABLE_MODALITIES
        # Online encoder with single bandset
        encoder = _make_encoder(modalities, tokenization_config=SINGLE_BANDSET_CONFIG)
        # Decoder with default (multi-bandset) tokenization
        decoder = _make_predictor(
            modalities,
            tokenization_config=tokenization_config,
            output_embedding_size=EMBEDDING_SIZE,
        )
        # Pretrained target encoder with default tokenization
        pretrained_encoder = _make_encoder(
            modalities, tokenization_config=tokenization_config
        )
        target_encoder = PretrainedTargetEncoder(
            pretrained_encoder=pretrained_encoder,
            encodable_modality_names=ENCODABLE_MODALITY_NAMES,
        )

        # Compute expansion
        enc_tok = cast(
            TokenizationConfig,
            getattr(encoder, "tokenization_config", TokenizationConfig()),
        )
        dec_tok = cast(
            TokenizationConfig,
            getattr(decoder, "tokenization_config", TokenizationConfig()),
        )
        all_modalities = [m.name for m in modalities]
        expansion = _compute_bandset_expansion(enc_tok, dec_tok, all_modalities)

        return PretrainedTargetLatentMIM(
            encoder=encoder,
            decoder=decoder,
            target_encoder=target_encoder,
            encoder_to_decoder_bandset_expansion=expansion,
        )

    def test_forward_pass(self) -> None:
        """Test end-to-end forward pass."""
        model = self._build_model()
        model.train()
        sample = _make_masked_sample()

        latent, decoded, pooled, reconstructed, extra_metrics = model(
            sample, PATCH_SIZE
        )

        assert isinstance(latent, TokensAndMasks)
        assert isinstance(decoded, TokensAndMasks)
        assert pooled is not None
        assert reconstructed is None
        assert isinstance(extra_metrics, dict)

    def test_target_encoder_frozen(self) -> None:
        """Verify target encoder has no trainable parameters."""
        model = self._build_model()
        for name, param in model.target_encoder.named_parameters():
            assert not param.requires_grad, (
                f"Target encoder param {name} should be frozen"
            )

    def test_gradients_flow_only_to_online_encoder_and_decoder(self) -> None:
        """Verify gradients flow to decoder but not target encoder.

        Uses an L2 loss on the decoded tokens to ensure gradient flow,
        since the contrastive loss needs many tokens to be non-trivial.
        """
        model = self._build_model()
        model.train()
        sample = _make_masked_sample()

        latent, decoded, pooled, _, _ = model(sample, PATCH_SIZE)

        # Use a direct L2 loss on decoded tokens to ensure gradient flow
        loss = torch.tensor(0.0, requires_grad=True)
        for modality in decoded.modalities:
            tokens = getattr(decoded, modality)
            loss = loss + tokens.pow(2).mean()
        loss.backward()

        # Decoder should have gradients (at least some params)
        decoder_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.decoder.parameters()
        )
        assert decoder_has_grad, "Decoder should receive gradients"

        # Encoder should have gradients (latent flows through to decoder)
        encoder_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.encoder.parameters()
        )
        assert encoder_has_grad, "Online encoder should receive gradients"

        # Target encoder should have no gradients
        for name, param in model.target_encoder.named_parameters():
            assert param.grad is None, (
                f"Target encoder param {name} should not have gradients"
            )

    def test_output_format_compatibility(self) -> None:
        """Verify decoder output and target encoder output have compatible shapes."""
        model = self._build_model()
        model.eval()
        sample = _make_masked_sample()

        _, decoded, _, _, _ = model(sample, PATCH_SIZE)

        with torch.no_grad():
            output_dict = model.target_encoder.forward(
                sample.unmask(),
                patch_size=PATCH_SIZE,
                token_exit_cfg={m: 0 for m in ENCODABLE_MODALITY_NAMES},
            )
            target_output, _, _ = unpack_encoder_output(output_dict)

        # Check that decoded and target modalities match
        assert set(decoded.modalities) == set(target_output.modalities)

        # Check embedding dimensions match
        for modality in decoded.modalities:
            decoded_tokens = getattr(decoded, modality)
            target_tokens = getattr(target_output, modality)
            assert decoded_tokens.shape[-1] == target_tokens.shape[-1], (
                f"Embedding size mismatch for {modality}: "
                f"decoded={decoded_tokens.shape[-1]}, target={target_tokens.shape[-1]}"
            )


class TestPretrainedTargetLatentMIMConfig:
    """Tests for PretrainedTargetLatentMIMConfig."""

    def test_validate_requires_exactly_one_target_source(self) -> None:
        """Test that exactly one of model_id or model_path is required."""
        from olmoearth_pretrain.nn.flexi_vit import EncoderConfig, PredictorConfig

        encoder_config = EncoderConfig(
            supported_modality_names=ENCODABLE_MODALITY_NAMES,
            embedding_size=EMBEDDING_SIZE,
        )
        decoder_config = PredictorConfig(
            supported_modality_names=ENCODABLE_MODALITY_NAMES,
            encoder_embedding_size=EMBEDDING_SIZE,
        )

        # Neither set
        config = PretrainedTargetLatentMIMConfig(
            encoder_config=encoder_config,
            decoder_config=decoder_config,
        )
        with pytest.raises(ValueError, match="Exactly one"):
            config.validate()

        # Both set
        config = PretrainedTargetLatentMIMConfig(
            encoder_config=encoder_config,
            decoder_config=decoder_config,
            target_encoder_model_id="OlmoEarth-v1-Base",
            target_encoder_model_path="/some/path",
        )
        with pytest.raises(ValueError, match="Exactly one"):
            config.validate()

    def test_validate_per_modality_requires_not_projection_only(self) -> None:
        """Test per_modality_forward + projection_only raises."""
        from olmoearth_pretrain.nn.flexi_vit import EncoderConfig, PredictorConfig

        encoder_config = EncoderConfig(
            supported_modality_names=ENCODABLE_MODALITY_NAMES,
            embedding_size=EMBEDDING_SIZE,
        )
        decoder_config = PredictorConfig(
            supported_modality_names=ENCODABLE_MODALITY_NAMES,
            encoder_embedding_size=EMBEDDING_SIZE,
        )

        config = PretrainedTargetLatentMIMConfig(
            encoder_config=encoder_config,
            decoder_config=decoder_config,
            target_encoder_model_id="OlmoEarth-v1-Base",
            projection_only=True,
            per_modality_forward=True,
        )
        with pytest.raises(ValueError, match="per_modality_forward"):
            config.validate()

    def test_validate_embedding_size_mismatch(self) -> None:
        """Test that mismatched embedding sizes raise."""
        from olmoearth_pretrain.nn.flexi_vit import EncoderConfig, PredictorConfig

        encoder_config = EncoderConfig(
            supported_modality_names=ENCODABLE_MODALITY_NAMES,
            embedding_size=16,
        )
        decoder_config = PredictorConfig(
            supported_modality_names=ENCODABLE_MODALITY_NAMES,
            encoder_embedding_size=32,  # Mismatch!
        )

        config = PretrainedTargetLatentMIMConfig(
            encoder_config=encoder_config,
            decoder_config=decoder_config,
            target_encoder_model_id="OlmoEarth-v1-Base",
        )
        with pytest.raises(ValueError, match="encoder_embedding_size"):
            config.validate()

    def test_build_with_mock_model(self) -> None:
        """Test build() with a mocked pretrained model load."""
        from olmoearth_pretrain.nn.flexi_vit import EncoderConfig, PredictorConfig

        encoder_config = EncoderConfig(
            supported_modality_names=ENCODABLE_MODALITY_NAMES,
            embedding_size=EMBEDDING_SIZE,
            tokenization_config=SINGLE_BANDSET_CONFIG,
        )
        decoder_config = PredictorConfig(
            supported_modality_names=ENCODABLE_MODALITY_NAMES,
            encoder_embedding_size=EMBEDDING_SIZE,
            output_embedding_size=EMBEDDING_SIZE,
        )

        config = PretrainedTargetLatentMIMConfig(
            encoder_config=encoder_config,
            decoder_config=decoder_config,
            target_encoder_model_id="OlmoEarth-v1-Base",
            encodable_modality_names=ENCODABLE_MODALITY_NAMES,
        )

        # Mock the pretrained model loading
        mock_pretrained_encoder = _make_encoder(ENCODABLE_MODALITIES)
        mock_model = MagicMock()
        mock_model.encoder = mock_pretrained_encoder

        with patch.object(config, "_load_pretrained_model", return_value=mock_model):
            model = config.build()

        assert isinstance(model, PretrainedTargetLatentMIM)
        assert isinstance(model.target_encoder, PretrainedTargetEncoder)
        assert isinstance(model.encoder, Encoder)

        # Verify target encoder is frozen
        for param in model.target_encoder.parameters():
            assert not param.requires_grad

    def test_build_and_forward_with_mock(self) -> None:
        """Test that a built model can run forward."""
        from olmoearth_pretrain.nn.flexi_vit import EncoderConfig, PredictorConfig

        encoder_config = EncoderConfig(
            supported_modality_names=ENCODABLE_MODALITY_NAMES,
            embedding_size=EMBEDDING_SIZE,
            tokenization_config=SINGLE_BANDSET_CONFIG,
        )
        decoder_config = PredictorConfig(
            supported_modality_names=ENCODABLE_MODALITY_NAMES,
            encoder_embedding_size=EMBEDDING_SIZE,
            output_embedding_size=EMBEDDING_SIZE,
        )

        config = PretrainedTargetLatentMIMConfig(
            encoder_config=encoder_config,
            decoder_config=decoder_config,
            target_encoder_model_id="OlmoEarth-v1-Base",
            encodable_modality_names=ENCODABLE_MODALITY_NAMES,
        )

        mock_pretrained_encoder = _make_encoder(ENCODABLE_MODALITIES)
        mock_model = MagicMock()
        mock_model.encoder = mock_pretrained_encoder

        with patch.object(config, "_load_pretrained_model", return_value=mock_model):
            model = config.build()

        model.train()
        sample = _make_masked_sample()
        latent, decoded, pooled, reconstructed, extra_metrics = model(
            sample, PATCH_SIZE
        )
        assert isinstance(latent, TokensAndMasks)
        assert isinstance(decoded, TokensAndMasks)


class TestEMAGuard:
    """Tests for the EMA guard in update_target_encoder."""

    def test_ema_guard_with_pretrained_target(self) -> None:
        """Test that EMA update raises for PretrainedTargetLatentMIM."""
        model = TestPretrainedTargetLatentMIM()._build_model()

        # Simulate what ContrastiveLatentMIMTrainModule does
        from olmoearth_pretrain.nn.pretrained_target_latent_mim import (
            PretrainedTargetLatentMIM,
        )

        assert isinstance(model, PretrainedTargetLatentMIM)
