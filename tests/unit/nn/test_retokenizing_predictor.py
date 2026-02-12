"""Unit tests for the retokenizing predictor module."""

import pytest
import torch

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.datatypes import MaskValue
from olmoearth_pretrain.nn.flexi_vit import (
    Predictor,
    PredictorConfig,
    TokensAndMasks,
)
from olmoearth_pretrain.nn.retokenizing_predictor import (
    RetokenizingPredictor,
    RetokenizingPredictorConfig,
)
from olmoearth_pretrain.nn.tokenization import (
    ModalityTokenization,
    TokenizationConfig,
)


def _collapsed_tokenization() -> TokenizationConfig:
    """Collapsed S2 tokenization: all 12 bands in 1 token."""
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
    """Default tokenization (no overrides)."""
    return TokenizationConfig()


class TestRetokenizingPredictor:
    """Tests for the RetokenizingPredictor class."""

    @pytest.fixture
    def supported_modality_names(self) -> list[str]:
        """Supported modality names for testing."""
        return [Modality.SENTINEL2_L2A.name, Modality.LATLON.name]

    @pytest.fixture
    def embedding_size(self) -> int:
        """Embedding size for testing."""
        return 16

    @pytest.fixture
    def retokenizing_predictor(
        self, supported_modality_names: list[str], embedding_size: int
    ) -> RetokenizingPredictor:
        """Build a RetokenizingPredictor: collapsed input -> default output."""
        supported_modalities = [Modality.get(n) for n in supported_modality_names]
        inner_predictor = Predictor(
            supported_modalities=supported_modalities,
            encoder_embedding_size=embedding_size,
            decoder_embedding_size=embedding_size,
            depth=1,
            mlp_ratio=2.0,
            num_heads=2,
            max_sequence_length=12,
            drop_path=0.0,
            output_embedding_size=embedding_size,
            tokenization_config=_default_tokenization(),
        )
        return RetokenizingPredictor(
            predictor=inner_predictor,
            input_tokenization_config=_collapsed_tokenization(),
            encoder_embedding_size=embedding_size,
            supported_modality_names=supported_modality_names,
        )

    def test_retokenize_shape(
        self, retokenizing_predictor: RetokenizingPredictor, embedding_size: int
    ) -> None:
        """Test that _retokenize converts bandset dims correctly."""
        B, P_H, P_W, T = 2, 2, 2, 1
        D = embedding_size
        # Collapsed S2: 1 bandset
        s2_tokens = torch.randn(B, P_H, P_W, T, 1, D)
        s2_mask = torch.full((B, P_H, P_W, T, 1), MaskValue.DECODER.value)
        # Latlon: 1 bandset (unchanged)
        ll_tokens = torch.randn(B, 1, D)
        ll_mask = torch.full((B, 1), MaskValue.DECODER.value)

        x = TokensAndMasks(
            sentinel2_l2a=s2_tokens,
            sentinel2_l2a_mask=s2_mask,
            latlon=ll_tokens,
            latlon_mask=ll_mask,
        )
        retokenized = retokenizing_predictor._retokenize(x)

        # S2 should now have 3 bandsets (default)
        assert retokenized.sentinel2_l2a is not None
        assert retokenized.sentinel2_l2a_mask is not None
        assert retokenized.sentinel2_l2a.shape == (B, P_H, P_W, T, 3, D)
        assert retokenized.sentinel2_l2a_mask.shape == (B, P_H, P_W, T, 3)
        # Latlon unchanged
        assert retokenized.latlon is not None
        assert retokenized.latlon_mask is not None
        assert retokenized.latlon.shape == (B, 1, D)
        assert retokenized.latlon_mask.shape == (B, 1)

    def test_retokenize_mask_broadcast(
        self, retokenizing_predictor: RetokenizingPredictor, embedding_size: int
    ) -> None:
        """Test mask values are correctly broadcast during retokenization."""
        B, P_H, P_W, T = 1, 1, 1, 1
        D = embedding_size

        # Mix of DECODER and MISSING values
        s2_tokens = torch.randn(B, P_H, P_W, T, 1, D)
        s2_mask = torch.tensor([[[[[MaskValue.DECODER.value]]]]]).float()

        x = TokensAndMasks(
            sentinel2_l2a=s2_tokens,
            sentinel2_l2a_mask=s2_mask,
        )
        retokenized = retokenizing_predictor._retokenize(x)

        # All 3 bandsets should have DECODER value
        expected = torch.full((B, P_H, P_W, T, 3), MaskValue.DECODER.value)
        assert torch.equal(retokenized.sentinel2_l2a_mask, expected)

    def test_retokenize_missing_preserved(
        self, retokenizing_predictor: RetokenizingPredictor, embedding_size: int
    ) -> None:
        """Test MISSING mask values are broadcast correctly."""
        B, P_H, P_W, T = 1, 1, 1, 1
        D = embedding_size

        s2_tokens = torch.randn(B, P_H, P_W, T, 1, D)
        s2_mask = torch.tensor([[[[[MaskValue.MISSING.value]]]]]).float()

        x = TokensAndMasks(
            sentinel2_l2a=s2_tokens,
            sentinel2_l2a_mask=s2_mask,
        )
        retokenized = retokenizing_predictor._retokenize(x)

        expected = torch.full((B, P_H, P_W, T, 3), MaskValue.MISSING.value)
        assert torch.equal(retokenized.sentinel2_l2a_mask, expected)

    def test_no_retokenize_when_same(self, embedding_size: int) -> None:
        """Test that no retokenization happens when configs match."""
        supported_modality_names = [Modality.SENTINEL2_L2A.name, Modality.LATLON.name]
        supported_modalities = [Modality.get(n) for n in supported_modality_names]
        same_config = _default_tokenization()

        inner_predictor = Predictor(
            supported_modalities=supported_modalities,
            encoder_embedding_size=embedding_size,
            decoder_embedding_size=embedding_size,
            depth=1,
            mlp_ratio=2.0,
            num_heads=2,
            max_sequence_length=12,
            drop_path=0.0,
            output_embedding_size=embedding_size,
            tokenization_config=same_config,
        )
        rp = RetokenizingPredictor(
            predictor=inner_predictor,
            input_tokenization_config=same_config,
            encoder_embedding_size=embedding_size,
            supported_modality_names=supported_modality_names,
        )
        # No retokenizers should be created
        assert len(rp.retokenizers) == 0
        assert len(rp._retokenize_info) == 0

    def test_forward_produces_target_tokenization(
        self, retokenizing_predictor: RetokenizingPredictor, embedding_size: int
    ) -> None:
        """Test full forward pass produces output with target tokenization."""
        B, P_H, P_W, T = 2, 2, 2, 1
        D = embedding_size

        # Input from encoder: collapsed S2 (1 bandset)
        s2_tokens = torch.randn(B, P_H, P_W, T, 1, D)
        s2_mask = torch.zeros(B, P_H, P_W, T, 1)
        # Some DECODER tokens
        s2_mask[:, 0, :, :, :] = MaskValue.DECODER.value

        ll_tokens = torch.randn(B, 1, D)
        ll_mask = torch.zeros(B, 1)

        x = TokensAndMasks(
            sentinel2_l2a=s2_tokens,
            sentinel2_l2a_mask=s2_mask,
            latlon=ll_tokens,
            latlon_mask=ll_mask,
        )
        timestamps = (
            torch.tensor([[15, 7, 2023]], dtype=torch.long)
            .unsqueeze(0)
            .expand(B, -1, -1)
        )

        output = retokenizing_predictor(x, timestamps, patch_size=8)

        # Output should have 3 bandsets for S2 (target/default tokenization)
        assert output.sentinel2_l2a.shape[-2] == 3
        assert output.sentinel2_l2a_mask.shape[-1] == 3


class TestRetokenizingPredictorConfig:
    """Tests for the RetokenizingPredictorConfig."""

    def test_build(self) -> None:
        """Test that config builds a RetokenizingPredictor."""
        supported_modality_names = [Modality.SENTINEL2_L2A.name, Modality.LATLON.name]
        inner_config = PredictorConfig(
            supported_modality_names=supported_modality_names,
            encoder_embedding_size=8,
            decoder_embedding_size=8,
            depth=1,
            mlp_ratio=2.0,
            num_heads=2,
            max_sequence_length=12,
            tokenization_config=_default_tokenization(),
        )
        config = RetokenizingPredictorConfig(
            predictor_config=inner_config,
            input_tokenization_config=_collapsed_tokenization(),
        )
        model = config.build()
        assert isinstance(model, RetokenizingPredictor)
        assert "sentinel2_l2a" in model.retokenizers

    def test_properties(self) -> None:
        """Test config properties delegate to inner config."""
        supported_modality_names = [Modality.SENTINEL2_L2A.name, Modality.LATLON.name]
        inner_config = PredictorConfig(
            supported_modality_names=supported_modality_names,
            encoder_embedding_size=16,
            decoder_embedding_size=8,
            max_sequence_length=6,
        )
        config = RetokenizingPredictorConfig(
            predictor_config=inner_config,
            input_tokenization_config=_collapsed_tokenization(),
        )
        assert config.encoder_embedding_size == 16
        assert config.max_sequence_length == 6
        assert len(config.supported_modalities) == 2
