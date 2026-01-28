"""Integration tests for encoder with custom tokenization."""

import pytest
import torch

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.data.dataset import OlmoEarthSample
from olmoearth_pretrain.nn.flexi_vit import EncoderConfig
from olmoearth_pretrain.nn.tokenization import (
    ModalityTokenization,
    TokenizationConfig,
)
from olmoearth_pretrain.train.masking import MaskingConfig
from olmoearth_pretrain.train.train_module.contrastive_latentmim import (
    _propagate_tokenization_config,
)


class TestEncoderWithCustomTokenization:
    """Integration tests for encoder with custom tokenization."""

    def test_encoder_builds_with_custom_tokenization(self) -> None:
        """Encoder should build with custom tokenization config."""
        config = EncoderConfig(
            supported_modality_names=[Modality.SENTINEL2_L2A.name],
            embedding_size=64,
            depth=2,
            tokenization_config=TokenizationConfig(
                overrides={
                    Modality.SENTINEL2_L2A.name: ModalityTokenization(
                        band_groups=[
                            ["B02", "B03", "B04", "B08"],
                            ["B05", "B06", "B07", "B8A", "B11", "B12"],
                        ]
                    )
                }
            ),
        )

        encoder = config.build()

        # Should have 2 embedding modules for sentinel2_l2a (one per bandset)
        assert (
            len(encoder.patch_embeddings.per_modality_embeddings["sentinel2_l2a"]) == 2
        )

    def test_encoder_output_shape_matches_tokenization(self) -> None:
        """Output should have correct number of bandset tokens."""
        # Create encoder with 2 bandsets (each band separate for sentinel1)
        config = EncoderConfig(
            supported_modality_names=[Modality.SENTINEL1.name],
            embedding_size=64,
            depth=1,
            tokenization_config=TokenizationConfig(
                overrides={
                    Modality.SENTINEL1.name: ModalityTokenization(
                        band_groups=[
                            ["vv"],
                            ["vh"],
                        ]
                    )
                }
            ),
        )

        encoder = config.build()
        # Verify the patch embeddings have 2 modules
        assert len(encoder.patch_embeddings.per_modality_embeddings["sentinel1"]) == 2

    def test_existing_model_loads_without_tokenization_config(self) -> None:
        """Configs without tokenization_config should work (backwards compat)."""
        # Simulate loading old config that doesn't have tokenization_config
        old_config_dict = {
            "supported_modality_names": [Modality.SENTINEL2_L2A.name],
            "embedding_size": 64,
            "depth": 2,
            # Note: no tokenization_config field
        }

        config = EncoderConfig.from_dict(old_config_dict)
        encoder = config.build()

        # Should have 3 bandsets for sentinel2_l2a (the default)
        assert (
            len(
                encoder.patch_embeddings.per_modality_embeddings[
                    Modality.SENTINEL2_L2A.name
                ]
            )
            == 3
        )

    def test_encoder_with_per_band_tokenization(self) -> None:
        """Test encoder where each band is its own token."""
        # Get all sentinel2_l2a bands
        s2_bands = Modality.SENTINEL2_L2A.band_order

        config = EncoderConfig(
            supported_modality_names=[Modality.SENTINEL2_L2A.name],
            embedding_size=64,
            depth=1,
            tokenization_config=TokenizationConfig(
                overrides={
                    Modality.SENTINEL2_L2A.name: ModalityTokenization(
                        band_groups=[[band] for band in s2_bands]
                    )
                }
            ),
        )

        encoder = config.build()

        # Should have as many embedding modules as bands
        num_modules = len(
            encoder.patch_embeddings.per_modality_embeddings[
                Modality.SENTINEL2_L2A.name
            ]
        )
        assert num_modules == len(s2_bands)

    def test_mixed_modalities_with_partial_override(self) -> None:
        """Test encoder with some modalities overridden and some using defaults."""
        config = EncoderConfig(
            supported_modality_names=[
                Modality.SENTINEL2_L2A.name,
                Modality.SENTINEL1.name,
            ],
            embedding_size=64,
            depth=1,
            tokenization_config=TokenizationConfig(
                overrides={
                    # Override sentinel1 to have each band separate
                    Modality.SENTINEL1.name: ModalityTokenization(
                        band_groups=[
                            ["vv"],
                            ["vh"],
                        ]
                    )
                    # sentinel2_l2a uses default (3 bandsets)
                }
            ),
        )

        encoder = config.build()

        # sentinel2_l2a should have default 3 bandsets
        assert (
            len(
                encoder.patch_embeddings.per_modality_embeddings[
                    Modality.SENTINEL2_L2A.name
                ]
            )
            == 3
        )
        # sentinel1 should have 2 (overridden)
        assert (
            len(
                encoder.patch_embeddings.per_modality_embeddings[
                    Modality.SENTINEL1.name
                ]
            )
            == 2
        )

    def test_config_validation_fails_on_invalid_band(self) -> None:
        """Config validation should fail for invalid band names."""
        config = EncoderConfig(
            supported_modality_names=[Modality.SENTINEL2_L2A.name],
            embedding_size=64,
            depth=1,
            tokenization_config=TokenizationConfig(
                overrides={
                    Modality.SENTINEL2_L2A.name: ModalityTokenization(
                        band_groups=[
                            ["INVALID_BAND"],
                        ]
                    )
                }
            ),
        )

        with pytest.raises(ValueError, match="Band 'INVALID_BAND' not found"):
            config.build()

    def test_tokenization_config_preserved_in_encoder(self) -> None:
        """TokenizationConfig should be accessible from built encoder."""
        tokenization_config = TokenizationConfig(
            overrides={
                Modality.SENTINEL1.name: ModalityTokenization(
                    band_groups=[
                        ["vv"],
                        ["vh"],
                    ]
                )
            }
        )

        config = EncoderConfig(
            supported_modality_names=[Modality.SENTINEL1.name],
            embedding_size=64,
            depth=1,
            tokenization_config=tokenization_config,
        )

        encoder = config.build()

        # Tokenization config should be accessible
        assert encoder.tokenization_config is not None
        assert (
            encoder.tokenization_config.get_num_bandsets(Modality.SENTINEL1.name) == 2
        )


def test_masking_and_encoder_use_same_bandset_count() -> None:
    """Test that masking and encoder use consistent bandset counts from tokenization config."""
    s2_bands = Modality.SENTINEL2_L2A.band_order
    tokenization_config = TokenizationConfig(
        overrides={
            Modality.SENTINEL2_L2A.name: ModalityTokenization(
                band_groups=[
                    list(s2_bands)  # All bands in single token
                ]
            )
        }
    )

    encoder = EncoderConfig(
        supported_modality_names=[Modality.SENTINEL2_L2A.name],
        embedding_size=16,
        depth=1,
        num_heads=2,
        max_patch_size=1,
        min_patch_size=1,
        max_sequence_length=2,
        tokenization_config=tokenization_config,
    ).build()

    masking_strategy = MaskingConfig(
        strategy_config={"type": "random", "encode_ratio": 0.5, "decode_ratio": 0.5}
    ).build()
    _propagate_tokenization_config(masking_strategy, tokenization_config)

    sample = OlmoEarthSample(
        sentinel2_l2a=torch.zeros((1, 2, 2, 1, 12), dtype=torch.float32),
        timestamps=torch.zeros((1, 1, 3), dtype=torch.long),
        latlon=torch.zeros((1, 2), dtype=torch.float32),
    )

    masked = masking_strategy.apply_mask(sample, patch_size=1)

    expected_bandsets = tokenization_config.get_num_bandsets(
        Modality.SENTINEL2_L2A.name
    )
    assert masked.sentinel2_l2a_mask is not None
    assert masked.sentinel2_l2a_mask.shape[-1] == expected_bandsets

    output = encoder(masked, patch_size=1)
    tokens_and_masks = output["tokens_and_masks"]
    tokens = getattr(tokens_and_masks, Modality.SENTINEL2_L2A.name)
    assert tokens.shape[-2] == expected_bandsets
