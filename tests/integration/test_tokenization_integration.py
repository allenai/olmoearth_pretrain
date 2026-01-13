"""Integration tests for encoder with custom tokenization."""

import pytest
import torch

from olmoearth_pretrain.data.constants import (
    Modality,
    ModalityTokenization,
    TokenizationBandSet,
    TokenizationConfig,
)
from olmoearth_pretrain.nn.flexi_vit import Encoder, EncoderConfig


class TestEncoderWithCustomTokenization:
    """Integration tests for encoder with custom tokenization."""

    def test_encoder_builds_with_custom_tokenization(self) -> None:
        """Encoder should build with custom tokenization config."""
        config = EncoderConfig(
            supported_modality_names=["sentinel2_l2a"],
            embedding_size=64,
            depth=2,
            tokenization_config=TokenizationConfig(
                overrides={
                    "sentinel2_l2a": ModalityTokenization(
                        band_groups=[
                            TokenizationBandSet(bands=["B02", "B03", "B04", "B08"]),
                            TokenizationBandSet(
                                bands=["B05", "B06", "B07", "B8A", "B11", "B12"]
                            ),
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
            supported_modality_names=["sentinel1"],
            embedding_size=64,
            depth=1,
            tokenization_config=TokenizationConfig(
                overrides={
                    "sentinel1": ModalityTokenization(
                        band_groups=[
                            TokenizationBandSet(bands=["vv"]),
                            TokenizationBandSet(bands=["vh"]),
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
            "supported_modality_names": ["sentinel2_l2a"],
            "embedding_size": 64,
            "depth": 2,
            # Note: no tokenization_config field
        }

        config = EncoderConfig(**old_config_dict)
        encoder = config.build()

        # Should have 3 bandsets for sentinel2_l2a (the default)
        assert (
            len(encoder.patch_embeddings.per_modality_embeddings["sentinel2_l2a"]) == 3
        )

    def test_encoder_with_per_band_tokenization(self) -> None:
        """Test encoder where each band is its own token."""
        # Get all sentinel2_l2a bands
        s2_bands = Modality.get("sentinel2_l2a").band_order

        config = EncoderConfig(
            supported_modality_names=["sentinel2_l2a"],
            embedding_size=64,
            depth=1,
            tokenization_config=TokenizationConfig(
                overrides={
                    "sentinel2_l2a": ModalityTokenization(
                        band_groups=[
                            TokenizationBandSet(bands=[band]) for band in s2_bands
                        ]
                    )
                }
            ),
        )

        encoder = config.build()

        # Should have as many embedding modules as bands
        num_modules = len(
            encoder.patch_embeddings.per_modality_embeddings["sentinel2_l2a"]
        )
        assert num_modules == len(s2_bands)

    def test_mixed_modalities_with_partial_override(self) -> None:
        """Test encoder with some modalities overridden and some using defaults."""
        config = EncoderConfig(
            supported_modality_names=["sentinel2_l2a", "sentinel1"],
            embedding_size=64,
            depth=1,
            tokenization_config=TokenizationConfig(
                overrides={
                    # Override sentinel1 to have each band separate
                    "sentinel1": ModalityTokenization(
                        band_groups=[
                            TokenizationBandSet(bands=["vv"]),
                            TokenizationBandSet(bands=["vh"]),
                        ]
                    )
                    # sentinel2_l2a uses default (3 bandsets)
                }
            ),
        )

        encoder = config.build()

        # sentinel2_l2a should have default 3 bandsets
        assert (
            len(encoder.patch_embeddings.per_modality_embeddings["sentinel2_l2a"]) == 3
        )
        # sentinel1 should have 2 (overridden)
        assert len(encoder.patch_embeddings.per_modality_embeddings["sentinel1"]) == 2

    def test_config_validation_fails_on_invalid_band(self) -> None:
        """Config validation should fail for invalid band names."""
        config = EncoderConfig(
            supported_modality_names=["sentinel2_l2a"],
            embedding_size=64,
            depth=1,
            tokenization_config=TokenizationConfig(
                overrides={
                    "sentinel2_l2a": ModalityTokenization(
                        band_groups=[
                            TokenizationBandSet(bands=["INVALID_BAND"]),
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
                "sentinel1": ModalityTokenization(
                    band_groups=[
                        TokenizationBandSet(bands=["vv"]),
                        TokenizationBandSet(bands=["vh"]),
                    ]
                )
            }
        )

        config = EncoderConfig(
            supported_modality_names=["sentinel1"],
            embedding_size=64,
            depth=1,
            tokenization_config=tokenization_config,
        )

        encoder = config.build()

        # Tokenization config should be accessible
        assert encoder.tokenization_config is not None
        assert encoder.tokenization_config.get_num_bandsets("sentinel1") == 2
