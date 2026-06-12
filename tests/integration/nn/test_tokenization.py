"""Integration tests for encoder with custom tokenization."""

import pytest
import torch

from olmoearth_pretrain.datatypes import OlmoEarthSample
from olmoearth_pretrain.modalities import Modality
from olmoearth_pretrain.nn.flexi_vit import EncoderConfig
from olmoearth_pretrain.nn.tokenization import (
    ModalityTokenization,
    TokenizationConfig,
)
from olmoearth_pretrain.train.masking import (
    MaskingConfig,
    propagate_tokenization_config,
)


class TestEncoderWithCustomTokenization:
    """Integration tests for encoder with custom tokenization."""

    @pytest.mark.parametrize(
        ("supported_modality_names", "overrides", "expected_bandsets"),
        [
            (
                [Modality.SENTINEL2_L2A.name],
                {
                    Modality.SENTINEL2_L2A.name: ModalityTokenization(
                        band_groups=[
                            ["B02", "B03", "B04", "B08"],
                            ["B05", "B06", "B07", "B8A", "B11", "B12"],
                        ]
                    )
                },
                {Modality.SENTINEL2_L2A.name: 2},
            ),
            (
                [Modality.SENTINEL2_L2A.name],
                {
                    Modality.SENTINEL2_L2A.name: ModalityTokenization(
                        band_groups=[
                            [band] for band in Modality.SENTINEL2_L2A.band_order
                        ]
                    )
                },
                {Modality.SENTINEL2_L2A.name: len(Modality.SENTINEL2_L2A.band_order)},
            ),
            (
                [Modality.SENTINEL2_L2A.name, Modality.SENTINEL1.name],
                {
                    Modality.SENTINEL1.name: ModalityTokenization(
                        band_groups=[["vv"], ["vh"]]
                    )
                },
                {Modality.SENTINEL2_L2A.name: 3, Modality.SENTINEL1.name: 2},
            ),
        ],
    )
    def test_encoder_uses_configured_bandsets(
        self,
        supported_modality_names: list[str],
        overrides: dict[str, ModalityTokenization],
        expected_bandsets: dict[str, int],
    ) -> None:
        """Encoder should build one patch embedding per configured bandset."""
        encoder = EncoderConfig(
            supported_modality_names=supported_modality_names,
            embedding_size=64,
            depth=1,
            tokenization_config=TokenizationConfig(overrides=overrides),
        ).build()

        assert encoder.tokenization_config is not None
        for modality_name, num_bandsets in expected_bandsets.items():
            assert encoder.tokenization_config.get_num_bandsets(modality_name) == (
                num_bandsets
            )
            assert (
                len(encoder.patch_embeddings.per_modality_embeddings[modality_name])
                == num_bandsets
            )

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
    propagate_tokenization_config(masking_strategy, tokenization_config)

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
