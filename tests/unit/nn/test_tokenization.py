"""Tests for tokenization configuration."""

import pytest

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.nn.tokenization import (
    ModalityTokenization,
    TokenizationConfig,
)


class TestTokenizationConfig:
    """Tests for configurable tokenization."""

    def test_default_config_matches_constants(self) -> None:
        """No overrides should return same indices as ModalitySpec."""
        config = TokenizationConfig()

        # Sentinel2 L2A has 3 bandsets by default
        default_indices = Modality.SENTINEL2_L2A.bandsets_as_indices()
        config_indices = config.get_bandset_indices(Modality.SENTINEL2_L2A.name)

        assert config_indices == default_indices
        assert config.get_num_bandsets(Modality.SENTINEL2_L2A.name) == 3

    def test_custom_single_band_tokenization(self) -> None:
        """Each band as its own token."""
        config = TokenizationConfig(
            overrides={
                Modality.SENTINEL1.name: ModalityTokenization(
                    band_groups=[
                        ["vv"],
                        ["vh"],
                    ]
                )
            }
        )

        indices = config.get_bandset_indices(Modality.SENTINEL1.name)
        assert indices == [[0], [1]]
        assert config.get_num_bandsets(Modality.SENTINEL1.name) == 2

    def test_custom_grouped_tokenization(self) -> None:
        """Custom grouping of bands."""
        config = TokenizationConfig(
            overrides={
                Modality.SENTINEL2_L2A.name: ModalityTokenization(
                    band_groups=[
                        # RGB-like group
                        ["B02", "B03", "B04"],
                        # NIR group
                        ["B08", "B8A"],
                        # SWIR group
                        ["B11", "B12"],
                    ]
                )
            }
        )

        indices = config.get_bandset_indices(Modality.SENTINEL2_L2A.name)
        # B02=0, B03=1, B04=2, B08=3, B8A=7, B11=8, B12=9
        assert indices == [[0, 1, 2], [3, 7], [8, 9]]
        assert config.get_num_bandsets(Modality.SENTINEL2_L2A.name) == 3

    def test_band_order_preserved_in_group(self) -> None:
        """Bands within a group maintain requested order, not data order."""
        config = TokenizationConfig(
            overrides={
                Modality.SENTINEL2_L2A.name: ModalityTokenization(
                    band_groups=[
                        # Request B04 before B02 (reversed from data order)
                        ["B04", "B02"],
                    ]
                )
            }
        )

        indices = config.get_bandset_indices(Modality.SENTINEL2_L2A.name)
        # B04=2, B02=0 (order from config, not data)
        assert indices == [[2, 0]]

    def test_invalid_band_name_raises(self) -> None:
        """Unknown band name should raise ValueError."""
        config = TokenizationConfig(
            overrides={
                Modality.SENTINEL2_L2A.name: ModalityTokenization(
                    band_groups=[
                        ["B02", "INVALID_BAND"],
                    ]
                )
            }
        )

        with pytest.raises(ValueError, match="Band 'INVALID_BAND' not found"):
            config.get_bandset_indices(Modality.SENTINEL2_L2A.name)

    def test_modality_without_override_uses_default(self) -> None:
        """Modalities not in overrides use default bandsets."""
        config = TokenizationConfig(
            overrides={
                Modality.SENTINEL1.name: ModalityTokenization(band_groups=[["vv"]])
            }
        )

        # sentinel2_l2a not overridden, should use default
        s2_indices = config.get_bandset_indices(Modality.SENTINEL2_L2A.name)
        assert s2_indices == Modality.SENTINEL2_L2A.bandsets_as_indices()

        # sentinel1 is overridden
        s1_indices = config.get_bandset_indices(Modality.SENTINEL1.name)
        assert s1_indices == [[0]]

    def test_validation_catches_invalid_band_name(self) -> None:
        """Validation should catch invalid band names."""
        config = TokenizationConfig(
            overrides={
                Modality.SENTINEL2_L2A.name: ModalityTokenization(
                    band_groups=[
                        ["B02", "INVALID_BAND"],
                    ]
                )
            }
        )

        with pytest.raises(ValueError, match="Band 'INVALID_BAND' not found"):
            config.validate()

    def test_validation_catches_invalid_modality_name(self) -> None:
        """Validation should catch invalid modality names."""
        config = TokenizationConfig(
            overrides={
                "INVALID_MODALITY": ModalityTokenization(
                    band_groups=[["B02"]],
                )
            }
        )

        with pytest.raises(ValueError, match="Invalid modality name in overrides"):
            config.validate()

    def test_get_num_bands_per_bandset(self) -> None:
        """Test getting number of bands per bandset."""
        config = TokenizationConfig(
            overrides={
                Modality.SENTINEL2_L2A.name: ModalityTokenization(
                    band_groups=[
                        ["B02", "B03", "B04"],
                        ["B08"],
                    ]
                )
            }
        )

        bands_per_bandset = config.get_num_bands_per_bandset(
            Modality.SENTINEL2_L2A.name
        )
        assert bands_per_bandset == [3, 1]

        # Default for sentinel1
        s1_bands = config.get_num_bands_per_bandset(Modality.SENTINEL1.name)
        assert s1_bands == [2]  # sentinel1 has 2 bands in 1 bandset

    def test_modality_tokenization_num_band_sets(self) -> None:
        """ModalityTokenization.num_band_sets property."""
        tokenization = ModalityTokenization(
            band_groups=[
                ["B02", "B03"],
                ["B04"],
                ["B08"],
            ]
        )
        assert tokenization.num_band_sets == 3

    def test_full_sentinel2_per_band_tokenization(self) -> None:
        """Test making each Sentinel-2 band its own token."""
        s2_bands = Modality.SENTINEL2_L2A.band_order
        config = TokenizationConfig(
            overrides={
                Modality.SENTINEL2_L2A.name: ModalityTokenization(
                    band_groups=[[band] for band in s2_bands]
                )
            }
        )

        indices = config.get_bandset_indices(Modality.SENTINEL2_L2A.name)
        # Each band should have its own index
        assert len(indices) == len(s2_bands)
        for i, idx_list in enumerate(indices):
            assert idx_list == [i]

        assert config.get_num_bandsets(Modality.SENTINEL2_L2A.name) == len(s2_bands)


class TestTokenizationWithMasking:
    """Tests for tokenization config propagation to masking strategies."""

    def test_tokenization_config_propagates_to_nested_masking_strategies(self) -> None:
        """Tokenization config should propagate to nested masking strategies."""
        from olmoearth_pretrain.train.masking import RandomFixedModalityMaskingStrategy
        from olmoearth_pretrain.train.train_module.contrastive_latentmim import (
            _propagate_tokenization_config,
        )

        tokenization_config = TokenizationConfig(
            overrides={
                Modality.SENTINEL2_L2A.name: ModalityTokenization(
                    band_groups=[["B02", "B03"]]
                )
            }
        )

        strategy = RandomFixedModalityMaskingStrategy(
            decoded_modalities=[Modality.SENTINEL2_L2A.name]
        )

        _propagate_tokenization_config(strategy, tokenization_config)

        assert strategy.tokenization_config is tokenization_config
        # Wrapped strategy should also receive the config
        assert strategy.strategy.tokenization_config is tokenization_config
