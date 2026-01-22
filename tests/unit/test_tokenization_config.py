"""Tests for configurable tokenization."""

import pytest

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.nn.tokenization import (
    ModalityTokenization,
    TokenizationBandSet,
    TokenizationConfig,
)


class TestTokenizationConfig:
    """Tests for configurable tokenization."""

    def test_default_config_matches_constants(self) -> None:
        """No overrides should return same indices as ModalitySpec."""
        config = TokenizationConfig()

        # Sentinel2 L2A has 3 bandsets by default
        default_indices = Modality.get("sentinel2_l2a").bandsets_as_indices()
        config_indices = config.get_bandset_indices("sentinel2_l2a")

        assert config_indices == default_indices
        assert config.get_num_bandsets("sentinel2_l2a") == 3

    def test_custom_single_band_tokenization(self) -> None:
        """Each band as its own token."""
        # Sentinel1 has bands ["vv", "vh"] at indices [0, 1]
        config = TokenizationConfig(
            overrides={
                "sentinel1": ModalityTokenization(
                    band_groups=[
                        TokenizationBandSet(bands=["vv"]),
                        TokenizationBandSet(bands=["vh"]),
                    ]
                )
            }
        )

        indices = config.get_bandset_indices("sentinel1")
        assert indices == [[0], [1]]
        assert config.get_num_bandsets("sentinel1") == 2

    def test_custom_grouped_tokenization(self) -> None:
        """Custom grouping of bands."""
        # Sentinel2 L2A band_order: [B02, B03, B04, B08, B05, B06, B07, B8A, B11, B12, B01, B09]
        config = TokenizationConfig(
            overrides={
                "sentinel2_l2a": ModalityTokenization(
                    band_groups=[
                        # RGB-like group
                        TokenizationBandSet(bands=["B02", "B03", "B04"]),
                        # NIR group
                        TokenizationBandSet(bands=["B08", "B8A"]),
                        # SWIR group
                        TokenizationBandSet(bands=["B11", "B12"]),
                    ]
                )
            }
        )

        indices = config.get_bandset_indices("sentinel2_l2a")
        # B02=0, B03=1, B04=2, B08=3, B8A=7, B11=8, B12=9
        assert indices == [[0, 1, 2], [3, 7], [8, 9]]
        assert config.get_num_bandsets("sentinel2_l2a") == 3

    def test_band_order_preserved_in_group(self) -> None:
        """Bands within a group maintain requested order, not data order."""
        config = TokenizationConfig(
            overrides={
                "sentinel2_l2a": ModalityTokenization(
                    band_groups=[
                        # Request B04 before B02 (reversed from data order)
                        TokenizationBandSet(bands=["B04", "B02"]),
                    ]
                )
            }
        )

        indices = config.get_bandset_indices("sentinel2_l2a")
        # B04=2, B02=0 (order from config, not data)
        assert indices == [[2, 0]]

    def test_invalid_band_name_raises(self) -> None:
        """Unknown band name should raise KeyError."""
        config = TokenizationConfig(
            overrides={
                "sentinel2_l2a": ModalityTokenization(
                    band_groups=[
                        TokenizationBandSet(bands=["B02", "INVALID_BAND"]),
                    ]
                )
            }
        )

        with pytest.raises(KeyError):
            config.get_bandset_indices("sentinel2_l2a")

    def test_modality_without_override_uses_default(self) -> None:
        """Modalities not in overrides use default bandsets."""
        config = TokenizationConfig(
            overrides={
                "sentinel1": ModalityTokenization(
                    band_groups=[TokenizationBandSet(bands=["vv"])]
                )
            }
        )

        # sentinel2_l2a not overridden, should use default
        s2_indices = config.get_bandset_indices("sentinel2_l2a")
        assert s2_indices == Modality.get("sentinel2_l2a").bandsets_as_indices()

        # sentinel1 is overridden
        s1_indices = config.get_bandset_indices("sentinel1")
        assert s1_indices == [[0]]

    def test_validation_catches_invalid_band_name(self) -> None:
        """Validation should catch invalid band names."""
        config = TokenizationConfig(
            overrides={
                "sentinel2_l2a": ModalityTokenization(
                    band_groups=[
                        TokenizationBandSet(bands=["B02", "INVALID_BAND"]),
                    ]
                )
            }
        )

        with pytest.raises(ValueError, match="Band 'INVALID_BAND' not found"):
            config.validate()

    def test_get_num_bands_per_bandset(self) -> None:
        """Test getting number of bands per bandset."""
        config = TokenizationConfig(
            overrides={
                "sentinel2_l2a": ModalityTokenization(
                    band_groups=[
                        TokenizationBandSet(bands=["B02", "B03", "B04"]),
                        TokenizationBandSet(bands=["B08"]),
                    ]
                )
            }
        )

        bands_per_bandset = config.get_num_bands_per_bandset("sentinel2_l2a")
        assert bands_per_bandset == [3, 1]

        # Default for sentinel1
        s1_bands = config.get_num_bands_per_bandset("sentinel1")
        assert s1_bands == [2]  # sentinel1 has 2 bands in 1 bandset

    def test_modality_tokenization_num_band_sets(self) -> None:
        """ModalityTokenization.num_band_sets property."""
        tokenization = ModalityTokenization(
            band_groups=[
                TokenizationBandSet(bands=["B02", "B03"]),
                TokenizationBandSet(bands=["B04"]),
                TokenizationBandSet(bands=["B08"]),
            ]
        )
        assert tokenization.num_band_sets == 3

    def test_full_sentinel2_per_band_tokenization(self) -> None:
        """Test making each Sentinel-2 band its own token."""
        # This is a common experiment - each band as separate token
        s2_bands = Modality.get("sentinel2_l2a").band_order
        config = TokenizationConfig(
            overrides={
                "sentinel2_l2a": ModalityTokenization(
                    band_groups=[TokenizationBandSet(bands=[band]) for band in s2_bands]
                )
            }
        )

        indices = config.get_bandset_indices("sentinel2_l2a")
        # Each band should have its own index
        assert len(indices) == len(s2_bands)
        for i, idx_list in enumerate(indices):
            assert idx_list == [i]

        assert config.get_num_bandsets("sentinel2_l2a") == len(s2_bands)
