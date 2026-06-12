"""Test modality constants."""

import olmoearth_pretrain.data.constants as legacy_constants
import olmoearth_pretrain.modalities as modalities
from olmoearth_pretrain.modalities import Modality


def test_data_constants_reexports_modalities() -> None:
    """The historical constants module should re-export the canonical objects."""
    object_names = [
        "BandSet",
        "Modality",
        "ModalitySpec",
        "TimeSpan",
        "get_modality_specs_from_names",
        "get_resolution",
    ]
    for name in object_names:
        assert getattr(legacy_constants, name) is getattr(modalities, name)
    for name in legacy_constants.__all__:
        assert getattr(legacy_constants, name) == getattr(modalities, name)


def test_modality_spec_band_order() -> None:
    """Test that the band order is correct.

    This should be the order the data is stacked in
    """
    expected_band_order_sentinel2 = [
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
        "B10",
    ]

    expected_band_order_landsat = [
        "B8",
        "B1",
        "B2",
        "B3",
        "B4",
        "B5",
        "B6",
        "B7",
        "B9",
        "B10",
        "B11",
    ]
    assert Modality.SENTINEL2.band_order == expected_band_order_sentinel2
    assert Modality.LANDSAT.band_order == expected_band_order_landsat


def test_modality_spec_num_bands() -> None:
    """Test that the number of channels is correct."""
    assert Modality.SENTINEL2.num_bands == 13
    assert Modality.LANDSAT.num_bands == 11


def test_band_sets_as_indices() -> None:
    """Test that the band sets as indices are correct."""
    assert Modality.SENTINEL2.bandsets_as_indices() == [
        [0, 1, 2, 3],
        [4, 5, 6, 7, 8, 9],
        [10, 11, 12],
    ]
