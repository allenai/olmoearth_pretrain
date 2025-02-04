"""Unit tests for HeliosSample."""

from helios.data.dataset import HeliosSample


def test_all_attrs_have_bands() -> None:
    """Test all attributes are described in attribute_to_bands."""
    attribute_to_bands = HeliosSample.attribute_to_bands()
    for attribute_name in HeliosSample._fields:
        assert attribute_name in attribute_to_bands
