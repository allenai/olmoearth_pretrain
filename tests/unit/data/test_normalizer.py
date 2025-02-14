"""Test the normalizer."""

import numpy as np

from helios.data.constants import Modality
from helios.data.normalize import Normalizer, Strategy


def test_normalize_predefined() -> None:
    """Test the normalize function with predefined strategy."""
    # make data in dtype uint16
    data = np.random.randint(0, 10000, (256, 256, 12, 13), dtype=np.uint16)
    modality = Modality.SENTINEL2
    normalizer = Normalizer(Strategy.PREDEFINED)
    normalized_data = normalizer.normalize(modality, data)
    min_vals = np.array([0] * 13)
    max_vals = np.array([10000] * 13)
    expected_data = (data - min_vals) / (max_vals - min_vals)
    assert normalized_data.shape == data.shape
    assert normalized_data.dtype == np.float64
    # assert values are between 0 and 1
    assert np.all(normalized_data >= 0)
    assert np.all(normalized_data <= 1)
    assert np.allclose(normalized_data, expected_data)


def test_normalize_computed() -> None:
    """Test the normalize function with computed strategy."""
    data = np.random.randint(0, 10000, (256, 256, 12, 13), dtype=np.uint16)
    modality = Modality.SENTINEL2
    normalizer = Normalizer(Strategy.COMPUTED)
    normalized_data = normalizer.normalize(modality, data)
    assert normalized_data.shape == data.shape
    assert normalized_data.dtype == np.float64
