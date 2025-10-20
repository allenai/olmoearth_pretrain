"""Test the normalizer."""

import numpy as np

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.data.normalize import Normalizer, Strategy


def test_normalize_predefined() -> None:
    """Test the normalize function with predefined strategy."""
    # make data in dtype uint16
    data = np.random.randint(0, 10000, (256, 256, 12, 12), dtype=np.uint16)
    modality = Modality.SENTINEL2_L2A
    normalizer = Normalizer(Strategy.PREDEFINED)
    normalized_data = normalizer.normalize(modality, data)
    min_vals = np.array([0] * 12)
    max_vals = np.array([10000] * 12)
    expected_data = (data - min_vals) / (max_vals - min_vals)
    assert normalized_data.shape == data.shape
    assert normalized_data.dtype == np.float64
    # assert values are between 0 and 1
    assert np.all(normalized_data >= 0)
    assert np.all(normalized_data <= 1)
    assert np.allclose(normalized_data, expected_data)


def test_normalize_computed() -> None:
    """Test the normalize function with computed strategy."""
    data = np.random.randint(0, 10000, (256, 256, 12, 12), dtype=np.uint16)
    modality = Modality.SENTINEL2_L2A
    normalizer = Normalizer(Strategy.COMPUTED)
    normalized_data = normalizer.normalize(modality, data)
    assert normalized_data.shape == data.shape
    assert normalized_data.dtype == np.float64
