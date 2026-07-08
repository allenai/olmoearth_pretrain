"""Test the normalizer."""

import numpy as np

from olmoearth_pretrain.data.constants import MISSING_VALUE, Modality
from olmoearth_pretrain.data.normalize import (
    Normalizer,
    Strategy,
    load_arcsinh_tanh_config,
)


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


def test_normalize_arcsinh_tanh_range_and_shape() -> None:
    """arcsinh_tanh normalization outputs values in (-1, 1) with the same shape."""
    data = np.random.randint(1, 20000, (32, 32, 12, 12), dtype=np.uint16)
    modality = Modality.SENTINEL2_L2A
    normalizer = Normalizer(Strategy.ARCSINH_TANH)
    normalized = normalizer.normalize(modality, data)
    assert normalized.shape == data.shape
    assert np.all(normalized > -1.0)
    assert np.all(normalized < 1.0)
    assert np.isfinite(normalized).all()


def test_normalize_arcsinh_tanh_monotonic() -> None:
    """arcsinh_tanh is monotonically increasing in the raw band value."""
    modality = Modality.SENTINEL2_L2A
    normalizer = Normalizer(Strategy.ARCSINH_TANH)
    # Ascending values for a single band, broadcast across the band dimension.
    values = np.linspace(1, 20000, 50)
    data = np.tile(values[:, None], (1, modality.num_bands))
    normalized = normalizer.normalize(modality, data)
    # Each band column should be strictly increasing.
    diffs = np.diff(normalized, axis=0)
    assert np.all(diffs > 0)


def test_normalize_arcsinh_tanh_identity_band() -> None:
    """Sentinel-1 (dB) bands use the identity transform: tanh(z-score)."""
    config = load_arcsinh_tanh_config()
    assert config["sentinel1"]["vv"]["transform"] == "identity"
    modality = Modality.SENTINEL1
    normalizer = Normalizer(Strategy.ARCSINH_TANH)
    # Feed the per-band mean; identity + z-score => 0 => tanh(0) == 0.
    means = np.array([config["sentinel1"][b]["mean"] for b in modality.band_order])
    data = np.broadcast_to(means, (4, 4, 3, modality.num_bands)).copy()
    normalized = normalizer.normalize(modality, data)
    assert np.allclose(normalized, 0.0, atol=1e-6)


def test_normalize_arcsinh_tanh_tanh_gain() -> None:
    """A larger tanh_gain saturates the tails more (larger magnitude output)."""
    data = np.full((8, 8, 12, 12), 8000.0)
    modality = Modality.SENTINEL2_L2A
    low_gain = Normalizer(Strategy.ARCSINH_TANH, tanh_gain=0.5).normalize(
        modality, data
    )
    high_gain = Normalizer(Strategy.ARCSINH_TANH, tanh_gain=2.0).normalize(
        modality, data
    )
    # Values above the per-band mean map to positive outputs; higher gain pushes
    # them closer to 1.
    assert np.all(high_gain >= low_gain - 1e-9)
    assert np.all(np.abs(high_gain) <= 1.0)


def test_normalize_arcsinh_tanh_missing_value_safe() -> None:
    """MISSING_VALUE pixels must not produce NaN/inf under arcsinh_tanh."""
    modality = Modality.SENTINEL2_L2A
    data = np.full((4, 4, 12, modality.num_bands), float(MISSING_VALUE))
    normalized = Normalizer(Strategy.ARCSINH_TANH).normalize(modality, data)
    assert np.isfinite(normalized).all()
    # Large-negative input saturates toward -1.
    assert np.all(normalized < 0)
