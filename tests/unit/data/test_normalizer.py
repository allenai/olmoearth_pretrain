"""Test the normalizer."""

import numpy as np
import pytest

from olmoearth_pretrain.data.dataset import OlmoEarthDataset
from olmoearth_pretrain.data.normalize import (
    NormalizationConfigError,
    Normalizer,
    Strategy,
)
from olmoearth_pretrain.modalities import Modality


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


def test_normalize_missing_config_raises_specific_error() -> None:
    """Missing normalization stats should raise a dedicated config error."""
    data = np.random.randint(0, 10000, (16, 16, 12), dtype=np.uint16)
    normalizer = Normalizer(Strategy.COMPUTED)
    normalizer.norm_config = {}

    with pytest.raises(NormalizationConfigError):
        normalizer.normalize(Modality.SENTINEL2_L2A, data)


class _FakeNormalizer:
    def __init__(self, result: np.ndarray | None = None, exc: Exception | None = None):
        self.result = result
        self.exc = exc

    def normalize(self, modality: Modality, image: np.ndarray) -> np.ndarray:
        if self.exc is not None:
            raise self.exc
        assert self.result is not None
        return self.result


def test_dataset_normalize_image_falls_back_only_for_config_errors() -> None:
    """Dataset normalization fallback should not hide non-config failures."""
    image = np.zeros((1, 1, 12), dtype=np.float32)
    fallback_result = np.ones_like(image)
    dataset = object.__new__(OlmoEarthDataset)

    dataset.normalizer_computed = _FakeNormalizer(
        exc=NormalizationConfigError("missing stats")
    )
    dataset.normalizer_predefined = _FakeNormalizer(result=fallback_result)

    assert np.array_equal(
        dataset.normalize_image(Modality.SENTINEL2_L2A, image), fallback_result
    )

    dataset.normalizer_computed = _FakeNormalizer(exc=ValueError("shape error"))
    dataset.normalizer_predefined = _FakeNormalizer(result=fallback_result)

    with pytest.raises(ValueError, match="shape error"):
        dataset.normalize_image(Modality.SENTINEL2_L2A, image)
