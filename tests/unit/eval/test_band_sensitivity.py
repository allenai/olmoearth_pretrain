"""Tests for per-band occlusion sensitivity diagnostics."""

import pytest
import torch

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.evals.band_sensitivity import (
    embedding_drift,
    occlude_band,
    reliance_profile,
)
from olmoearth_pretrain.train.masking import MaskedOlmoEarthSample


def _sample(num_bands: int = 12) -> MaskedOlmoEarthSample:
    return MaskedOlmoEarthSample(
        timestamps=torch.zeros(2, 1, 3),
        sentinel2_l2a=torch.rand(2, 4, 4, 1, num_bands) + 1.0,
        sentinel2_l2a_mask=torch.zeros(2, 4, 4, 1, 1),
    )


def test_occlude_band_zeros_only_the_requested_band() -> None:
    """Other bands, and the mask, are left exactly as they were."""
    sample = _sample()
    occluded = occlude_band(sample, "sentinel2_l2a", 8)

    assert sample.sentinel2_l2a is not None
    assert occluded.sentinel2_l2a is not None
    assert (occluded.sentinel2_l2a[..., 8] == 0).all()
    kept = [i for i in range(12) if i != 8]
    torch.testing.assert_close(
        occluded.sentinel2_l2a[..., kept], sample.sentinel2_l2a[..., kept]
    )
    # The token is still encoded; only its information content changed.
    torch.testing.assert_close(occluded.sentinel2_l2a_mask, sample.sentinel2_l2a_mask)
    # The input the caller passed in is untouched.
    assert (sample.sentinel2_l2a[..., 8] != 0).any()


def test_occlude_band_rejects_absent_modality_and_bad_index() -> None:
    """Misconfiguration fails loudly rather than silently measuring nothing."""
    sample = _sample()
    with pytest.raises(ValueError, match="no such modality"):
        occlude_band(sample, "landsat", 0)
    with pytest.raises(IndexError, match="out of range"):
        occlude_band(sample, "sentinel2_l2a", 12)


def test_embedding_drift_is_zero_for_an_unread_band() -> None:
    """An occlusion the model ignores registers as no movement."""
    reference = torch.randn(16, 8)
    identical = embedding_drift(reference, reference.clone())
    assert identical["emb_cos"] == pytest.approx(1.0, abs=1e-5)
    assert identical["emb_rel_l2"] == pytest.approx(0.0, abs=1e-5)

    moved = embedding_drift(reference, reference + torch.randn(16, 8))
    assert moved["emb_cos"] < 0.99
    assert moved["emb_rel_l2"] > 0.0


def test_reliance_profile_counts_the_bands_actually_read() -> None:
    """Effective band count tracks how widely reliance is spread."""
    bands = [f"B{i}" for i in range(12)]

    spread = reliance_profile(dict.fromkeys(bands, 0.5))
    assert spread["effective_num_bands"] == pytest.approx(12.0, rel=1e-6)
    assert spread["max_band_share"] == pytest.approx(1 / 12, rel=1e-6)

    # Collapsed onto four bands: the other eight can be removed for free.
    collapsed = {b: (1.0 if i < 4 else 0.0) for i, b in enumerate(bands)}
    assert reliance_profile(collapsed)["effective_num_bands"] == pytest.approx(
        4.0, rel=1e-6
    )

    single = {b: (1.0 if i == 0 else 0.0) for i, b in enumerate(bands)}
    assert reliance_profile(single)["effective_num_bands"] == pytest.approx(1.0)
    assert reliance_profile(single)["max_band_share"] == pytest.approx(1.0)


def test_reliance_profile_handles_noise_and_degenerate_profiles() -> None:
    """Negative drops floor at zero; an all-zero profile is not called uniform."""
    noisy = reliance_profile({"B02": 1.0, "B03": -0.02, "B04": 0.0})
    assert noisy["effective_num_bands"] == pytest.approx(1.0)

    dead = reliance_profile(dict.fromkeys(["B02", "B03"], 0.0))
    assert dead["effective_num_bands"] == 0.0

    assert reliance_profile({}) == {}


def test_s2_band_order_matches_the_sweep_width() -> None:
    """The sweep enumerates exactly the bands the eval dataset supplies."""
    band_order = Modality.get(Modality.SENTINEL2_L2A.name).band_order
    assert len(band_order) == 12
    assert band_order[0] == "B02"
    assert "B11" in band_order and "B12" in band_order
