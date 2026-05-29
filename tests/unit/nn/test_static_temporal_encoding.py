"""Tests for get_static_temporal_encoding."""

import pytest
import torch

from olmoearth_pretrain.nn.encodings import get_static_temporal_encoding


def _ts(year: int, month: int, day: int) -> torch.Tensor:
    """Build a single (B=1, T=1, 3) timestamp tensor of (day, month, year)."""
    return torch.tensor([[[day, month, year]]], dtype=torch.long)


def test_shape_and_dtype() -> None:
    """Output has shape ``(..., encoding_dim)`` and matches input float dtype."""
    ts = _ts(2023, 5, 15)
    e = get_static_temporal_encoding(ts, encoding_dim=64)
    assert e.shape == (1, 1, 64)
    assert e.dtype == torch.float32


def test_odd_dim_raises() -> None:
    """encoding_dim must be even (sin/cos split)."""
    with pytest.raises(ValueError, match="must be even"):
        get_static_temporal_encoding(_ts(2023, 5, 15), encoding_dim=63)


def test_deterministic_same_timestamp() -> None:
    """Two calls with the same timestamp return identical encodings."""
    a = get_static_temporal_encoding(_ts(2023, 5, 15), 96)
    b = get_static_temporal_encoding(_ts(2023, 5, 15), 96)
    assert torch.allclose(a, b)


def test_different_timestamps_differ() -> None:
    """Different years on the same day produce different encodings."""
    a = get_static_temporal_encoding(_ts(2023, 5, 15), 96)
    b = get_static_temporal_encoding(_ts(2024, 5, 15), 96)
    assert not torch.allclose(a, b, atol=1e-4)


def test_same_day_of_year_across_years_partial_match() -> None:
    """1-cycle/year frequency channel should match for same day-of-year."""
    a = get_static_temporal_encoding(_ts(2023, 5, 15), 64)
    b = get_static_temporal_encoding(_ts(2024, 5, 15), 64)
    # they should agree on some channels (the year-periodic ones) but differ overall
    # so the diff is non-zero but not maximal.
    diff = (a - b).abs()
    assert diff.max() > 1e-4, "yearly difference should be visible somewhere"
    assert diff.mean() < 1.0, "should not be maximally different"


def test_unit_norm_per_channel_pair() -> None:
    """sin^2 + cos^2 = 1 for each freq pair."""
    e = get_static_temporal_encoding(_ts(2023, 5, 15), 64)
    n = 64 // 2
    sin_part = e[..., :n]
    cos_part = e[..., n:]
    sq = sin_part**2 + cos_part**2
    assert torch.allclose(sq, torch.ones_like(sq), atol=1e-5)


def test_batched_shape() -> None:
    """Handles arbitrary leading dims."""
    ts = torch.randint(1, 28, (3, 5, 3), dtype=torch.long)
    ts[..., 1] = torch.randint(0, 12, (3, 5), dtype=torch.long)
    ts[..., 2] = torch.randint(2018, 2024, (3, 5), dtype=torch.long)
    e = get_static_temporal_encoding(ts, 128)
    assert e.shape == (3, 5, 128)
