"""Tests for get_static_latlon_encoding."""

import pytest
import torch

from olmoearth_pretrain.nn.encodings import get_static_latlon_encoding


def test_shape_and_dtype() -> None:
    """Output has shape ``(..., encoding_dim)`` and matches input float dtype."""
    latlon = torch.tensor([[37.0, -122.0], [-30.0, 150.0]])
    e = get_static_latlon_encoding(latlon, encoding_dim=96)
    assert e.shape == (2, 96)
    assert e.dtype == torch.float32


def test_dim_not_divisible_by_6_raises() -> None:
    """encoding_dim must be divisible by 6 (split across xyz × sin/cos)."""
    latlon = torch.tensor([[0.0, 0.0]])
    with pytest.raises(ValueError, match="divisible by 6"):
        get_static_latlon_encoding(latlon, encoding_dim=64)


def test_deterministic() -> None:
    """Two calls with the same latlon return identical encodings."""
    latlon = torch.tensor([[37.0, -122.0]])
    a = get_static_latlon_encoding(latlon, 96)
    b = get_static_latlon_encoding(latlon, 96)
    assert torch.allclose(a, b)


def test_lon_wrap_around_to_machine_precision() -> None:
    """lon=180 and lon=-180 are the same point; encoding should match."""
    pts = torch.tensor([[45.0, 180.0], [45.0, -180.0]])
    e = get_static_latlon_encoding(pts, 192)
    assert (e[0] - e[1]).abs().max() < 1e-5


def test_pole_invariant_in_longitude() -> None:
    """At the north pole (lat=90), longitude is meaningless; encoding should match."""
    pts = torch.tensor([[90.0, 0.0], [90.0, 90.0], [90.0, -120.0]])
    e = get_static_latlon_encoding(pts, 192)
    assert (e[0] - e[1]).abs().max() < 1e-5
    assert (e[0] - e[2]).abs().max() < 1e-5


def test_different_locations_differ() -> None:
    """Distinct lat/lon points produce distinct encodings."""
    pts = torch.tensor([[37.0, -122.0], [-30.0, 150.0], [0.0, 0.0]])
    e = get_static_latlon_encoding(pts, 192)
    assert (e[0] - e[1]).abs().max() > 0.1
    assert (e[0] - e[2]).abs().max() > 0.1
    assert (e[1] - e[2]).abs().max() > 0.1


def test_unit_norm_per_freq_axis_pair() -> None:
    """For each (axis, freq) pair, sin^2 + cos^2 == 1."""
    pts = torch.tensor([[37.0, -122.0]])
    e = get_static_latlon_encoding(pts, 96)
    half = 96 // 2
    sin_part = e[..., :half]
    cos_part = e[..., half:]
    sq = sin_part**2 + cos_part**2
    assert torch.allclose(sq, torch.ones_like(sq), atol=1e-5)


def test_batched_leading_dims() -> None:
    """Handles arbitrary leading dims."""
    latlon = torch.rand(3, 4, 2)
    latlon[..., 0] = latlon[..., 0] * 180 - 90
    latlon[..., 1] = latlon[..., 1] * 360 - 180
    e = get_static_latlon_encoding(latlon, 96)
    assert e.shape == (3, 4, 96)
