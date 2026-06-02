"""Tests for the minimal 3-number temporal and lat/lon encodings."""

import torch

from olmoearth_pretrain.nn.encodings import (
    get_simple_latlon_encoding,
    get_simple_temporal_encoding,
)


def _ts(year: int, month: int, day: int) -> torch.Tensor:
    """Build a (1, 1, 3) timestamp tensor of (day, month, year)."""
    return torch.tensor([[[day, month, year]]], dtype=torch.long)


# ---- simple temporal -----------------------------------------------------


def test_simple_temporal_shape() -> None:
    """Output is 3 channels: [frac_year, sin, cos]."""
    e = get_simple_temporal_encoding(_ts(2023, 4, 15))
    assert e.shape == (1, 1, 3)


def test_simple_temporal_frac_year_value() -> None:
    """Channel 0 is years-from-2020 (continuous)."""
    e = get_simple_temporal_encoding(_ts(2020, 0, 1))  # ~start of 2020
    assert abs(e[0, 0, 0].item()) < 0.02  # frac_year ~ 0


def test_simple_temporal_year_offset_is_linear() -> None:
    """Same day-of-year one year apart differs by ~1.0 on channel 0."""
    a = get_simple_temporal_encoding(_ts(2023, 4, 15))
    b = get_simple_temporal_encoding(_ts(2024, 4, 15))
    assert abs((b[0, 0, 0] - a[0, 0, 0]).item() - 1.0) < 1e-3


def test_simple_temporal_annual_phase_matches_across_years() -> None:
    """sin/cos (channels 1,2) match for the same day-of-year across years."""
    a = get_simple_temporal_encoding(_ts(2021, 6, 10))
    b = get_simple_temporal_encoding(_ts(2024, 6, 10))
    assert torch.allclose(a[..., 1:], b[..., 1:], atol=1e-3)


def test_simple_temporal_sincos_unit_norm() -> None:
    """sin^2 + cos^2 == 1 on channels 1,2."""
    e = get_simple_temporal_encoding(_ts(2023, 7, 20))
    sin, cos = e[0, 0, 1], e[0, 0, 2]
    assert abs((sin**2 + cos**2).item() - 1.0) < 1e-5


def test_simple_temporal_different_days_differ() -> None:
    """Different days-of-year give different annual phase."""
    a = get_simple_temporal_encoding(_ts(2023, 0, 1))
    b = get_simple_temporal_encoding(_ts(2023, 6, 1))
    assert (a[..., 1:] - b[..., 1:]).abs().max() > 0.1


def test_simple_temporal_batched() -> None:
    """Handles arbitrary leading dims."""
    ts = torch.randint(1, 28, (3, 5, 3), dtype=torch.long)
    ts[..., 1] = torch.randint(0, 12, (3, 5), dtype=torch.long)
    ts[..., 2] = torch.randint(2018, 2024, (3, 5), dtype=torch.long)
    e = get_simple_temporal_encoding(ts)
    assert e.shape == (3, 5, 3)


# ---- simple lat/lon ------------------------------------------------------


def test_simple_latlon_shape() -> None:
    """Output is 3 channels: unit-sphere (x, y, z)."""
    e = get_simple_latlon_encoding(torch.tensor([[37.0, -122.0]]))
    assert e.shape == (1, 3)


def test_simple_latlon_unit_norm() -> None:
    """(x, y, z) lies on the unit sphere."""
    pts = torch.tensor([[37.0, -122.0], [-30.0, 150.0], [0.0, 0.0], [90.0, 45.0]])
    e = get_simple_latlon_encoding(pts)
    assert torch.allclose(e.norm(dim=-1), torch.ones(4), atol=1e-5)


def test_simple_latlon_lon_wraparound() -> None:
    """lon=180 and lon=-180 map to the same point."""
    pts = torch.tensor([[45.0, 180.0], [45.0, -180.0]])
    e = get_simple_latlon_encoding(pts)
    assert (e[0] - e[1]).abs().max() < 1e-6


def test_simple_latlon_pole_invariant() -> None:
    """At the pole, longitude is irrelevant."""
    pts = torch.tensor([[90.0, 0.0], [90.0, 137.0]])
    e = get_simple_latlon_encoding(pts)
    assert (e[0] - e[1]).abs().max() < 1e-6


def test_simple_latlon_known_points() -> None:
    """Equator/prime-meridian -> (1,0,0); north pole -> (0,0,1)."""
    e = get_simple_latlon_encoding(torch.tensor([[0.0, 0.0], [90.0, 0.0]]))
    assert torch.allclose(e[0], torch.tensor([1.0, 0.0, 0.0]), atol=1e-5)
    assert torch.allclose(e[1], torch.tensor([0.0, 0.0, 1.0]), atol=1e-5)


def test_simple_latlon_batched() -> None:
    """Handles arbitrary leading dims."""
    latlon = torch.rand(3, 4, 2)
    latlon[..., 0] = latlon[..., 0] * 180 - 90
    latlon[..., 1] = latlon[..., 1] * 360 - 180
    e = get_simple_latlon_encoding(latlon)
    assert e.shape == (3, 4, 3)
