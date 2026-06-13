"""Tests for batch-level metadata dropout view modes."""

import pytest
import torch

from olmoearth_pretrain.data.collate import MetadataDropout


def test_opposite_year_views_are_anticorrelated() -> None:
    """Opposite mode: exactly one view drops the year, every draw."""
    md = MetadataDropout(
        year_dropout_rate=0.5, latlon_dropout_rate=0.0, view_mode="opposite"
    )
    torch.manual_seed(0)
    for _ in range(50):
        (year_a, latlon_a), (year_b, latlon_b) = md.draw_opposite()
        assert year_a != year_b  # always opposite
        assert latlon_a is False and latlon_b is False  # disabled field never drops


def test_opposite_latlon_views_are_anticorrelated() -> None:
    """Opposite mode: exactly one view drops latlon, every draw."""
    md = MetadataDropout(
        year_dropout_rate=0.0, latlon_dropout_rate=0.5, view_mode="opposite"
    )
    torch.manual_seed(1)
    for _ in range(50):
        (year_a, latlon_a), (year_b, latlon_b) = md.draw_opposite()
        assert latlon_a != latlon_b
        assert year_a is False and year_b is False


def test_opposite_fields_independent() -> None:
    """Year and latlon split independently (which view drops each can differ)."""
    md = MetadataDropout(
        year_dropout_rate=0.5, latlon_dropout_rate=0.5, view_mode="opposite"
    )
    torch.manual_seed(2)
    seen_year_a_latlon_b = False
    seen_year_b_latlon_a = False
    for _ in range(200):
        (year_a, latlon_a), (year_b, latlon_b) = md.draw_opposite()
        assert year_a != year_b
        assert latlon_a != latlon_b
        if year_a and latlon_b:
            seen_year_a_latlon_b = True
        if year_b and latlon_a:
            seen_year_b_latlon_a = True
    # Over 200 draws both cross-assignments should appear (independence).
    assert seen_year_a_latlon_b and seen_year_b_latlon_a


def test_opposite_which_view_is_balanced() -> None:
    """Across draws, each view drops the field ~50% of the time."""
    md = MetadataDropout(
        year_dropout_rate=1.0, latlon_dropout_rate=0.0, view_mode="opposite"
    )
    torch.manual_seed(3)
    a_drops = sum(md.draw_opposite()[0][0] for _ in range(400))
    assert 150 < a_drops < 250  # ~200/400


def test_opposite_disabled_when_rate_zero() -> None:
    """Both rates 0: no drops in either view."""
    md = MetadataDropout(
        year_dropout_rate=0.0, latlon_dropout_rate=0.0, view_mode="opposite"
    )
    (year_a, latlon_a), (year_b, latlon_b) = md.draw_opposite()
    assert not any([year_a, latlon_a, year_b, latlon_b])


def test_invalid_view_mode_rejected() -> None:
    """Unknown view_mode raises."""
    with pytest.raises(ValueError, match="view_mode"):
        MetadataDropout(view_mode="bogus")
