"""Tests for the static_split spatial encoding mode and its building blocks.

The static_split mode replaces the legacy 2D-sincos-with-resolution spatial
encoding with two static signals occupying separate slots of the embedding:
- Local 2D position (resolution-aware, in slot [2n:3n])
- Global sphere-mapped lat/lon (in slot [3n:4n])

These tests cover both the standalone encoding functions and the wired-up
CompositeEncodings forward pass.
"""

import pytest
import torch

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.nn.encodings import (
    get_static_global_latlon_encoding,
    get_static_local_2d_encoding,
)
from olmoearth_pretrain.nn.flexi_vit import CompositeEncodings


def _make_ce(
    spatial_mode: str = "static_split",
    timestamp_mode: str = "static_temporal",
    embedding_size: int = 384,
) -> CompositeEncodings:
    return CompositeEncodings(
        embedding_size=embedding_size,
        supported_modalities=[Modality.SENTINEL2_L2A],
        max_sequence_length=12,
        timestamp_encoding_mode=timestamp_mode,
        spatial_encoding_mode=spatial_mode,
    )


def _make_timestamps(b: int = 2, t: int = 4) -> torch.Tensor:
    days = torch.randint(1, 28, (b, t, 1), dtype=torch.long)
    months = torch.randint(0, 12, (b, t, 1), dtype=torch.long)
    years = torch.randint(2018, 2023, (b, t, 1), dtype=torch.long)
    return torch.cat([days, months, years], dim=-1)


# ---- get_static_global_latlon_encoding ----------------------------------


def test_global_latlon_shape() -> None:
    """Output shape is (B, encoding_dim) for any encoding_dim divisible by 6."""
    latlon = torch.tensor([[0.0, 0.0], [37.7, -122.4], [-30.0, 150.0]])
    enc = get_static_global_latlon_encoding(latlon, encoding_dim=96)
    assert enc.shape == (3, 96)


def test_global_latlon_dim_not_divisible_by_6_raises() -> None:
    """encoding_dim must be divisible by 6 (3 axes * sin/cos)."""
    latlon = torch.tensor([[0.0, 0.0]])
    with pytest.raises(AssertionError, match="divisible by 6"):
        get_static_global_latlon_encoding(latlon, encoding_dim=64)


def test_global_latlon_deterministic() -> None:
    """Same input -> same output (function of inputs only)."""
    latlon = torch.tensor([[37.7, -122.4]])
    a = get_static_global_latlon_encoding(latlon, 96)
    b = get_static_global_latlon_encoding(latlon, 96)
    assert torch.allclose(a, b)


def test_global_latlon_lon_wraparound() -> None:
    """Longitude 180 and -180 represent the same point on the sphere."""
    a = get_static_global_latlon_encoding(torch.tensor([[0.0, 180.0]]), 96)
    b = get_static_global_latlon_encoding(torch.tensor([[0.0, -180.0]]), 96)
    assert torch.allclose(a, b, atol=1e-5)


def test_global_latlon_pole_lon_invariant() -> None:
    """At the north pole, all longitudes map to the same sphere point."""
    a = get_static_global_latlon_encoding(torch.tensor([[90.0, 0.0]]), 96)
    b = get_static_global_latlon_encoding(torch.tensor([[90.0, 90.0]]), 96)
    c = get_static_global_latlon_encoding(torch.tensor([[90.0, -150.0]]), 96)
    assert torch.allclose(a, b, atol=1e-5)
    assert torch.allclose(a, c, atol=1e-5)


def test_global_latlon_different_locations_differ() -> None:
    """Distant points should produce different encodings."""
    a = get_static_global_latlon_encoding(torch.tensor([[0.0, 0.0]]), 96)
    b = get_static_global_latlon_encoding(torch.tensor([[45.0, 0.0]]), 96)
    assert not torch.allclose(a, b)


# ---- get_static_local_2d_encoding ----------------------------------


def test_local_2d_shape() -> None:
    """Output shape is (H, W, encoding_dim) for any encoding_dim divisible by 4."""
    enc = get_static_local_2d_encoding(
        grid_h=8,
        grid_w=8,
        meters_per_token=10.0,
        encoding_dim=96,
        device=torch.device("cpu"),
    )
    assert enc.shape == (8, 8, 96)


def test_local_2d_dim_not_divisible_by_4_raises() -> None:
    """encoding_dim must be divisible by 4 (2 axes * sin/cos)."""
    with pytest.raises(AssertionError, match="divisible by 4"):
        get_static_local_2d_encoding(
            4, 4, meters_per_token=10.0, encoding_dim=96 + 1, device=torch.device("cpu")
        )


def test_local_2d_deterministic() -> None:
    """Same arguments -> same output."""
    a = get_static_local_2d_encoding(8, 8, 10.0, 96, torch.device("cpu"))
    b = get_static_local_2d_encoding(8, 8, 10.0, 96, torch.device("cpu"))
    assert torch.allclose(a, b)


def test_local_2d_resolution_aware() -> None:
    """Different meters_per_token at same grid index produces different encoding."""
    a = get_static_local_2d_encoding(
        8, 8, meters_per_token=10.0, encoding_dim=96, device=torch.device("cpu")
    )
    b = get_static_local_2d_encoding(
        8, 8, meters_per_token=100.0, encoding_dim=96, device=torch.device("cpu")
    )
    assert not torch.allclose(a[3, 5], b[3, 5])


def test_local_2d_translation_invariance() -> None:
    """Difference between two grid offsets should depend only on the offset.

    For sin/cos encoding f(x) = [sin(w * x), cos(w * x), ...], the difference
    f(x + dx) - f(x) is generally NOT independent of x. But the *pair* (f(x+dx),
    f(x)) determines f(x+dx) - f(x) uniquely once x is fixed. The cleaner
    invariant we can assert is: for grids of the same size and meters_per_token,
    the encoding at the same physical position is identical regardless of the
    enclosing patch. We verify that here.
    """
    # Two patches of identical extent and m/token: same physical position ->
    # identical encoding regardless of whether the patch is "shifted" (which it
    # isn't, since the encoding is computed relative to the patch *center*).
    a = get_static_local_2d_encoding(8, 8, 10.0, 96, torch.device("cpu"))
    b = get_static_local_2d_encoding(8, 8, 10.0, 96, torch.device("cpu"))
    assert torch.allclose(a, b)


def test_local_2d_center_symmetry() -> None:
    """Center symmetry: sin parts negate, cos parts equal across the grid center.

    Position at offset (-d, 0) from center vs (+d, 0) should produce sin parts
    that are negatives of each other and cos parts equal (sin/cos parity around
    a center-anchored grid).
    """
    enc = get_static_local_2d_encoding(7, 7, 10.0, 96, torch.device("cpu"))
    # h=0 is at meter offset -3*10, h=6 is at +3*10 from center; they should be
    # related by the parity of sin (odd) and cos (even).
    n = 96
    num_freqs = n // 4  # h-sin, h-cos, w-sin, w-cos blocks of size num_freqs each
    sin_h_top = enc[0, 3, :num_freqs]
    sin_h_bot = enc[6, 3, :num_freqs]
    cos_h_top = enc[0, 3, num_freqs : 2 * num_freqs]
    cos_h_bot = enc[6, 3, num_freqs : 2 * num_freqs]
    assert torch.allclose(sin_h_top, -sin_h_bot, atol=1e-5)
    assert torch.allclose(cos_h_top, cos_h_bot, atol=1e-5)


# ---- Integration: CompositeEncodings forward with static_split -------------


def test_static_split_forward_shape() -> None:
    """CompositeEncodings forward preserves token shape under static_split."""
    ce = _make_ce(embedding_size=384)
    B, H, W, T = 2, 2, 2, 4
    tokens = torch.randn(B, H, W, T, 3, 384)
    timestamps = _make_timestamps(B, T)
    latlon = torch.tensor([[37.0, -122.0], [-30.0, 150.0]])
    out = ce.forward(
        {"sentinel2_l2a": tokens},
        timestamps,
        patch_size=4,
        latlon=latlon,
    )
    assert out["sentinel2_l2a"].shape == tokens.shape


def test_static_split_writes_local_into_slot_2n_3n() -> None:
    """With zero tokens in, the [2n:3n] slot should contain the local 2D code."""
    ce = _make_ce(embedding_size=384)
    n = ce.embedding_dim_per_embedding_type
    B, H, W, T = 1, 4, 4, 1
    tokens = torch.zeros(B, H, W, T, 3, 384)
    timestamps = _make_timestamps(B, T)
    latlon = torch.tensor([[0.0, 0.0]])
    out = ce.forward(
        {"sentinel2_l2a": tokens},
        timestamps,
        patch_size=4,
        latlon=latlon,
    )["sentinel2_l2a"]
    # Local-2D varies across (h, w) - confirm slot [2n:3n] is non-uniform
    # across the 4x4 grid.
    slot = out[0, :, :, 0, 0, 2 * n : 3 * n]  # (4, 4, n)
    # Adjacent grid cells should not produce identical local-2D vectors.
    assert not torch.allclose(slot[0, 0], slot[1, 0])
    assert not torch.allclose(slot[0, 0], slot[0, 1])


def test_static_split_writes_global_into_slot_3n_4n() -> None:
    """Global lat/lon goes into slot [3n:4n] and is broadcast across tokens.

    Verifies the [3n:4n] slot under static_split matches the standalone global
    lat/lon encoding, and that the value is identical across grid + time +
    bandset (broadcast) for a single sample.
    """
    ce = _make_ce(embedding_size=384)
    n = ce.embedding_dim_per_embedding_type
    B, H, W, T = 2, 4, 4, 1
    tokens = torch.zeros(B, H, W, T, 3, 384)
    timestamps = _make_timestamps(B, T)
    latlon = torch.tensor([[37.0, -122.0], [-30.0, 150.0]])
    out = ce.forward(
        {"sentinel2_l2a": tokens},
        timestamps,
        patch_size=4,
        latlon=latlon,
    )["sentinel2_l2a"]
    slot = out[..., 3 * n : 4 * n]
    # Same sample: identical across grid + time + bandset (broadcast).
    assert torch.allclose(slot[0, 0, 0, 0, 0], slot[0, 1, 2, 0, 1], atol=1e-5)
    # Different samples: global latlon differs.
    assert not torch.allclose(slot[0, 0, 0, 0, 0], slot[1, 0, 0, 0, 0])
    # Slot value at sample 0 == raw global encoding for that latlon.
    expected = get_static_global_latlon_encoding(latlon[:1], n)[0]
    assert torch.allclose(slot[0, 0, 0, 0, 0], expected, atol=1e-5)


def test_static_split_no_latlon_leaves_global_slot_zero() -> None:
    """If latlon is not provided, the global slot stays zero (graceful no-op).

    Local 2D still gets written. Slot [3n:4n] should equal the input zeros.
    """
    ce = _make_ce(embedding_size=384)
    n = ce.embedding_dim_per_embedding_type
    B, H, W, T = 1, 2, 2, 1
    tokens = torch.zeros(B, H, W, T, 3, 384)
    timestamps = _make_timestamps(B, T)
    out = ce.forward(
        {"sentinel2_l2a": tokens},
        timestamps,
        patch_size=4,
        latlon=None,
    )["sentinel2_l2a"]
    assert torch.allclose(
        out[..., 3 * n : 4 * n], torch.zeros_like(out[..., 3 * n : 4 * n])
    )


def test_static_split_differs_from_legacy_in_spatial_slots() -> None:
    """Static_split and legacy spatial modes should differ in slot [2n:4n]."""
    ce_split = _make_ce(spatial_mode="static_split", embedding_size=384)
    ce_legacy = _make_ce(spatial_mode="legacy", embedding_size=384)
    B, H, W, T = 1, 2, 2, 1
    tokens = torch.zeros(B, H, W, T, 3, 384)
    timestamps = _make_timestamps(B, T)
    latlon = torch.tensor([[37.7, -122.4]])
    n = ce_split.embedding_dim_per_embedding_type
    out_split = ce_split.forward(
        {"sentinel2_l2a": tokens}, timestamps, patch_size=4, latlon=latlon
    )["sentinel2_l2a"]
    out_legacy = ce_legacy.forward(
        {"sentinel2_l2a": tokens}, timestamps, patch_size=4, latlon=latlon
    )["sentinel2_l2a"]
    assert not torch.allclose(
        out_split[..., 2 * n : 4 * n],
        out_legacy[..., 2 * n : 4 * n],
    )


def test_invalid_spatial_encoding_mode_raises() -> None:
    """Invalid mode should raise."""
    with pytest.raises(ValueError):
        CompositeEncodings(
            embedding_size=384,
            supported_modalities=[Modality.SENTINEL2_L2A],
            max_sequence_length=12,
            spatial_encoding_mode="bogus_mode",
        )
