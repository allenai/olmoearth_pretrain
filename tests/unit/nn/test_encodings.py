"""Unit tests for different encodings of data."""

import pytest
import torch

from olmoearth_pretrain.nn.encodings import (
    LatLonMLP,
    TimestampMLP,
    compute_per_token_latlon,
    get_1d_sincos_pos_encoding,
    get_2d_sincos_pos_encoding,
    get_2d_sincos_pos_encoding_with_resolution,
    get_month_encoding_table,
    get_timestamp_encoding,
    latlon_to_learned_input,
    timestamps_to_learned_input,
)


def test_get_1d_sincos_pos_encoding() -> None:
    """Test that the 1D sinusoidal position encoding is correct."""
    atol = 1e-4
    rtol = 1e-4
    expected_output = torch.tensor(
        [
            [0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 1.0000, 1.0000, 1.0000],
            [0.8415, 0.5332, 0.3110, 0.1769, 0.5403, 0.8460, 0.9504, 0.9842],
            [0.9093, 0.9021, 0.5911, 0.3482, -0.4161, 0.4315, 0.8066, 0.9374],
            [0.1411, 0.9933, 0.8126, 0.5085, -0.9900, -0.1160, 0.5828, 0.8610],
        ]
    )
    encoding_dim = 8
    pos = torch.tensor([0, 1, 2, 3])
    encoding = get_1d_sincos_pos_encoding(pos, encoding_dim)
    assert encoding.shape == (4, encoding_dim)
    assert torch.allclose(encoding, expected_output, atol=atol, rtol=rtol)


def test_get_2d_sincos_pos_encoding() -> None:
    """Test that the 2D sinusoidal position encoding is correct."""
    atol = 1e-4
    rtol = 1e-4
    expected_output = torch.tensor(
        [
            [0.0000, 0.0000, 1.0000, 1.0000, 0.0000, 0.0000, 1.0000, 1.0000],
            [0.8415, 0.3110, 0.5403, 0.9504, 0.0000, 0.0000, 1.0000, 1.0000],
            [0.9093, 0.5911, -0.4161, 0.8066, 0.0000, 0.0000, 1.0000, 1.0000],
            [0.0000, 0.0000, 1.0000, 1.0000, 0.8415, 0.3110, 0.5403, 0.9504],
            [0.8415, 0.3110, 0.5403, 0.9504, 0.8415, 0.3110, 0.5403, 0.9504],
            [0.9093, 0.5911, -0.4161, 0.8066, 0.8415, 0.3110, 0.5403, 0.9504],
            [0.0000, 0.0000, 1.0000, 1.0000, 0.9093, 0.5911, -0.4161, 0.8066],
            [0.8415, 0.3110, 0.5403, 0.9504, 0.9093, 0.5911, -0.4161, 0.8066],
            [0.9093, 0.5911, -0.4161, 0.8066, 0.9093, 0.5911, -0.4161, 0.8066],
            [0.0000, 0.0000, 1.0000, 1.0000, 0.0000, 0.0000, 1.0000, 1.0000],
            [0.8415, 0.3110, 0.5403, 0.9504, 0.0000, 0.0000, 1.0000, 1.0000],
            [0.9093, 0.5911, -0.4161, 0.8066, 0.0000, 0.0000, 1.0000, 1.0000],
            [0.0000, 0.0000, 1.0000, 1.0000, 0.8415, 0.3110, 0.5403, 0.9504],
            [0.8415, 0.3110, 0.5403, 0.9504, 0.8415, 0.3110, 0.5403, 0.9504],
            [0.9093, 0.5911, -0.4161, 0.8066, 0.8415, 0.3110, 0.5403, 0.9504],
            [0.0000, 0.0000, 1.0000, 1.0000, 0.9093, 0.5911, -0.4161, 0.8066],
            [0.8415, 0.3110, 0.5403, 0.9504, 0.9093, 0.5911, -0.4161, 0.8066],
            [0.9093, 0.5911, -0.4161, 0.8066, 0.9093, 0.5911, -0.4161, 0.8066],
        ]
    )
    encoding_dim = 8
    grid = torch.tensor(
        [
            [
                [[0.0, 1.0, 2.0], [0.0, 1.0, 2.0], [0.0, 1.0, 2.0]],
                [[0.0, 1.0, 2.0], [0.0, 1.0, 2.0], [0.0, 1.0, 2.0]],
            ],
            [
                [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
                [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
            ],
        ]
    )
    encoding = get_2d_sincos_pos_encoding(grid, encoding_dim)
    assert encoding.shape == (18, encoding_dim)
    assert torch.allclose(encoding, expected_output, atol=atol, rtol=rtol)


def test_get_2d_sincos_pos_encoding_with_resolution() -> None:
    """Test that the 2D sinusoidal position encoding with resolution is correct."""
    atol = 1e-4
    rtol = 1e-4
    expected_output = torch.tensor(
        [
            [
                [0.0000, 1.0000, 0.0000, 1.0000],
                [0.9093, -0.4161, 0.0000, 1.0000],
                [0.0000, 1.0000, 0.9093, -0.4161],
                [0.9093, -0.4161, 0.9093, -0.4161],
            ],
            [
                [0.0000, 1.0000, 0.0000, 1.0000],
                [0.9093, -0.4161, 0.0000, 1.0000],
                [0.0000, 1.0000, 0.9093, -0.4161],
                [0.9093, -0.4161, 0.9093, -0.4161],
            ],
        ]
    )
    encoding_dim = 4
    grid_size = 2
    res = torch.tensor([2.0, 2.0])
    device = torch.device("cpu")
    encoding = get_2d_sincos_pos_encoding_with_resolution(
        grid_size, res, encoding_dim, device
    )
    assert encoding.shape == (2, grid_size * grid_size, encoding_dim)
    assert torch.allclose(encoding, expected_output, atol=atol, rtol=rtol)


def test_get_month_encoding_table() -> None:
    """Test that the month encoding table is correct."""
    atol = 1e-4
    rtol = 1e-4
    expected_output = torch.tensor(
        [
            [0.0000e00, 1.0000e00],
            [5.0000e-01, 8.6603e-01],
            [8.6603e-01, 5.0000e-01],
            [1.0000e00, -4.3711e-08],
            [8.6603e-01, -5.0000e-01],
            [5.0000e-01, -8.6603e-01],
            [-8.7423e-08, -1.0000e00],
            [-5.0000e-01, -8.6603e-01],
            [-8.6603e-01, -5.0000e-01],
            [-1.0000e00, 1.1925e-08],
            [-8.6603e-01, 5.0000e-01],
            [-5.0000e-01, 8.6603e-01],
        ]
    )
    encoding_dim = 2
    encoding = get_month_encoding_table(encoding_dim)
    assert encoding.shape == (12, encoding_dim)
    assert torch.allclose(encoding, expected_output, atol=atol, rtol=rtol)


def test_get_timestamp_encoding_output_shape() -> None:
    """Test that the timestamp encoding has the correct output shape."""
    B, T, D = 2, 6, 8
    # day=15, month=5 (0-indexed), year=2021
    timestamps = torch.tensor([[[15, 5, 2021]] * T] * B)
    encoding = get_timestamp_encoding(timestamps, D)
    assert encoding.shape == (B, T, D)


def test_get_timestamp_encoding_deterministic() -> None:
    """Test that the same input always produces the same output."""
    timestamps = torch.tensor([[[1, 0, 2020], [15, 6, 2021]]])
    enc1 = get_timestamp_encoding(timestamps, 8)
    enc2 = get_timestamp_encoding(timestamps, 8)
    assert torch.equal(enc1, enc2)


def test_get_timestamp_encoding_different_dates_differ() -> None:
    """Test that different timestamps produce different encodings."""
    ts1 = torch.tensor([[[1, 0, 2020], [1, 0, 2020]]])  # Jan 2020
    ts2 = torch.tensor([[[1, 6, 2021], [1, 6, 2021]]])  # Jul 2021
    enc1 = get_timestamp_encoding(ts1, 8)
    enc2 = get_timestamp_encoding(ts2, 8)
    assert not torch.allclose(enc1, enc2)


def test_get_timestamp_encoding_dim_must_be_even() -> None:
    """Test that odd encoding_dim raises an assertion."""
    timestamps = torch.tensor([[[1, 0, 2020]]])
    with pytest.raises(AssertionError):
        get_timestamp_encoding(timestamps, 7)


def test_get_timestamp_encoding_values_bounded() -> None:
    """Test that encoding values are in [-1, 1] (sin/cos range)."""
    timestamps = torch.tensor([[[1, 0, 2015], [28, 11, 2025]]])
    encoding = get_timestamp_encoding(timestamps, 16)
    assert encoding.min() >= -1.0
    assert encoding.max() <= 1.0


# --- timestamps_to_learned_input tests ---


def test_timestamps_to_learned_input_shape() -> None:
    """Output shape should match input batch and time dims with 3 features."""
    timestamps = torch.tensor([[[1, 0, 2020], [15, 6, 2021]]])
    out = timestamps_to_learned_input(timestamps)
    assert out.shape == (1, 2, 3)


def test_timestamps_to_learned_input_jan_2021() -> None:
    """Jan 1 2021 should have fractional year ~1.0 and sin/cos near the cycle start."""
    timestamps = torch.tensor([[[1, 0, 2021]]])
    out = timestamps_to_learned_input(timestamps)
    frac = out[0, 0, 0].item()
    # year=2021 + (0*30.4375 + 1)/365.25 - 2020 = ~1.00274
    assert abs(frac - 1.00274) < 0.01


def test_timestamps_to_learned_input_july_2020() -> None:
    """July 2020 should have fractional year ~0.5."""
    # month=6 (0-indexed July), day=1
    timestamps = torch.tensor([[[1, 6, 2020]]])
    out = timestamps_to_learned_input(timestamps)
    frac = out[0, 0, 0].item()
    # (6*30.4375 + 1)/365.25 = ~0.502
    assert abs(frac - 0.5) < 0.05


def test_timestamps_to_learned_input_sin_cos_consistency() -> None:
    """Sin and cos outputs should match torch.sin/cos of 2*pi*frac."""
    import numpy as np

    timestamps = torch.tensor([[[15, 3, 2022]]])
    out = timestamps_to_learned_input(timestamps)
    frac = out[0, 0, 0].item()
    assert abs(out[0, 0, 1].item() - np.sin(2 * np.pi * frac)) < 1e-5
    assert abs(out[0, 0, 2].item() - np.cos(2 * np.pi * frac)) < 1e-5


def test_timestamps_to_learned_input_different_dates_differ() -> None:
    """Different dates should produce different representations."""
    ts1 = torch.tensor([[[1, 0, 2020]]])
    ts2 = torch.tensor([[[1, 6, 2022]]])
    out1 = timestamps_to_learned_input(ts1)
    out2 = timestamps_to_learned_input(ts2)
    assert not torch.allclose(out1, out2)


# --- TimestampMLP tests ---


def test_timestamp_mlp_output_shape() -> None:
    """MLP output should have the correct shape."""
    mlp = TimestampMLP(output_dim=96, hidden_dim=64)
    timestamps = torch.tensor([[[1, 0, 2020], [15, 6, 2021]]] * 2)  # (2, 2, 3)
    out = mlp(timestamps)
    assert out.shape == (2, 2, 96)


def test_timestamp_mlp_is_differentiable() -> None:
    """Gradients should flow through the MLP."""
    mlp = TimestampMLP(output_dim=32, hidden_dim=16)
    timestamps = torch.tensor([[[1, 0, 2020], [15, 6, 2021]]], dtype=torch.float32)
    out = mlp(timestamps)
    out.sum().backward()
    for p in mlp.parameters():
        assert p.grad is not None


def test_timestamp_mlp_custom_hidden_dim() -> None:
    """Different hidden dims should produce different parameter counts."""
    mlp_small = TimestampMLP(output_dim=32, hidden_dim=16)
    mlp_large = TimestampMLP(output_dim=32, hidden_dim=64)
    params_small = sum(p.numel() for p in mlp_small.parameters())
    params_large = sum(p.numel() for p in mlp_large.parameters())
    assert params_large > params_small


# --- latlon_to_learned_input tests ---


def test_latlon_to_learned_input_shape() -> None:
    """Output should have 3 + 6*num_freqs features."""
    latlon_2d = torch.tensor([[45.0, 90.0], [0.0, -180.0]])
    # Default num_freqs=20 -> 3 + 120 = 123
    assert latlon_to_learned_input(latlon_2d).shape == (2, 123)

    latlon_4d = torch.randn(2, 4, 4, 2)
    assert latlon_to_learned_input(latlon_4d).shape == (2, 4, 4, 123)

    # Custom num_freqs
    assert latlon_to_learned_input(latlon_2d, num_freqs=4).shape == (2, 27)


def test_latlon_to_learned_input_unit_sphere() -> None:
    """First 3 features should be unit sphere coordinates (x, y, z)."""
    latlon = torch.tensor([[0.0, 0.0]])  # equator, prime meridian
    out = latlon_to_learned_input(latlon)
    # (cos(0)*cos(0), cos(0)*sin(0), sin(0)) = (1, 0, 0)
    assert abs(out[0, 0].item() - 1.0) < 1e-5
    assert abs(out[0, 1].item() - 0.0) < 1e-5
    assert abs(out[0, 2].item() - 0.0) < 1e-5

    latlon_pole = torch.tensor([[90.0, 0.0]])  # north pole
    out_pole = latlon_to_learned_input(latlon_pole)
    # (cos(90)*cos(0), cos(90)*sin(0), sin(90)) = (0, 0, 1)
    assert abs(out_pole[0, 0].item()) < 1e-5
    assert abs(out_pole[0, 1].item()) < 1e-5
    assert abs(out_pole[0, 2].item() - 1.0) < 1e-5


def test_latlon_to_learned_input_lon_wraparound() -> None:
    """lon=-180 and lon=180 should map to the same point on the unit sphere."""
    ll1 = torch.tensor([[0.0, -180.0]])
    ll2 = torch.tensor([[0.0, 180.0]])
    out1 = latlon_to_learned_input(ll1)
    out2 = latlon_to_learned_input(ll2)
    # Raw sphere coordinates (first 3) should be identical
    assert torch.allclose(out1[0, :3], out2[0, :3], atol=1e-5)
    # Full encoding close within float precision amplified by high freqs
    assert torch.allclose(out1, out2, atol=0.5)


def test_latlon_to_learned_input_nearby_points_differ() -> None:
    """Points 10m apart should produce different high-frequency encodings."""
    # ~10m offset in latitude ≈ 0.00009 degrees
    ll1 = torch.tensor([[45.0, 10.0]])
    ll2 = torch.tensor([[45.00009, 10.0]])
    out1 = latlon_to_learned_input(ll1, num_freqs=20)
    out2 = latlon_to_learned_input(ll2, num_freqs=20)
    # Should NOT be close — high frequencies resolve this
    assert not torch.allclose(out1, out2, atol=0.01)


# --- compute_per_token_latlon tests ---


def test_compute_per_token_latlon_shape() -> None:
    """Output shape should be (B, H, W, 2)."""
    latlon = torch.tensor([[45.0, 10.0], [-30.0, 120.0]])
    out = compute_per_token_latlon(latlon, grid_h=4, grid_w=4, meters_per_token=160.0)
    assert out.shape == (2, 4, 4, 2)


def test_compute_per_token_latlon_center_matches_tile() -> None:
    """For an odd grid, the center token should equal the tile center."""
    latlon = torch.tensor([[45.0, 10.0]])
    out = compute_per_token_latlon(latlon, grid_h=3, grid_w=3, meters_per_token=160.0)
    # Center is at (1, 1) for a 3x3 grid
    assert abs(out[0, 1, 1, 0].item() - 45.0) < 1e-4
    assert abs(out[0, 1, 1, 1].item() - 10.0) < 1e-4


def test_compute_per_token_latlon_direction() -> None:
    """Increasing h should decrease lat (south), increasing w should increase lon (east)."""
    latlon = torch.tensor([[0.0, 0.0]])  # equator, prime meridian
    out = compute_per_token_latlon(
        latlon, grid_h=3, grid_w=3, meters_per_token=111320.0
    )
    # h=0 is north of h=2
    assert out[0, 0, 1, 0].item() > out[0, 2, 1, 0].item()
    # w=2 is east of w=0
    assert out[0, 1, 2, 1].item() > out[0, 1, 0, 1].item()


# --- LatLonMLP tests ---


def test_latlon_mlp_output_shape() -> None:
    """MLP should work with both (B, 2) and (B, H, W, 2) inputs."""
    mlp = LatLonMLP(output_dim=48, hidden_dim=32)
    out_2d = mlp(torch.tensor([[45.0, 10.0], [0.0, 0.0]]))
    assert out_2d.shape == (2, 48)

    out_4d = mlp(torch.randn(2, 4, 4, 2))
    assert out_4d.shape == (2, 4, 4, 48)


def test_latlon_mlp_is_differentiable() -> None:
    """Gradients should flow through the MLP."""
    mlp = LatLonMLP(output_dim=32, hidden_dim=16)
    latlon = torch.tensor([[45.0, 10.0]], requires_grad=False)
    out = mlp(latlon)
    out.sum().backward()
    for p in mlp.parameters():
        assert p.grad is not None
