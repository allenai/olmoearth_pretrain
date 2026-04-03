"""Unit tests for different encodings of data."""

import pytest
import torch

from olmoearth_pretrain.nn.encodings import (
    TimestampMLP,
    get_1d_sincos_pos_encoding,
    get_2d_sincos_pos_encoding,
    get_2d_sincos_pos_encoding_with_resolution,
    get_month_encoding_table,
    get_timestamp_encoding,
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
