"""Unit tests for different encodings of data."""

import torch

from olmoearth_pretrain.nn.encodings import (
    apply_2d_rope,
    build_window_mask,
    get_1d_sincos_pos_encoding,
    get_2d_sincos_pos_encoding,
    get_2d_sincos_pos_encoding_with_resolution,
    get_month_encoding_table,
)


def _grid_positions(side: int) -> torch.Tensor:
    """A (1, side*side, 2) row-major integer (row, col) grid."""
    rows = torch.arange(side).repeat_interleave(side)
    cols = torch.arange(side).repeat(side)
    return torch.stack([rows, cols], dim=-1).float().unsqueeze(0)


def test_build_window_mask_centered_window() -> None:
    """A query attends exactly the Chebyshev neighbourhood of half_extent cells."""
    pos = _grid_positions(5)  # 5x5 grid, coords 0..4
    half = 1.0  # window side 3 (|d| <= 1)
    mask = build_window_mask(pos, pos, half)[0, 0]  # (25, 25)

    coords = pos[0]
    for i in range(25):
        for j in range(25):
            within = bool((coords[i] - coords[j]).abs().max() <= half + 1e-6)
            assert mask[i, j].item() == within


def test_build_window_mask_full_when_window_covers_grid() -> None:
    """When the window spans the whole grid the mask is all-True (== full attention)."""
    pos = _grid_positions(4)
    mask = build_window_mask(pos, pos, half_extent=10.0)
    assert mask.all()


def test_build_window_mask_global_tokens_attend_everywhere() -> None:
    """Global queries/keys bypass the window constraint."""
    pos = _grid_positions(4)
    n = pos.shape[1]
    q_global = torch.zeros(1, n, dtype=torch.bool)
    q_global[0, 0] = True  # first query is global
    k_global = torch.zeros(1, n, dtype=torch.bool)
    k_global[0, -1] = True  # last key is global
    mask = build_window_mask(
        pos, pos, half_extent=0.0, q_is_global=q_global, k_is_global=k_global
    )[0, 0]
    assert mask[0].all()  # global query sees all keys
    assert mask[:, -1].all()  # global key seen by all queries


def test_build_window_mask_respects_key_valid() -> None:
    """Invalid keys never participate, even inside the window."""
    pos = _grid_positions(4)
    n = pos.shape[1]
    key_valid = torch.ones(1, n, dtype=torch.bool)
    key_valid[0, 5] = False
    mask = build_window_mask(pos, pos, half_extent=10.0, key_valid=key_valid)[0, 0]
    assert not mask[:, 5].any()


def test_build_window_mask_no_empty_rows() -> None:
    """A starved query (no valid key in window) falls back to valid keys, never empty."""
    pos = _grid_positions(4)
    n = pos.shape[1]
    key_valid = torch.zeros(1, n, dtype=torch.bool)
    key_valid[0, 0] = True  # only one valid key, far from most queries
    mask = build_window_mask(pos, pos, half_extent=0.0, key_valid=key_valid)[0, 0]
    # Every query keeps at least one key (avoids NaN softmax); only valid keys allowed.
    assert mask.any(dim=-1).all()
    assert not mask[:, 1:].any()


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


def test_apply_2d_rope_zero_positions_identity() -> None:
    """Zero-valued positions should leave Q/K unchanged."""
    x = torch.randn(2, 3, 4, 8)
    positions = torch.zeros(2, 4, 2)
    out = apply_2d_rope(x, positions)
    assert torch.allclose(out, x)


def test_apply_2d_rope_preserves_norms() -> None:
    """RoPE should rotate feature pairs without changing vector norms."""
    x = torch.randn(2, 3, 4, 8)
    positions = torch.tensor(
        [
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            [[2.0, 3.0], [3.0, 2.0], [4.0, 1.0], [1.0, 4.0]],
        ]
    )
    out = apply_2d_rope(x, positions)
    assert torch.allclose(out.norm(dim=-1), x.norm(dim=-1), atol=1e-5, rtol=1e-5)


def test_apply_2d_rope_packed_shape() -> None:
    """Packed flash-attention layout should also be supported."""
    x = torch.randn(5, 2, 8)
    positions = torch.arange(10, dtype=torch.float32).reshape(5, 2)
    out = apply_2d_rope(x, positions)
    assert out.shape == x.shape
