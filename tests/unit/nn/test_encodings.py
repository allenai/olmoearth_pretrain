"""Unit tests for different encodings of data."""

import pytest
import torch

from olmoearth_pretrain.nn.encodings import (
    apply_2d_rope,
    apply_2d_rope_mixed,
    get_1d_sincos_pos_encoding,
    get_2d_sincos_pos_encoding,
    get_2d_sincos_pos_encoding_with_resolution,
    get_month_encoding_table,
    init_2d_rope_mixed_freqs,
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


def test_init_2d_rope_mixed_freqs_shape_and_finiteness() -> None:
    """Init should produce (2, H, D/2) finite frequencies."""
    head_dim, num_heads = 16, 4
    freqs = init_2d_rope_mixed_freqs(head_dim, num_heads, base=10.0, rotate=True)
    assert freqs.shape == (2, num_heads, head_dim // 2)
    assert torch.isfinite(freqs).all()


def test_init_2d_rope_mixed_freqs_no_rotation_deterministic() -> None:
    """With rotate=False the init should be deterministic across heads."""
    freqs = init_2d_rope_mixed_freqs(16, 3, base=10.0, rotate=False)
    for h in range(1, freqs.shape[1]):
        assert torch.allclose(freqs[:, h], freqs[:, 0])


def test_init_2d_rope_mixed_freqs_rejects_bad_head_dim() -> None:
    """head_dim not divisible by 4 should raise."""
    with pytest.raises(ValueError):
        init_2d_rope_mixed_freqs(head_dim=6, num_heads=2)


def test_apply_2d_rope_mixed_zero_positions_identity() -> None:
    """Zero positions -> all rotation angles are zero -> identity."""
    head_dim, num_heads, n = 8, 2, 4
    x = torch.randn(2, num_heads, n, head_dim)
    positions = torch.zeros(2, n, 2)
    freqs = init_2d_rope_mixed_freqs(head_dim, num_heads)
    out = apply_2d_rope_mixed(x, positions, freqs)
    assert torch.allclose(out, x, atol=1e-6, rtol=1e-6)


def test_apply_2d_rope_mixed_zero_freqs_identity() -> None:
    """Zero frequencies -> rotation angle is zero -> identity."""
    head_dim, num_heads, n = 8, 2, 4
    x = torch.randn(2, num_heads, n, head_dim)
    positions = torch.randn(2, n, 2)
    freqs = torch.zeros(2, num_heads, head_dim // 2)
    out = apply_2d_rope_mixed(x, positions, freqs)
    assert torch.allclose(out, x, atol=1e-6, rtol=1e-6)


def test_apply_2d_rope_mixed_preserves_norms() -> None:
    """Rotation in each complex pair must preserve per-pair norms."""
    head_dim, num_heads, n = 16, 3, 5
    x = torch.randn(2, num_heads, n, head_dim)
    positions = torch.randn(2, n, 2)
    freqs = init_2d_rope_mixed_freqs(head_dim, num_heads)
    out = apply_2d_rope_mixed(x, positions, freqs)
    assert torch.allclose(out.norm(dim=-1), x.norm(dim=-1), atol=1e-5, rtol=1e-5)


def test_apply_2d_rope_mixed_packed_shape() -> None:
    """Packed flash-attention layout (N, H, D) should be supported."""
    head_dim, num_heads, n = 8, 2, 5
    x = torch.randn(n, num_heads, head_dim)
    positions = torch.arange(n * 2, dtype=torch.float32).reshape(n, 2)
    freqs = init_2d_rope_mixed_freqs(head_dim, num_heads)
    out = apply_2d_rope_mixed(x, positions, freqs)
    assert out.shape == x.shape


def test_apply_2d_rope_mixed_relative_position_invariance() -> None:
    """q·k inner product should depend only on relative (row, col) offsets."""
    head_dim, num_heads = 8, 2
    freqs = init_2d_rope_mixed_freqs(head_dim, num_heads)
    q = torch.randn(num_heads, head_dim)
    k = torch.randn(num_heads, head_dim)

    p_a = torch.tensor([[0.0, 0.0], [3.0, 4.0]])  # (N=2, 2)
    p_b = p_a + torch.tensor([[1.5, -2.0]])

    # Pack as (N, H, D) so we can reuse the packed rope path.
    def attn(p: torch.Tensor) -> torch.Tensor:
        q_ = q.unsqueeze(0).expand(2, -1, -1).clone()  # (N, H, D)
        k_ = k.unsqueeze(0).expand(2, -1, -1).clone()
        q_rot = apply_2d_rope_mixed(q_, p, freqs)
        k_rot = apply_2d_rope_mixed(k_, p, freqs)
        # q[0] vs k[1] interaction per head
        return (q_rot[0] * k_rot[1]).sum(dim=-1)

    assert torch.allclose(attn(p_a), attn(p_b), atol=1e-5, rtol=1e-5)


def test_apply_2d_rope_mixed_freqs_gradient_flow() -> None:
    """Gradients should flow back into the learnable freqs."""
    head_dim, num_heads, n = 8, 2, 4
    x = torch.randn(1, num_heads, n, head_dim)
    positions = torch.randn(1, n, 2)
    freqs = init_2d_rope_mixed_freqs(head_dim, num_heads).clone().requires_grad_(True)
    out = apply_2d_rope_mixed(x, positions, freqs)
    out.sum().backward()
    assert freqs.grad is not None
    assert torch.isfinite(freqs.grad).all()
    assert freqs.grad.abs().sum() > 0


def test_apply_2d_rope_mixed_rejects_freqs_shape_mismatch() -> None:
    """Mismatched num_heads in freqs should raise."""
    head_dim, num_heads, n = 8, 2, 4
    x = torch.randn(1, num_heads, n, head_dim)
    positions = torch.zeros(1, n, 2)
    # Wrong num_heads (3 instead of 2)
    freqs = init_2d_rope_mixed_freqs(head_dim, num_heads=3)
    with pytest.raises(ValueError):
        apply_2d_rope_mixed(x, positions, freqs)
