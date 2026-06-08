"""Unit tests for different encodings of data."""

import pytest
import torch

from olmoearth_pretrain.nn.encodings import (
    apply_1d_rope,
    apply_2d_axial_rope,
    apply_2d_mixed_rope,
    apply_3d_axial_rope,
    apply_3d_mixed_rope,
    axial_3d_dim_split,
    get_1d_sincos_pos_encoding,
    get_2d_sincos_pos_encoding,
    get_2d_sincos_pos_encoding_with_resolution,
    get_month_encoding_table,
    init_2d_mixed_rope_freqs,
    init_3d_mixed_rope_freqs,
    timestamps_to_days,
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


def test_apply_2d_axial_rope_zero_positions_identity() -> None:
    """Zero-valued positions should leave Q/K unchanged."""
    x = torch.randn(2, 3, 4, 8)
    positions = torch.zeros(2, 4, 2)
    out = apply_2d_axial_rope(x, positions)
    assert torch.allclose(out, x)


def test_apply_2d_axial_rope_preserves_norms() -> None:
    """RoPE should rotate feature pairs without changing vector norms."""
    x = torch.randn(2, 3, 4, 8)
    positions = torch.tensor(
        [
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            [[2.0, 3.0], [3.0, 2.0], [4.0, 1.0], [1.0, 4.0]],
        ]
    )
    out = apply_2d_axial_rope(x, positions)
    assert torch.allclose(out.norm(dim=-1), x.norm(dim=-1), atol=1e-5, rtol=1e-5)


def test_apply_2d_axial_rope_packed_shape() -> None:
    """Packed flash-attention layout should also be supported."""
    x = torch.randn(5, 2, 8)
    positions = torch.arange(10, dtype=torch.float32).reshape(5, 2)
    out = apply_2d_axial_rope(x, positions)
    assert out.shape == x.shape


def test_init_2d_mixed_rope_freqs_shape_and_finiteness() -> None:
    """Init should produce (2, H, D/2) finite frequencies."""
    head_dim, num_heads = 16, 4
    freqs = init_2d_mixed_rope_freqs(head_dim, num_heads, base=10.0, rotate=True)
    assert freqs.shape == (2, num_heads, head_dim // 2)
    assert torch.isfinite(freqs).all()


def test_init_2d_mixed_rope_freqs_no_rotation_deterministic() -> None:
    """With rotate=False the init should be deterministic across heads."""
    freqs = init_2d_mixed_rope_freqs(16, 3, base=10.0, rotate=False)
    for h in range(1, freqs.shape[1]):
        assert torch.allclose(freqs[:, h], freqs[:, 0])


def test_init_2d_mixed_rope_freqs_rejects_bad_head_dim() -> None:
    """head_dim not divisible by 4 should raise."""
    with pytest.raises(ValueError):
        init_2d_mixed_rope_freqs(head_dim=6, num_heads=2)


def test_apply_2d_mixed_rope_zero_positions_identity() -> None:
    """Zero positions -> all rotation angles are zero -> identity."""
    head_dim, num_heads, n = 8, 2, 4
    x = torch.randn(2, num_heads, n, head_dim)
    positions = torch.zeros(2, n, 2)
    freqs = init_2d_mixed_rope_freqs(head_dim, num_heads)
    out = apply_2d_mixed_rope(x, positions, freqs)
    assert torch.allclose(out, x, atol=1e-6, rtol=1e-6)


def test_apply_2d_mixed_rope_zero_freqs_identity() -> None:
    """Zero frequencies -> rotation angle is zero -> identity."""
    head_dim, num_heads, n = 8, 2, 4
    x = torch.randn(2, num_heads, n, head_dim)
    positions = torch.randn(2, n, 2)
    freqs = torch.zeros(2, num_heads, head_dim // 2)
    out = apply_2d_mixed_rope(x, positions, freqs)
    assert torch.allclose(out, x, atol=1e-6, rtol=1e-6)


def test_apply_2d_mixed_rope_preserves_norms() -> None:
    """Rotation in each complex pair must preserve per-pair norms."""
    head_dim, num_heads, n = 16, 3, 5
    x = torch.randn(2, num_heads, n, head_dim)
    positions = torch.randn(2, n, 2)
    freqs = init_2d_mixed_rope_freqs(head_dim, num_heads)
    out = apply_2d_mixed_rope(x, positions, freqs)
    assert torch.allclose(out.norm(dim=-1), x.norm(dim=-1), atol=1e-5, rtol=1e-5)


def test_apply_2d_mixed_rope_packed_shape() -> None:
    """Packed flash-attention layout (N, H, D) should be supported."""
    head_dim, num_heads, n = 8, 2, 5
    x = torch.randn(n, num_heads, head_dim)
    positions = torch.arange(n * 2, dtype=torch.float32).reshape(n, 2)
    freqs = init_2d_mixed_rope_freqs(head_dim, num_heads)
    out = apply_2d_mixed_rope(x, positions, freqs)
    assert out.shape == x.shape


def test_apply_2d_mixed_rope_relative_position_invariance() -> None:
    """q·k inner product should depend only on relative (row, col) offsets."""
    head_dim, num_heads = 8, 2
    freqs = init_2d_mixed_rope_freqs(head_dim, num_heads)
    q = torch.randn(num_heads, head_dim)
    k = torch.randn(num_heads, head_dim)

    p_a = torch.tensor([[0.0, 0.0], [3.0, 4.0]])  # (N=2, 2)
    p_b = p_a + torch.tensor([[1.5, -2.0]])

    # Pack as (N, H, D) so we can reuse the packed rope path.
    def attn(p: torch.Tensor) -> torch.Tensor:
        q_ = q.unsqueeze(0).expand(2, -1, -1).clone()  # (N, H, D)
        k_ = k.unsqueeze(0).expand(2, -1, -1).clone()
        q_rot = apply_2d_mixed_rope(q_, p, freqs)
        k_rot = apply_2d_mixed_rope(k_, p, freqs)
        # q[0] vs k[1] interaction per head
        return (q_rot[0] * k_rot[1]).sum(dim=-1)

    assert torch.allclose(attn(p_a), attn(p_b), atol=1e-5, rtol=1e-5)


def test_apply_2d_mixed_rope_freqs_gradient_flow() -> None:
    """Gradients should flow back into the learnable freqs."""
    head_dim, num_heads, n = 8, 2, 4
    x = torch.randn(1, num_heads, n, head_dim)
    positions = torch.randn(1, n, 2)
    freqs = init_2d_mixed_rope_freqs(head_dim, num_heads).clone().requires_grad_(True)
    out = apply_2d_mixed_rope(x, positions, freqs)
    out.sum().backward()
    assert freqs.grad is not None
    assert torch.isfinite(freqs.grad).all()
    assert freqs.grad.abs().sum() > 0


def test_apply_2d_mixed_rope_rejects_freqs_shape_mismatch() -> None:
    """Mismatched num_heads in freqs should raise."""
    head_dim, num_heads, n = 8, 2, 4
    x = torch.randn(1, num_heads, n, head_dim)
    positions = torch.zeros(1, n, 2)
    # Wrong num_heads (3 instead of 2)
    freqs = init_2d_mixed_rope_freqs(head_dim, num_heads=3)
    with pytest.raises(ValueError):
        apply_2d_mixed_rope(x, positions, freqs)


# ----- 3D axial RoPE -----


def test_axial_3d_dim_split_default() -> None:
    """Default 25/37.5/37.5 split should produce 16/24/24 for head_dim=64."""
    assert axial_3d_dim_split(64, 0.25) == (16, 24, 24)


def test_axial_3d_dim_split_rejects_non_div_4_remaining() -> None:
    """Splits where (head_dim - d_t) is not divisible by 4 should raise."""
    # head_dim=10, frac=0.2 -> d_t=2, remaining=8 -> 4/4 OK
    assert axial_3d_dim_split(10, 0.2) == (2, 4, 4)
    # head_dim=10, frac=0.4 -> d_t=4, remaining=6 -> not divisible by 4
    with pytest.raises(ValueError):
        axial_3d_dim_split(10, 0.4)


def test_axial_3d_dim_split_rejects_zero_temporal() -> None:
    """Tiny fractions that round to d_t=0 should raise."""
    with pytest.raises(ValueError):
        axial_3d_dim_split(64, 0.01)


def test_apply_3d_axial_rope_zero_positions_identity() -> None:
    """Zero-valued positions should leave Q/K unchanged."""
    x = torch.randn(2, 3, 4, 16)
    positions = torch.zeros(2, 4, 3)
    out = apply_3d_axial_rope(x, positions, temporal_dim_frac=0.25)
    assert torch.allclose(out, x)


def test_apply_3d_axial_rope_preserves_norms() -> None:
    """3D RoPE should rotate feature pairs without changing vector norms."""
    x = torch.randn(2, 3, 4, 16)
    positions = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 0.0, 1.0], [3.0, 2.0, 1.0]],
            [[1.0, 2.0, 3.0], [4.0, 0.0, 1.0], [0.0, 5.0, 0.0], [2.0, 2.0, 2.0]],
        ]
    )
    out = apply_3d_axial_rope(x, positions, temporal_dim_frac=0.25)
    assert torch.allclose(out.norm(dim=-1), x.norm(dim=-1), atol=1e-5, rtol=1e-5)


def test_apply_3d_axial_rope_packed_shape() -> None:
    """Packed flash-attention layout (N, H, D) should also be supported."""
    x = torch.randn(5, 2, 16)
    positions = torch.arange(15, dtype=torch.float32).reshape(5, 3)
    out = apply_3d_axial_rope(x, positions, temporal_dim_frac=0.25)
    assert out.shape == x.shape


def test_apply_3d_axial_rope_temporal_only_when_spatial_zero() -> None:
    """With (row=col=0), output should match 1D RoPE applied to the temporal slice."""
    x = torch.randn(1, 2, 3, 16)
    t_vals = torch.tensor([[1.0, 2.0, 3.0]])
    positions = torch.stack(
        [t_vals, torch.zeros_like(t_vals), torch.zeros_like(t_vals)], dim=-1
    )
    out = apply_3d_axial_rope(x, positions, temporal_dim_frac=0.25)

    d_t, _, _ = axial_3d_dim_split(16, 0.25)
    # Spatial slices stay unchanged when row/col positions are zero.
    expected_t = apply_1d_rope(x[..., :d_t], t_vals[:, None, :], 10000.0)
    assert torch.allclose(out[..., :d_t], expected_t, atol=1e-5, rtol=1e-5)
    assert torch.allclose(out[..., d_t:], x[..., d_t:], atol=1e-5, rtol=1e-5)


def test_apply_3d_axial_rope_relative_invariance() -> None:
    """q.k inner product depends only on relative (t, row, col) offsets."""
    head_dim, num_heads = 16, 2
    q = torch.randn(num_heads, head_dim)
    k = torch.randn(num_heads, head_dim)

    p_a = torch.tensor([[0.0, 0.0, 0.0], [2.0, 3.0, 4.0]])
    p_b = p_a + torch.tensor([[1.0, -1.0, 2.0]])

    def attn(p: torch.Tensor) -> torch.Tensor:
        q_ = q.unsqueeze(0).expand(2, -1, -1).clone()
        k_ = k.unsqueeze(0).expand(2, -1, -1).clone()
        q_rot = apply_3d_axial_rope(q_, p, temporal_dim_frac=0.25)
        k_rot = apply_3d_axial_rope(k_, p, temporal_dim_frac=0.25)
        return (q_rot[0] * k_rot[1]).sum(dim=-1)

    assert torch.allclose(attn(p_a), attn(p_b), atol=1e-5, rtol=1e-5)


# ----- 3D mixed RoPE -----


def test_init_3d_mixed_rope_freqs_shape_and_finiteness() -> None:
    """Init should produce (3, H, D/2) finite frequencies."""
    head_dim, num_heads = 16, 4
    freqs = init_3d_mixed_rope_freqs(head_dim, num_heads, base=10.0, rotate=True)
    assert freqs.shape == (3, num_heads, head_dim // 2)
    assert torch.isfinite(freqs).all()


def test_init_3d_mixed_rope_freqs_rejects_bad_head_dim() -> None:
    """head_dim not divisible by 4 should raise."""
    with pytest.raises(ValueError):
        init_3d_mixed_rope_freqs(head_dim=6, num_heads=2)


def test_init_3d_mixed_rope_freqs_no_rotation_axial_directions() -> None:
    """Without rotation, init should have d1=t-axis, d2=row-axis -> col freqs all zero."""
    freqs = init_3d_mixed_rope_freqs(16, 3, base=10.0, rotate=False)
    # Axis 2 (col) should have zero frequencies under the deterministic init.
    assert torch.allclose(freqs[2], torch.zeros_like(freqs[2]))
    # All heads should match (no per-head randomness).
    for h in range(1, freqs.shape[1]):
        assert torch.allclose(freqs[:, h], freqs[:, 0])


def test_apply_3d_mixed_rope_zero_positions_identity() -> None:
    """Zero positions -> rotation angle is zero -> identity."""
    head_dim, num_heads, n = 8, 2, 4
    x = torch.randn(2, num_heads, n, head_dim)
    positions = torch.zeros(2, n, 3)
    freqs = init_3d_mixed_rope_freqs(head_dim, num_heads)
    out = apply_3d_mixed_rope(x, positions, freqs)
    assert torch.allclose(out, x, atol=1e-6, rtol=1e-6)


def test_apply_3d_mixed_rope_zero_freqs_identity() -> None:
    """Zero frequencies -> rotation angle is zero -> identity."""
    head_dim, num_heads, n = 8, 2, 4
    x = torch.randn(2, num_heads, n, head_dim)
    positions = torch.randn(2, n, 3)
    freqs = torch.zeros(3, num_heads, head_dim // 2)
    out = apply_3d_mixed_rope(x, positions, freqs)
    assert torch.allclose(out, x, atol=1e-6, rtol=1e-6)


def test_apply_3d_mixed_rope_preserves_norms() -> None:
    """Rotation in each complex pair must preserve per-pair norms."""
    head_dim, num_heads, n = 16, 3, 5
    x = torch.randn(2, num_heads, n, head_dim)
    positions = torch.randn(2, n, 3)
    freqs = init_3d_mixed_rope_freqs(head_dim, num_heads)
    out = apply_3d_mixed_rope(x, positions, freqs)
    assert torch.allclose(out.norm(dim=-1), x.norm(dim=-1), atol=1e-5, rtol=1e-5)


def test_apply_3d_mixed_rope_packed_shape() -> None:
    """Packed flash-attention layout (N, H, D) should be supported."""
    head_dim, num_heads, n = 8, 2, 5
    x = torch.randn(n, num_heads, head_dim)
    positions = torch.arange(n * 3, dtype=torch.float32).reshape(n, 3)
    freqs = init_3d_mixed_rope_freqs(head_dim, num_heads)
    out = apply_3d_mixed_rope(x, positions, freqs)
    assert out.shape == x.shape


def test_apply_3d_mixed_rope_relative_position_invariance() -> None:
    """q.k inner product should depend only on relative (t, row, col) offsets."""
    head_dim, num_heads = 8, 2
    freqs = init_3d_mixed_rope_freqs(head_dim, num_heads)
    q = torch.randn(num_heads, head_dim)
    k = torch.randn(num_heads, head_dim)

    p_a = torch.tensor([[0.0, 0.0, 0.0], [2.0, 3.0, 4.0]])
    p_b = p_a + torch.tensor([[1.5, -2.0, 0.5]])

    def attn(p: torch.Tensor) -> torch.Tensor:
        q_ = q.unsqueeze(0).expand(2, -1, -1).clone()
        k_ = k.unsqueeze(0).expand(2, -1, -1).clone()
        q_rot = apply_3d_mixed_rope(q_, p, freqs)
        k_rot = apply_3d_mixed_rope(k_, p, freqs)
        return (q_rot[0] * k_rot[1]).sum(dim=-1)

    assert torch.allclose(attn(p_a), attn(p_b), atol=1e-5, rtol=1e-5)


def test_apply_3d_mixed_rope_freqs_gradient_flow() -> None:
    """Gradients should flow back into the learnable freqs."""
    head_dim, num_heads, n = 8, 2, 4
    x = torch.randn(1, num_heads, n, head_dim)
    positions = torch.randn(1, n, 3)
    freqs = init_3d_mixed_rope_freqs(head_dim, num_heads).clone().requires_grad_(True)
    out = apply_3d_mixed_rope(x, positions, freqs)
    out.sum().backward()
    assert freqs.grad is not None
    assert torch.isfinite(freqs.grad).all()
    assert freqs.grad.abs().sum() > 0


def test_apply_3d_mixed_rope_rejects_freqs_shape_mismatch() -> None:
    """Mismatched num_heads in freqs should raise."""
    head_dim, num_heads, n = 8, 2, 4
    x = torch.randn(1, num_heads, n, head_dim)
    positions = torch.zeros(1, n, 3)
    freqs = init_3d_mixed_rope_freqs(head_dim, num_heads=3)
    with pytest.raises(ValueError):
        apply_3d_mixed_rope(x, positions, freqs)


def test_apply_3d_axial_rope_temporal_base_changes_only_temporal_slice() -> None:
    """A different temporal_base should rotate only the temporal chunk."""
    x = torch.randn(1, 2, 4, 16)
    positions = torch.tensor([[[1.0, 0.0, 0.0]] * 4])  # all (t=1, 0, 0)

    out_default = apply_3d_axial_rope(
        x, positions, base=10000.0, temporal_dim_frac=0.25
    )
    out_separate = apply_3d_axial_rope(
        x,
        positions,
        base=10000.0,
        temporal_dim_frac=0.25,
        temporal_base=1000.0,
    )

    d_t, _, _ = axial_3d_dim_split(16, 0.25)
    # Spatial slice must match (positions are zero on row/col, base unchanged).
    assert torch.allclose(out_default[..., d_t:], out_separate[..., d_t:])
    # Temporal slice must differ (different base => different rotation angles).
    assert not torch.allclose(out_default[..., :d_t], out_separate[..., :d_t])
    # Both still preserve norms.
    assert torch.allclose(out_separate.norm(dim=-1), x.norm(dim=-1), atol=1e-5)


# ----- timestamps_to_days -----


def test_timestamps_to_days_known_values() -> None:
    """Sanity check days computation for hand-picked dates."""
    # Helios convention: (day, month-1, year). All anchored at 2000-01-01.
    timestamps = torch.tensor(
        [
            [1, 0, 2000],  # 2000-01-01 -> day 0
            [1, 1, 2000],  # 2000-02-01 -> day 31
            [1, 2, 2000],  # 2000-03-01 -> day 59 (non-leap offset)
            [15, 7, 2023],  # 2023-08-15
        ],
        dtype=torch.long,
    )
    days = timestamps_to_days(timestamps)
    assert days.shape == (4,)
    assert days[0].item() == pytest.approx(0.0)
    assert days[1].item() == pytest.approx(31.0)
    assert days[2].item() == pytest.approx(59.0)
    expected_2023 = 23 * 365.25 + 212 + 14
    assert days[3].item() == pytest.approx(expected_2023, abs=1e-3)


def test_timestamps_to_days_relative_deltas_match_calendar() -> None:
    """Within a sample, deltas should approximate true calendar gaps."""
    timestamps = torch.tensor(
        [
            [[15, 0, 2023], [15, 1, 2023], [15, 2, 2023]],  # Jan/Feb/Mar 15
        ],
        dtype=torch.long,
    )
    days = timestamps_to_days(timestamps)  # (1, 3)
    deltas = days[0, 1:] - days[0, :-1]
    # Gaps between (Jan15->Feb15) and (Feb15->Mar15) should be ~31 and ~28.
    assert deltas[0].item() == pytest.approx(31.0, abs=1.0)
    assert deltas[1].item() == pytest.approx(28.0, abs=1.0)


def test_timestamps_to_days_rejects_bad_last_dim() -> None:
    """Non-3 last dim should raise."""
    with pytest.raises(ValueError):
        timestamps_to_days(torch.zeros(2, 2, dtype=torch.long))
