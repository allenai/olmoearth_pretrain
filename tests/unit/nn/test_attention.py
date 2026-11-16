"""Unit tests for attention components."""

import pytest
import torch

from olmoearth_pretrain.nn.attention import Attention


def _grid_positions(b: int, side: int) -> torch.Tensor:
    rows = torch.arange(side).repeat_interleave(side)
    cols = torch.arange(side).repeat(side)
    grid = torch.stack([rows, cols], dim=-1).float()  # (side*side, 2)
    return grid.unsqueeze(0).expand(b, -1, -1).contiguous()


@pytest.mark.parametrize(
    "position_encoding,num_coords",
    [("rope", 2), ("rope_mixed", 2), ("rope_3d", 3), ("rope_3d_mixed", 3)],
)
def test_compiled_attention_applies_rope(
    position_encoding: str, num_coords: int
) -> None:
    """Compiled attention output must match eager for every RoPE mode.

    Regression test: Dynamo mis-evaluated the ``PositionEncoding.is_rope`` guard in
    Attention.forward when ``self.position_encoding`` is a plain config string
    (str-vs-StrEnum-set membership), so compiled blocks silently skipped RoPE.
    The position_encoding is passed as a plain str, like when built from a config.
    """
    torch._dynamo.reset()
    # head_dim=16 satisfies %4 (2D/mixed modes) and the axial 3D dim split.
    b, h, side, d = 2, 2, 4, 16
    n = side * side
    attn = Attention(
        dim=h * d,
        num_heads=h,
        attn_drop=0.0,
        position_encoding=position_encoding,
    ).eval()

    torch.manual_seed(0)
    x = torch.randn(b, n, h * d)
    positions = _grid_positions(b, side)
    if num_coords == 3:
        # Prepend a nonzero temporal coordinate: (t, row, col).
        t = torch.arange(n).remainder(3).float()
        t = t.unsqueeze(0).expand(b, -1).unsqueeze(-1)
        positions = torch.cat([t, positions], dim=-1)

    with torch.no_grad():
        eager = attn(x, rope_positions=positions)
        compiled = torch.compile(attn, backend="eager", fullgraph=True)(
            x, rope_positions=positions
        )
        # Sanity check that RoPE actually affects the output, so a compiled graph
        # that dropped RoPE could not pass this test vacuously.
        no_rope = Attention(dim=h * d, num_heads=h, attn_drop=0.0).eval()
        no_rope.load_state_dict(attn.state_dict(), strict=False)
        assert not torch.allclose(no_rope(x), eager)

    torch.testing.assert_close(compiled, eager)


def test_routed_mlp_applies_each_tokens_expert() -> None:
    """Each token goes through its own expert, and only that one."""
    from olmoearth_pretrain.nn.attention import RoutedMlp

    torch.manual_seed(0)
    mlp = RoutedMlp(3, in_features=8, hidden_features=16)
    x = torch.randn(2, 5, 8)
    ids = torch.tensor([[0, 1, 2, 1, 0], [2, 2, 0, 1, 1]])
    out = mlp(x, ids)
    assert out.shape == x.shape
    for i, expert in enumerate(mlp.experts):
        assert torch.allclose(out[ids == i], expert(x[ids == i]), atol=1e-6)


def test_routed_mlp_unused_expert_still_gets_a_gradient() -> None:
    """An expert with no tokens stays in the graph (zero gradient, not None)."""
    from olmoearth_pretrain.nn.attention import RoutedMlp

    mlp = RoutedMlp(3, in_features=8, hidden_features=16)
    x = torch.randn(4, 8)
    mlp(x, torch.tensor([0, 0, 1, 1])).sum().backward()
    for name, param in mlp.named_parameters():
        assert param.grad is not None, name
    assert mlp.experts[2].fc1.weight.grad.abs().sum() == 0
    assert mlp.experts[0].fc1.weight.grad.abs().sum() > 0


def test_routed_mlp_rejects_mismatched_ids() -> None:
    """Expert ids must match the token shape."""
    from olmoearth_pretrain.nn.attention import RoutedMlp

    mlp = RoutedMlp(2, in_features=8)
    with pytest.raises(ValueError, match="expert_ids shape"):
        mlp(torch.randn(2, 5, 8), torch.zeros(2, 4, dtype=torch.long))
