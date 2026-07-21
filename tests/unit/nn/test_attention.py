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
