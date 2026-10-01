"""The mask-free inference attention equals the masked joint attention."""

from typing import Any

import pytest
import torch
import torch.nn.functional as F

from olmoearth_pretrain.nn.dense_joint_attention import (
    build_dense_joint_layout,
    dense_joint_attention,
)
from olmoearth_pretrain.nn.joint_latent import (
    JointLatentConfig,
    JointLatentTransformer,
    joint_attention_allowed,
)


def _joint_module(**overrides: Any) -> JointLatentTransformer:
    kwargs: dict = dict(
        embedding_size=32,
        num_heads=4,
        mlp_ratio=2.0,
        joint_depth=2,
        latent_only_depth=1,
        token_mlp=True,
        position_encoding="rope_3d_mixed",
        rope_base=10000.0,
        rope_mixed_base=10.0,
        temporal_rope_dim_frac=0.25,
        rope_temporal_base=None,
        qk_norm=True,
        latent_reads_all=True,
    )
    kwargs.update(overrides)
    return JointLatentTransformer(**kwargs).eval()


def _inputs(
    B: int, n_h: int, n_w: int, T: int, n_mod: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encoder-order tokens with padding, a missing cell-timestep and non-spatial tokens."""
    cells_one = torch.arange(n_h * n_w).repeat_interleave(T)
    cells = cells_one.repeat(n_mod)
    cells = torch.cat([cells, torch.tensor([-1, -1])]).expand(B, -1).clone()
    n_tokens = cells.shape[1]
    t = torch.arange(n_tokens).float() % T
    spatial = cells.clamp(min=0)
    positions = torch.stack(
        [t.expand(B, -1), (spatial // n_w).float(), (spatial % n_w).float()], dim=-1
    )
    tokens = torch.randn(B, n_tokens, 32)
    valid = torch.ones(B, n_tokens, dtype=torch.bool)
    valid[0, 5:9] = False  # padding inside the first modality's run
    valid[1, :T] = False  # sample 1: cell 0 of the first modality entirely missing
    return tokens, positions, valid, cells


@pytest.mark.parametrize("sort_by_cell", [True, False])
@pytest.mark.parametrize("patch_size", [1, 2])
def test_dense_inference_matches_the_masked_registers(
    sort_by_cell: bool, patch_size: int
) -> None:
    """Registers agree with the masked path (CPU reference kernels)."""
    torch.manual_seed(0)
    n_h = n_w = 3
    tokens, positions, valid, cells = _inputs(B=2, n_h=n_h, n_w=n_w, T=4, n_mod=2)
    kw: dict[str, Any] = dict(sort_by_cell=sort_by_cell, pixel_latents=True)
    masked = _joint_module(**kw)
    dense = _joint_module(dense_inference_attention=True, **kw)
    dense.load_state_dict(masked.state_dict())
    fw = dict(patch_size=patch_size, patch_spacing=1.0)
    with torch.no_grad():
        ref, ref_pos = masked(tokens, positions, valid, cells, (n_h, n_w), **fw)
        out, out_pos = dense(tokens, positions, valid, cells, (n_h, n_w), **fw)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out_pos, ref_pos)


def test_dense_attention_equals_masked_sdpa_on_valid_rows() -> None:
    """The function itself: every valid query row matches SDPA under the joint mask."""
    torch.manual_seed(0)
    B, H, D, n_latents = 2, 3, 8, 5
    cells = torch.tensor([[0, 0, 1, 1, 1, 2, -1, 2, 0], [2, 2, 2, 0, 1, 1, 0, -1, -1]])
    valid = torch.ones_like(cells, dtype=torch.bool)
    valid[0, 3] = False
    valid[1, 0] = False
    n_tokens = cells.shape[1]
    lat_cells = torch.tensor([0, 1, 2, 0, 1]).expand(B, -1)
    cell_id = torch.cat([cells, lat_cells], 1)
    is_latent = torch.cat(
        [
            torch.zeros(B, n_tokens, dtype=torch.bool),
            torch.ones(B, n_latents, dtype=torch.bool),
        ],
        1,
    )
    key_valid = torch.cat([valid, torch.ones(B, n_latents, dtype=torch.bool)], 1)
    allowed = joint_attention_allowed(cell_id, is_latent, key_valid, True)
    q, k, v = (torch.randn(B, H, n_tokens + n_latents, D) for _ in range(3))
    ref = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed[:, None])
    layout = build_dense_joint_layout(cells, valid, n_cells=3, n_latents=n_latents)
    out = dense_joint_attention(q, k, v, layout, use_flash=False)
    rows = key_valid  # padding query rows are never read
    torch.testing.assert_close(
        out.transpose(1, 2)[rows], ref.transpose(1, 2)[rows], atol=1e-5, rtol=1e-5
    )


def test_layout_segments_are_the_valid_tokens_of_each_cell() -> None:
    """One contiguous segment per (sample, cell), padding excluded, -1 its own cell."""
    cells = torch.tensor([[1, 0, 1, -1, 0], [0, 0, 1, 1, -1]])
    valid = torch.tensor([[1, 1, 1, 1, 0], [1, 0, 1, 1, 1]], dtype=torch.bool)
    layout = build_dense_joint_layout(cells, valid, n_cells=2, n_latents=1)
    flat = cells.reshape(-1)[layout.token_index]
    sample = layout.token_index // cells.shape[1]
    bounds = layout.cu_seqlens.tolist()
    segments = [
        (int(sample[a]), sorted(set(flat[a:b].tolist())), b - a)
        for a, b in zip(bounds[:-1], bounds[1:])
    ]
    assert segments == [
        (0, [-1], 1),
        (0, [0], 1),
        (0, [1], 2),
        (1, [-1], 1),
        (1, [0], 1),
        (1, [1], 2),
    ]
    assert layout.max_seqlen == 2 and not layout.all_valid


def test_dense_path_is_inference_only_and_needs_global_latent_reads() -> None:
    """Grad-enabled forwards, cell-local latent reads and local radii keep the mask."""
    dense = _joint_module(dense_inference_attention=True)
    with torch.no_grad():
        assert dense._use_dense_attention()
    assert not dense._use_dense_attention()  # grad enabled
    cases: list[dict[str, Any]] = [
        dict(latent_reads_all=False),
        dict(local_radius=1),
        dict(latent_radius=1),
    ]
    for overrides in cases:
        module = _joint_module(dense_inference_attention=True, **overrides)
        with torch.no_grad():
            assert not module._use_dense_attention(), overrides


def test_config_default_keeps_the_masked_path() -> None:
    """Old configs (no field) build the masked path; the flag round-trips."""
    build_kw: dict[str, Any] = dict(
        encoder_embedding_size=32,
        encoder_num_heads=4,
        mlp_ratio=2.0,
        position_encoding="rope_3d_mixed",
        rope_base=10000.0,
        qk_norm=False,
    )
    assert (
        not JointLatentConfig(register_dim=32)
        .build(**build_kw)
        .dense_inference_attention
    )
    on = JointLatentConfig(register_dim=32, dense_inference_attention=True)
    assert on.build(**build_kw).dense_inference_attention
    assert (
        "dense_inference_attention"
        not in JointLatentConfig(register_dim=32).as_config_dict()
    )
