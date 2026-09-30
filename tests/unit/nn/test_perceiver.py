"""Unit tests for the spatial Perceiver register bottleneck."""

import weakref
from collections.abc import Callable
from typing import Any

import pytest
import torch
from einops import rearrange
from torch import Tensor, nn

from olmoearth_pretrain.nn.flexi_vit import Perceiver

ENCODER_DIM = 64
DEPTH = 4
SPATIAL_GRID = (3, 5)


def _build_perceiver(per_depth_read_proj: bool, attn_dim: int | None) -> Perceiver:
    torch.manual_seed(0)
    perceiver = Perceiver(
        encoder_embedding_size=ENCODER_DIM,
        # A narrower register dim exercises the kv_proj Linear when attn_dim is None.
        register_dim=32,
        num_heads=4,
        mlp_ratio=2.0,
        latent_transformer_depth=DEPTH,
        use_2d_rope=True,
        per_depth_read_proj=per_depth_read_proj,
        attn_dim=attn_dim,
        student_dims=[16, 8],
        student_output_norm=True,
    )
    # Randomize the LayerNorm affines so each per-depth read produces distinct K/V.
    with torch.no_grad():
        for module in perceiver.modules():
            if isinstance(module, torch.nn.LayerNorm):
                module.weight.normal_()
                module.bias.normal_()
    return perceiver.eval()


def _make_inputs(
    batch_size: int = 2, num_tokens: int = 40
) -> tuple[Tensor, Tensor, Tensor]:
    generator = torch.Generator().manual_seed(1)
    patch_tokens = torch.randn(batch_size, num_tokens, ENCODER_DIM, generator=generator)
    patch_positions = (
        torch.randint(0, 5, (batch_size, num_tokens, 2), generator=generator).float()
        * 10.0
    )
    visible_mask = torch.rand(batch_size, num_tokens, generator=generator) > 0.3
    return patch_tokens, patch_positions, visible_mask


ForwardFn = Callable[
    [Tensor, Tensor, Tensor, tuple[int, int]],
    tuple[Tensor, Tensor | None, Tensor | None],
]


def _reference_forward(
    perceiver: Perceiver,
    patch_tokens: Tensor,
    patch_positions: Tensor,
    visible_mask: Tensor,
    spatial_grid: tuple[int, int],
) -> tuple[Tensor, Tensor | None, Tensor | None]:
    """Frozen copy of ``Perceiver.forward`` from before K/V was built lazily per read.

    Builds every read's K/V up front. Kept as the reference the lazy forward must match.
    """
    if perceiver.per_depth_read_proj:
        kv_per_read = [
            proj(norm(patch_tokens))
            for norm, proj in zip(perceiver.input_norms, perceiver.kv_projs)
        ]
    else:
        kv = perceiver.kv_proj(perceiver.input_norm(patch_tokens))
        kv_per_read = [kv] * len(perceiver.read_blocks)
    batch_size = patch_tokens.shape[0]
    num_registers = spatial_grid[0] * spatial_grid[1]
    registers = (
        perceiver.register.unsqueeze(0)
        .expand(batch_size, num_registers, -1)
        .contiguous()
    )
    register_positions = perceiver.build_register_positions(
        patch_positions, spatial_grid
    )
    read_attn_mask = visible_mask.bool()
    for i, (read_blk, kv) in enumerate(zip(perceiver.read_blocks, kv_per_read)):
        registers = read_blk(
            x=registers,
            y=kv,
            attn_mask=read_attn_mask,
            rope_positions=register_positions,
            rope_positions_y=patch_positions,
        )
        registers = perceiver.latent_blocks[i](
            x=registers,
            rope_positions=register_positions,
        )
    out = perceiver.norm(registers)
    out = rearrange(out, "b (h w) d -> b h w d", h=spatial_grid[0], w=spatial_grid[1])
    student_registers = (
        perceiver.student(out.detach()) if perceiver.student is not None else None
    )
    return out, register_positions, student_registers


_CONFIGS = pytest.mark.parametrize(
    ("per_depth_read_proj", "attn_dim"),
    [(True, ENCODER_DIM), (True, None), (False, ENCODER_DIM), (False, None)],
)


@_CONFIGS
def test_perceiver_forward_matches_reference(
    per_depth_read_proj: bool, attn_dim: int | None
) -> None:
    """The lazy per-read K/V forward is bit-identical to the eager reference."""
    perceiver = _build_perceiver(per_depth_read_proj, attn_dim)
    patch_tokens, patch_positions, visible_mask = _make_inputs()

    with torch.no_grad():
        expected = _reference_forward(
            perceiver, patch_tokens, patch_positions, visible_mask, SPATIAL_GRID
        )
        actual = perceiver(patch_tokens, patch_positions, visible_mask, SPATIAL_GRID)

    assert actual[0].shape == (2, *SPATIAL_GRID, 32)
    assert actual[2] is not None and actual[2].shape == (2, *SPATIAL_GRID, 16)
    for actual_tensor, expected_tensor in zip(actual, expected):
        assert expected_tensor is not None
        assert torch.equal(actual_tensor, expected_tensor)


@_CONFIGS
def test_perceiver_gradients_match_reference(
    per_depth_read_proj: bool, attn_dim: int | None
) -> None:
    """Parameter and input gradients are bit-identical to the eager reference."""
    perceiver = _build_perceiver(per_depth_read_proj, attn_dim)
    patch_tokens, patch_positions, visible_mask = _make_inputs()

    def grads(forward_fn: ForwardFn) -> tuple[Tensor, dict[str, Tensor]]:
        perceiver.zero_grad(set_to_none=True)
        tokens = patch_tokens.clone().requires_grad_(True)
        out, _, student = forward_fn(
            tokens, patch_positions, visible_mask, SPATIAL_GRID
        )
        assert student is not None
        (out.square().sum() + student.sum()).backward()
        assert tokens.grad is not None
        param_grads = {
            name: param.grad.clone()
            for name, param in perceiver.named_parameters()
            if param.grad is not None
        }
        return tokens.grad, param_grads

    expected_token_grad, expected_param_grads = grads(
        lambda *args: _reference_forward(perceiver, *args)
    )
    actual_token_grad, actual_param_grads = grads(perceiver)

    assert torch.equal(actual_token_grad, expected_token_grad)
    assert actual_param_grads.keys() == expected_param_grads.keys()
    for name, expected_grad in expected_param_grads.items():
        assert torch.equal(actual_param_grads[name], expected_grad), name


@pytest.mark.parametrize("attn_dim", [ENCODER_DIM, None])
def test_perceiver_keeps_one_per_read_kv_alive_at_inference(
    attn_dim: int | None,
) -> None:
    """Under no_grad, each read's K/V is built just before that read and freed after.

    Regression test for the v1.3 inference OOM, where all per-read K/V copies were
    built before the first read and held for the whole loop.
    """
    perceiver = _build_perceiver(per_depth_read_proj=True, attn_dim=attn_dim)
    patch_tokens, patch_positions, visible_mask = _make_inputs()

    kv_refs: list[weakref.ref] = []
    # For each read: (K/V tensors built so far, indices still alive, index consumed).
    state_at_read: list[tuple[int, list[int], int | None]] = []

    def record_kv(_module: nn.Module, _inputs: tuple[Any, ...], output: Tensor) -> None:
        kv_refs.append(weakref.ref(output))

    def record_read(
        _module: nn.Module, _args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> None:
        alive = [j for j, ref in enumerate(kv_refs) if ref() is not None]
        consumed = next(
            (j for j, ref in enumerate(kv_refs) if ref() is kwargs["y"]), None
        )
        state_at_read.append((len(kv_refs), alive, consumed))

    handles = [proj.register_forward_hook(record_kv) for proj in perceiver.kv_projs]
    handles += [
        blk.register_forward_pre_hook(record_read, with_kwargs=True)
        for blk in perceiver.read_blocks
    ]
    try:
        with torch.no_grad():
            perceiver(patch_tokens, patch_positions, visible_mask, SPATIAL_GRID)
    finally:
        for handle in handles:
            handle.remove()

    assert state_at_read == [(i + 1, [i], i) for i in range(DEPTH)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_perceiver_inference_peak_memory_cuda() -> None:
    """On CUDA, the lazy forward's inference peak is ~3 token-sized tensors lower."""
    device = torch.device("cuda")
    perceiver = _build_perceiver(per_depth_read_proj=True, attn_dim=ENCODER_DIM).to(
        device
    )
    patch_tokens, patch_positions, visible_mask = (
        t.to(device) for t in _make_inputs(batch_size=4, num_tokens=200_000)
    )
    token_bytes = patch_tokens.numel() * patch_tokens.element_size()

    def peak_bytes(forward_fn: ForwardFn) -> int:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            forward_fn(patch_tokens, patch_positions, visible_mask, SPATIAL_GRID)
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated()

    reference_peak = peak_bytes(lambda *args: _reference_forward(perceiver, *args))
    lazy_peak = peak_bytes(perceiver)

    assert reference_peak - lazy_peak >= 2.5 * token_bytes
