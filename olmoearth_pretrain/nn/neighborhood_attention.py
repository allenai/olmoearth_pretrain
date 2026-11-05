"""Training-time neighborhood attention (NA) on the patch grid.

Every query attends only to keys whose patch cell lies in the ``W x W`` box of cells
around the query's own cell, across all timesteps, band sets and modalities of those
cells. ``W`` is odd, so the box reaches ``(W - 1) / 2`` cells to each side; at the
domain edge it is shifted inward (not shrunk), so every query keeps a full ``W x W``
box whenever the grid is at least ``W`` cells wide. A grid of at most ``W`` cells is
therefore exactly full attention.

This is the Searchlight rule (``nn/searchlight.py``) applied during training, where MIM
masking leaves a different number of visible tokens in every cell (so NATTEN cannot be
used). It covers the same attentions Searchlight covers -- the ViT self-attention, the
AttentionPooler read and the location-query self-attention -- plus the decoder's
cross-attention from masked tokens to the location tokens.

Each token carries its ``(row, col)`` cell as a ``[B, N, 2]`` tensor built alongside the
RoPE positions (and reordered with them). A cell of ``-1`` marks a non-spatial token,
which attends and is attended globally.

The mask is a dense ``[B, 1, Nq, Nk]`` boolean for SDPA (fused memory-efficient
kernel) unless it would exceed ``DENSE_MASK_MAX_ELEMENTS``; then, on CUDA, it is a
FlexAttention ``BlockMask``. Measured on an H100 (``agent/favyen/2026-11-05-favyen-
neighborhood-attention-pretrain/bin/attn_bench.py``): dense SDPA costs ~1.2x the stock
key-padding SDPA in fwd+bwd, while FlexAttention costs 3-5x at the release shapes
(N ~ 0.9k-3k) and only wins at very long sequences (N ~ 9k-21k, where the dense mask
would also take several GB).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

# Half-width used for global (non-spatial) queries: wider than any grid.
_GLOBAL_REACH = 1 << 20

# Largest dense mask (B * Nq * Nk booleans) before switching to FlexAttention on CUDA.
DENSE_MASK_MAX_ELEMENTS = 1 << 31


def box_start(cell: Tensor, n: int, neighborhood_cells: int) -> Tensor:
    """First cell of the ``neighborhood_cells`` box around ``cell`` on an axis of ``n`` cells.

    Shifted inward at the edges; ``0`` when the axis is no wider than the box.
    """
    return (cell - neighborhood_cells // 2).clamp(
        min=0, max=max(n - neighborhood_cells, 0)
    )


def query_box_bounds(
    q_cells: Tensor, grid: tuple[int, int], neighborhood_cells: int
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Per-query ``[r0, r1) x [c0, c1)`` box bounds (int32); global queries see all."""
    q_cells = q_cells.long()
    r0 = box_start(q_cells[..., 0], grid[0], neighborhood_cells)
    c0 = box_start(q_cells[..., 1], grid[1], neighborhood_cells)
    is_global = q_cells[..., 0] < 0
    r0 = torch.where(is_global, -_GLOBAL_REACH, r0)
    c0 = torch.where(is_global, -_GLOBAL_REACH, c0)
    span = torch.where(is_global, 2 * _GLOBAL_REACH, neighborhood_cells)
    return r0.int(), (r0 + span).int(), c0.int(), (c0 + span).int()


def box_covers_grid(grid: tuple[int, int], neighborhood_cells: int) -> bool:
    """True when every box is the whole grid, i.e. NA is exactly full attention.

    Callers then keep their stock (full-attention) masks and kernels.
    """
    return grid[0] <= neighborhood_cells and grid[1] <= neighborhood_cells


@dataclass
class NeighborhoodMask:
    """What an attention block needs: exactly one of ``attn_mask`` / ``block_mask``."""

    attn_mask: Tensor | None = None
    block_mask: Any | None = None

    def kwargs(self) -> dict[str, Any]:
        """Keyword arguments for ``Block.forward``."""
        return {"attn_mask": self.attn_mask, "block_mask": self.block_mask}


def dense_neighborhood_mask(
    q_cells: Tensor,
    k_cells: Tensor,
    key_valid: Tensor | None,
    grid: tuple[int, int],
    neighborhood_cells: int,
) -> Tensor:
    """Dense ``[B, 1, Nq, Nk]`` boolean mask (True = attend).

    A query with no allowed key (padding, or a box with no visible token) falls back to
    every valid key so SDPA does not return NaN; the FlexAttention path returns zeros for
    such rows instead. Only padding rows (whose outputs are discarded) hit this in
    practice.
    """
    r0, r1, c0, c1 = query_box_bounds(q_cells, grid, neighborhood_cells)
    kr = k_cells[..., 0].int()[:, None, :]
    kc = k_cells[..., 1].int()[:, None, :]
    k_global = (k_cells[..., 0] < 0)[:, None, :]
    allowed = k_global | (
        (kr >= r0[..., None])
        & (kr < r1[..., None])
        & (kc >= c0[..., None])
        & (kc < c1[..., None])
    )
    if key_valid is not None:
        valid = key_valid.bool()[:, None, :]
        allowed = allowed & valid
        empty = ~allowed.any(dim=-1, keepdim=True)
        allowed = torch.where(empty, valid, allowed)
    return allowed[:, None]


_COMPILED_CREATE_BLOCK_MASK: Callable[..., Any] | None = None


def _create_block_mask(
    mask_mod: Callable[..., Tensor],
    batch: int,
    q_len: int,
    kv_len: int,
    device: torch.device,
) -> Any:
    """``create_block_mask`` compiled once per process, for rectangular masks.

    Compiled so Inductor fuses the predicate into the block reduction instead of
    materializing the dense ``[B, Nq, Nk]`` grid (see ``joint_latent.create_block_mask_cuda``).
    """
    global _COMPILED_CREATE_BLOCK_MASK
    if _COMPILED_CREATE_BLOCK_MASK is None:
        from torch.nn.attention.flex_attention import create_block_mask

        from olmoearth_pretrain.nn.joint_latent import _raise_recompile_limit

        _COMPILED_CREATE_BLOCK_MASK = torch.compile(create_block_mask, dynamic=True)
        _raise_recompile_limit()
    return _COMPILED_CREATE_BLOCK_MASK(
        mask_mod, batch, None, q_len, kv_len, device=device
    )


def flex_neighborhood_mask(
    q_cells: Tensor,
    k_cells: Tensor,
    key_valid: Tensor | None,
    grid: tuple[int, int],
    neighborhood_cells: int,
) -> Any:
    """FlexAttention ``BlockMask`` for the box rule (CUDA)."""
    r0, r1, c0, c1 = query_box_bounds(q_cells, grid, neighborhood_cells)
    kr = k_cells[..., 0].int()
    kc = k_cells[..., 1].int()
    k_global = k_cells[..., 0] < 0
    if key_valid is None:
        key_valid = torch.ones_like(k_global)
    key_valid = key_valid.bool()

    def mask_mod(b: Tensor, h: Tensor, q_idx: Tensor, kv_idx: Tensor) -> Tensor:
        r, c = kr[b, kv_idx], kc[b, kv_idx]
        inside = (
            (r >= r0[b, q_idx])
            & (r < r1[b, q_idx])
            & (c >= c0[b, q_idx])
            & (c < c1[b, q_idx])
        )
        return key_valid[b, kv_idx] & (inside | k_global[b, kv_idx])

    return _create_block_mask(
        mask_mod, q_cells.shape[0], q_cells.shape[1], k_cells.shape[1], q_cells.device
    )


def neighborhood_mask(
    q_cells: Tensor,
    k_cells: Tensor,
    key_valid: Tensor | None,
    grid: tuple[int, int],
    neighborhood_cells: int,
) -> NeighborhoodMask:
    """The box-rule mask for queries at ``q_cells`` over keys at ``k_cells``.

    Args:
        q_cells: ``[B, Nq, 2]`` query ``(row, col)`` cells (-1 = non-spatial).
        k_cells: ``[B, Nk, 2]`` key cells.
        key_valid: ``[B, Nk]`` bool, False for padding / masked keys (None = all valid).
        grid: ``(h, w)`` patch grid of the batch.
        neighborhood_cells: Odd box side ``W`` in cells.
    """
    if neighborhood_cells < 1 or neighborhood_cells % 2 == 0:
        raise ValueError(
            f"neighborhood_cells must be a positive odd number, got {neighborhood_cells}"
        )
    dense_elements = q_cells.shape[0] * q_cells.shape[1] * k_cells.shape[1]
    if q_cells.is_cuda and dense_elements > DENSE_MASK_MAX_ELEMENTS:
        return NeighborhoodMask(
            block_mask=flex_neighborhood_mask(
                q_cells, k_cells, key_valid, grid, neighborhood_cells
            )
        )
    return NeighborhoodMask(
        attn_mask=dense_neighborhood_mask(
            q_cells, k_cells, key_valid, grid, neighborhood_cells
        )
    )


def location_query_cells(
    batch_size: int,
    query_grid: tuple[int, int],
    queries_per_cell: int,
    device: torch.device,
) -> Tensor:
    """``[B, Hq * Wq, 2]`` patch cell of every location query (row-major grid).

    The query grid has ``queries_per_cell = patch_size / location_query_patch_size``
    queries per cell along each side, so query ``(i, j)`` sits in cell
    ``(i // queries_per_cell, j // queries_per_cell)``.
    """
    rows = torch.arange(query_grid[0], device=device) // queries_per_cell
    cols = torch.arange(query_grid[1], device=device) // queries_per_cell
    cells = torch.stack(torch.meshgrid(rows, cols, indexing="ij"), dim=-1)
    return cells.reshape(1, -1, 2).expand(batch_size, -1, -1)
