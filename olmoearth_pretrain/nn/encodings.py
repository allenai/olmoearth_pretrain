"""A collection of functions for creating position encodings for the OlmoEarth Pretrain model.

These functions are based on the following repository:
https://github.com/bair-climate-initiative/scale-mae/blob/main/mae/util/pos_embed.py

They cover the following:
- 2D sinusoidal position encoding (for spatial data)
- 1D sinusoidal position encoding (for temporal data)
- Month encoding (for temporal data)
"""

import numpy as np
import torch


def get_1d_sincos_pos_encoding(pos: torch.Tensor, encoding_dim: int) -> torch.Tensor:
    """Get 1D sin cos position encoding for a given set of positions.

    Args:
        pos: a list of positions to be encoded: size (L,) this can be a time or space dimension
        encoding_dim: output dimension for each position
    Returns:
        encoding: position encoding for the given positions: size (L, D)
    """
    assert encoding_dim % 2 == 0, f"encoding_dim must be even, got {encoding_dim}"
    omega = torch.arange(encoding_dim // 2, device=pos.device) / encoding_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (L,)
    out = torch.einsum("l,d->ld", pos, omega)  # (L, D/2), outer product
    encoding_sin = torch.sin(out)  # (L, D/2)
    encoding_cos = torch.cos(out)  # (L, D/2)

    encoding = torch.cat([encoding_sin, encoding_cos], dim=1)  # (L, D)
    return encoding


def get_2d_sincos_pos_encoding(grid: torch.Tensor, encoding_dim: int) -> torch.Tensor:
    """Get 2D sin cos position encoding for a given grid of positions.

    Args:
        grid: a grid of positions to be encoded: size  2 x h x w
        encoding_dim: output dimension for each position
    Returns:
        encoding: position encoding for the given grid: size (h*w, D)
    """
    assert encoding_dim % 2 == 0

    # use half of dimensions to encode grid_h
    encoding_dim_1d = encoding_dim // 2
    emb_h = get_1d_sincos_pos_encoding(grid[0], encoding_dim_1d)  # (h*w, D/2)
    emb_w = get_1d_sincos_pos_encoding(grid[1], encoding_dim_1d)  # (h*w, D/2)

    emb = torch.cat([emb_h, emb_w], dim=1)  # (h*w, D)
    return emb


def get_2d_sincos_pos_encoding_with_resolution(
    grid_size: int | tuple[int, int],
    res: torch.Tensor,
    encoding_dim: int,
    device: torch.device,
    cls_token: bool = False,
) -> torch.Tensor:
    """Get 2D sin cos position encoding for a given grid of positions with resolution.

    Args:
        grid_size: Grid size. If an int, uses a square grid (H=W=grid_size). If a
            tuple, interpreted as (H, W).
        res: array of size n, representing the resolution of a pixel (say, in meters),
                where n is the number of spatial dimensions
        encoding_dim: output dimension for each position
        cls_token: whether to add a cls token to the encoding
        device: device to run the encoding on
    Returns:
        encoding: position encoding for the given grid: size (H*W, D)
    """
    # TODO: What happens when the res array is bigger than 1?
    if isinstance(grid_size, tuple):
        grid_h_size, grid_w_size = grid_size
    else:
        grid_h_size = grid_w_size = grid_size

    grid_h = torch.arange(grid_h_size, device=device)
    grid_w = torch.arange(grid_w_size, device=device)
    grid = torch.meshgrid(grid_w, grid_h, indexing="xy")  # (h_grid, w_grid)
    grid = torch.stack(grid, dim=0)  # 2 x h x w

    # create resolution scaled grid
    grid = torch.einsum("chw,n->cnhw", grid, res)  # 2 x n x h x w
    _, n, h, w = grid.shape
    pos_embed = get_2d_sincos_pos_encoding(grid, encoding_dim)  # (nxH*W, D/2)
    pos_embed = pos_embed.reshape(n, h * w, encoding_dim)
    if cls_token:
        pos_embed = torch.cat(
            [
                torch.zeros([n, 1, encoding_dim], device=pos_embed.device),
                pos_embed,
            ],
            dim=1,
        )
    return pos_embed


def get_month_encoding_table(encoding_dim: int) -> torch.Tensor:
    """Sinusoid month encoding table, for 12 months indexed from 0-11.

    Args:
        encoding_dim: output dimension for each position
    Returns:
        month_table: position encoding for the given grid: size (M, D)
    """
    assert encoding_dim % 2 == 0
    angles = torch.arange(0, 13) / (12 / (2 * np.pi))

    dim_per_table = encoding_dim // 2
    sin_table = torch.sin(torch.stack([angles for _ in range(dim_per_table)], axis=-1))
    cos_table = torch.cos(torch.stack([angles for _ in range(dim_per_table)], axis=-1))
    month_table = torch.concatenate([sin_table[:-1], cos_table[:-1]], axis=-1)

    return month_table  # (M, D)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate adjacent feature pairs for rotary position embeddings."""
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)


def apply_1d_rope(
    x: torch.Tensor, positions: torch.Tensor, base: float
) -> torch.Tensor:
    """Apply 1D RoPE to the last dimension of ``x``."""
    if x.shape[-1] % 2 != 0:
        raise ValueError(f"RoPE dimension must be even, got {x.shape[-1]}")

    dtype = x.dtype
    inv_freq = 1.0 / (
        base
        ** (
            torch.arange(0, x.shape[-1], 2, device=x.device, dtype=torch.float32)
            / x.shape[-1]
        )
    )
    angles = positions.to(device=x.device, dtype=torch.float32).unsqueeze(-1) * inv_freq
    cos = torch.repeat_interleave(torch.cos(angles), repeats=2, dim=-1).to(dtype=dtype)
    sin = torch.repeat_interleave(torch.sin(angles), repeats=2, dim=-1).to(dtype=dtype)
    return (x * cos) + (rotate_half(x) * sin)


def apply_2d_rope(
    x: torch.Tensor,
    positions: torch.Tensor,
    base: float = 10000.0,
) -> torch.Tensor:
    """Apply axial 2D RoPE to attention query/key tensors.

    Args:
        x: Attention tensor with shape ``(B, H, N, D)`` or packed shape
            ``(N, H, D)``.
        positions: Spatial coordinates with shape ``(B, N, 2)`` or packed shape
            ``(N, 2)``. The last coordinate dimension is ``(row, col)``.
        base: RoPE frequency base.
    """
    if x.shape[-1] % 4 != 0:
        raise ValueError(
            f"2D RoPE head dimension must be divisible by 4, got {x.shape[-1]}"
        )
    if positions.shape[-1] != 2:
        raise ValueError(
            f"2D RoPE positions must end with size 2, got {positions.shape}"
        )
    if x.ndim not in (3, 4):
        raise ValueError(f"2D RoPE expects a 3D or 4D attention tensor, got {x.shape}")

    half_dim = x.shape[-1] // 2
    x_row, x_col = x[..., :half_dim], x[..., half_dim:]

    if x.ndim == 4:
        if positions.ndim != 3:
            raise ValueError(
                "unpacked 2D RoPE expects positions with shape "
                f"(B, N, 2), got {positions.shape}"
            )
        row_pos = positions[:, None, :, 0]
        col_pos = positions[:, None, :, 1]
    else:
        if positions.ndim != 2:
            raise ValueError(
                "packed 2D RoPE expects positions with shape "
                f"(N, 2), got {positions.shape}"
            )
        row_pos = positions[:, None, 0]
        col_pos = positions[:, None, 1]

    x_row = apply_1d_rope(x_row, row_pos, base)
    x_col = apply_1d_rope(x_col, col_pos, base)
    return torch.cat([x_row, x_col], dim=-1)


def build_window_mask(
    q_positions: torch.Tensor,
    k_positions: torch.Tensor,
    half_extent: float,
    q_is_global: torch.Tensor | None = None,
    k_is_global: torch.Tensor | None = None,
    key_valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """Boolean sliding-window (local) spatial attention mask.

    A query attends a key iff their ``(row, col)`` coordinates differ by at most
    ``half_extent`` on *both* axes (a Chebyshev / square window centred on the
    query), OR either token is flagged global. Global tokens (non-spatial
    modalities, register tokens) sit at the coordinate origin and would otherwise
    be trapped there, so they attend to / are attended by everything. The result
    is AND-ed with ``key_valid`` so padded / masked keys never participate.

    ``True`` means the pair takes part in attention, matching the convention of
    ``torch.nn.functional.scaled_dot_product_attention``.

    Args:
        q_positions: ``(B, Nq, 2)`` GSD-scaled ``(row, col)`` query coordinates.
        k_positions: ``(B, Nk, 2)`` GSD-scaled ``(row, col)`` key coordinates.
        half_extent: Window half-width in the same coordinate units as the
            positions (i.e. ``(window_size / 2) * per_patch_coordinate_step``).
        q_is_global: Optional ``(B, Nq)`` bool, True where a query attends all keys.
        k_is_global: Optional ``(B, Nk)`` bool, True where a key is attended by all.
        key_valid: Optional ``(B, Nk)`` bool, True where a key may participate.

    Returns:
        ``(B, 1, Nq, Nk)`` bool mask (broadcast over attention heads).
    """
    if q_positions.shape[-1] != 2 or k_positions.shape[-1] != 2:
        raise ValueError("window mask positions must end with size 2 (row, col)")
    drow = (q_positions[:, :, None, 0] - k_positions[:, None, :, 0]).abs()
    dcol = (q_positions[:, :, None, 1] - k_positions[:, None, :, 1]).abs()
    # Relative tolerance guards float error on integer-grid * gsd_ratio coordinates.
    thresh = half_extent * (1.0 + 1e-4) + 1e-6
    mask = (drow <= thresh) & (dcol <= thresh)  # (B, Nq, Nk)
    if q_is_global is not None:
        mask = mask | q_is_global[:, :, None]
    if k_is_global is not None:
        mask = mask | k_is_global[:, None, :]
    if key_valid is not None:
        mask = mask & key_valid[:, None, :]
    # Guarantee every query keeps >=1 valid key: a query whose window contains no valid
    # key (heavy masking + a small window) would softmax over an empty row and produce
    # NaNs that poison gradients even where the output is later discarded. Such starved
    # queries fall back to attending all valid keys.
    fallback = (
        key_valid[:, None, :].expand_as(mask)
        if key_valid is not None
        else torch.ones_like(mask)
    )
    empty = ~mask.any(dim=-1, keepdim=True)  # (B, Nq, 1)
    mask = torch.where(empty, fallback, mask)
    return mask[:, None]  # (B, 1, Nq, Nk)
