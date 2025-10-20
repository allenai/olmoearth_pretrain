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
    grid_size: int,
    res: torch.Tensor,
    encoding_dim: int,
    device: torch.device,
    cls_token: bool = False,
) -> torch.Tensor:
    """Get 2D sin cos position encoding for a given grid of positions with resolution.

    Args:
        grid_size: int of the grid height and width
        res: array of size n, representing the resolution of a pixel (say, in meters),
                where n is the number of spatial dimensions
        encoding_dim: output dimension for each position
        cls_token: whether to add a cls token to the encoding
        device: device to run the encoding on
    Returns:
        encoding: position encoding for the given grid: size (H*W, D)
    """
    # TODO: What happens when the res array is bigger than 1?
    grid_h = torch.arange(grid_size, device=device)
    grid_w = torch.arange(grid_size, device=device)
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
