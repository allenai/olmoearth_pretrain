"""A collection of functions for creating position encodings for the OlmoEarth Pretrain model.

These functions are based on the following repository:
https://github.com/bair-climate-initiative/scale-mae/blob/main/mae/util/pos_embed.py

They cover the following:
- 2D sinusoidal position encoding (for spatial data)
- 1D sinusoidal position encoding (for temporal data)
- Month encoding (for temporal data)
- Lat/lon encoding (for geographic location)
- Timestamp encoding (for full datetime information)
"""

from enum import StrEnum

import numpy as np
import torch


class TimestampEncodingMode(StrEnum):
    """Mode for encoding temporal information."""

    LEGACY = "legacy"
    UNIFIED = "unified"


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


def get_latlon_encoding(latlon: torch.Tensor, encoding_dim: int) -> torch.Tensor:
    """Sinusoidal encoding for geographic lat/lon coordinates.

    Normalizes latitude to [-1, 1] (from [-90, 90]) and longitude to [-1, 1]
    (from [-180, 180]), then applies 1D sinusoidal encoding to each coordinate
    and concatenates the results.

    Args:
        latlon: Tensor of shape (B, 2) where [:, 0] is latitude and [:, 1] is longitude,
            both in degrees.
        encoding_dim: Output encoding dimension (must be divisible by 2).

    Returns:
        encoding: Tensor of shape (B, encoding_dim).
    """
    assert encoding_dim % 2 == 0, f"encoding_dim must be even, got {encoding_dim}"
    dim_per_coord = encoding_dim // 2
    lat = latlon[:, 0] / 90.0  # normalize to [-1, 1]
    lon = latlon[:, 1] / 180.0  # normalize to [-1, 1]
    lat_enc = get_1d_sincos_pos_encoding(lat, dim_per_coord)  # (B, D/2)
    lon_enc = get_1d_sincos_pos_encoding(lon, dim_per_coord)  # (B, D/2)
    return torch.cat([lat_enc, lon_enc], dim=1)  # (B, D)


def get_timestamp_encoding(timestamps: torch.Tensor, encoding_dim: int) -> torch.Tensor:
    """Sinusoidal encoding for full timestamps.

    Converts [day, month (0-indexed), year] to a continuous fractional year
    and applies 1D sinusoidal encoding.

    Args:
        timestamps: Tensor of shape (B, T, 3) where [..., 0] is day (1-31),
            [..., 1] is month (0-indexed, 0-11), [..., 2] is year.
        encoding_dim: Output encoding dimension (must be even).

    Returns:
        encoding: Tensor of shape (B, T, encoding_dim).
    """
    assert encoding_dim % 2 == 0, f"encoding_dim must be even, got {encoding_dim}"
    day = timestamps[..., 0].float()
    month = timestamps[..., 1].float()
    year = timestamps[..., 2].float()

    # Convert to fractional year (e.g., 2021.5 for ~July 2021)
    day_of_year = month * 30.4375 + day
    fractional_year = year + day_of_year / 365.25

    b, t = fractional_year.shape
    flat = fractional_year.reshape(-1)  # (B*T,)
    enc = get_1d_sincos_pos_encoding(flat, encoding_dim)  # (B*T, D)
    return enc.reshape(b, t, encoding_dim)
