"""A collection of functions for creating position encodings for the OlmoEarth Pretrain model.

These functions are based on the following repository:
https://github.com/bair-climate-initiative/scale-mae/blob/main/mae/util/pos_embed.py

They cover the following:
- 2D sinusoidal position encoding (for spatial data)
- 1D sinusoidal position encoding (for temporal data)
- Month encoding (for temporal data)
- Static multi-frequency temporal encoding
- Static multi-frequency global lat/lon encoding (sphere-mapped)
- Static multi-frequency local 2D position encoding (resolution-aware)
"""

import math
from enum import StrEnum

import numpy as np
import torch


class TimestampEncodingMode(StrEnum):
    """Mode for encoding temporal information."""

    LEGACY = "legacy"
    STATIC_TEMPORAL = "static_temporal"


class SpatialEncodingMode(StrEnum):
    """Mode for encoding spatial information.

    LEGACY: 2D sinusoidal positional encoding scaled by the GSD ratio (the only
        spatial signal; no global location).
    STATIC_SPLIT: Two orthogonal signals filling separate slots:
        local 2D physical-position-within-patch (slot [2n:3n]) and global
        sphere-mapped lat/lon (slot [3n:4n]). Both fully static (no MLP).
    """

    LEGACY = "legacy"
    STATIC_SPLIT = "static_split"


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


def get_static_temporal_encoding(
    timestamps: torch.Tensor, encoding_dim: int
) -> torch.Tensor:
    """Static multi-frequency sinusoidal temporal encoding.

    Converts timestamps to a fractional year and applies geometric-spaced
    sinusoidal frequencies ranging from ~128-year periods to daily resolution.
    The 1-cycle/year frequency naturally produces identical values for the
    same day-of-year across different years.

    Args:
        timestamps: Tensor of shape (B, T, 3) where [..., 0] is day (1-31),
            [..., 1] is month (0-indexed, 0-11), [..., 2] is year.
        encoding_dim: Output encoding dimension (must be even).

    Returns:
        Tensor of shape (B, T, encoding_dim).
    """
    assert encoding_dim % 2 == 0, f"encoding_dim must be even, got {encoding_dim}"
    day = timestamps[..., 0].float()
    month = timestamps[..., 1].float()
    year = timestamps[..., 2].float()

    day_of_year = month * 30.4375 + day
    frac_year = year + day_of_year / 365.25 - 2020.0

    num_freqs = encoding_dim // 2
    exponents = torch.linspace(-7.0, 8.5, num_freqs, device=timestamps.device)
    freqs = 2.0 * math.pi * (2.0**exponents)  # (num_freqs,)

    angles = frac_year.unsqueeze(-1) * freqs  # (B, T, num_freqs)
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


def get_static_global_latlon_encoding(
    latlon: torch.Tensor, encoding_dim: int
) -> torch.Tensor:
    """Static multi-frequency sinusoidal encoding of global tile-center lat/lon.

    Maps lat/lon to a point on the unit sphere (x, y, z) so longitude wrap-
    around is handled naturally and there is no discontinuity at the poles or
    the antimeridian. Then applies geometric-spaced sinusoidal frequencies to
    each of x, y, z, log-spaced from full-sphere periods down to a minimum
    period chosen for numerical safety in float32 callers.

    The trig is computed in float64 internally (then cast back to the input
    dtype) so that lon=180 == lon=-180 and pole-invariance hold to machine
    precision rather than blowing up at the highest frequencies under float32.

    Output is split equally across (x, y, z) and (sin, cos), so encoding_dim
    must be divisible by 6.

    Args:
        latlon: Tensor of shape (B, 2) where [:, 0] is latitude in degrees
            [-90, 90] and [:, 1] is longitude in degrees [-180, 180].
        encoding_dim: Output encoding dimension (must be divisible by 6).

    Returns:
        Tensor of shape (B, encoding_dim).
    """
    assert encoding_dim % 6 == 0, (
        f"encoding_dim must be divisible by 6 (split across x/y/z and sin/cos), "
        f"got {encoding_dim}"
    )
    num_freqs = encoding_dim // 6
    in_dtype = latlon.dtype
    work = torch.float64

    lat_rad = latlon[..., 0].to(work) * (math.pi / 180.0)
    lon_rad = latlon[..., 1].to(work) * (math.pi / 180.0)

    # Map to unit sphere.
    cos_lat = torch.cos(lat_rad)
    x = cos_lat * torch.cos(lon_rad)
    y = cos_lat * torch.sin(lon_rad)
    z = torch.sin(lat_rad)
    xyz = torch.stack([x, y, z], dim=-1)  # (B, 3) in float64

    # Frequency band: lowest is one cycle per full sphere axis (period 2),
    # highest is capped at ~city scale on the sphere -- a period of ~3e-3
    # along the unit axis is ~20km on Earth, well below tile scale, while
    # staying numerically safe for float32 downstream consumers.
    # Use a linspace so we always span the same band regardless of num_freqs.
    exp_low = 0.0  # 2^0 * pi -> period 2 along axis (full sphere).
    exp_high = 9.0  # 2^9 * pi ~= 1608 -> period ~3.9e-3 along axis (~25km).
    if num_freqs == 1:
        exponents = torch.tensor([exp_low], device=latlon.device, dtype=work)
    else:
        exponents = torch.linspace(
            exp_low, exp_high, num_freqs, device=latlon.device, dtype=work
        )
    freqs = math.pi * (2.0**exponents)  # (L,)

    angles = xyz.unsqueeze(-1) * freqs  # (B, 3, L)
    sin_part = torch.sin(angles)
    cos_part = torch.cos(angles)
    flat_sin = sin_part.transpose(-1, -2).reshape(*xyz.shape[:-1], 3 * num_freqs)
    flat_cos = cos_part.transpose(-1, -2).reshape(*xyz.shape[:-1], 3 * num_freqs)
    out = torch.cat([flat_sin, flat_cos], dim=-1)  # (B, 6L) = (B, encoding_dim)
    return out.to(dtype=in_dtype)


def get_static_local_2d_encoding(
    grid_h: int,
    grid_w: int,
    meters_per_token: float,
    encoding_dim: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    min_period_m: float = 20.0,
    max_period_m: float = 1_000.0,
) -> torch.Tensor:
    """Static multi-frequency 2D sinusoidal encoding of position within an image patch.

    Encodes each token's (h, w) grid index as physical meters from the patch
    center, then applies geometric-spaced sinusoidal bands at fixed *absolute*
    physical periods (default 1m up to 100km). Because the frequencies are
    fixed in absolute units rather than tied to the patch extent, the same
    grid index under a different ``meters_per_token`` produces a different
    encoding -- the desired "resolution-aware" property: a 10m-per-token tile
    at grid (3, 5) is at a distinct physical position from a 100m-per-token
    tile at grid (3, 5), and so encodes differently.

    Output is split equally across (h, w) and (sin, cos), so encoding_dim
    must be divisible by 4.

    Args:
        grid_h: Number of tokens in the height dimension of the patch.
        grid_w: Number of tokens in the width dimension.
        meters_per_token: Physical extent of one token in meters
            (input_res * patch_size).
        encoding_dim: Output dim (must be divisible by 4).
        device: Output device.
        dtype: Output dtype.
        min_period_m: Smallest period in the freq band, in meters. Default 20m
            (Nyquist at the smallest meters_per_token we use, BASE_GSD=10m with
            patch_size=1; finer periods alias and contribute no per-token
            discrimination).
        max_period_m: Largest period, in meters. Default 1km (roughly 2x our
            largest patch extent, so the lowest-frequency channel acts as a
            slow-varying ramp across the patch without wasting channels on
            effectively-constant periods).

    Returns:
        Tensor of shape (grid_h, grid_w, encoding_dim). Same for every batch
        element -- broadcast over batch in the caller.
    """
    assert encoding_dim % 4 == 0, (
        f"encoding_dim must be divisible by 4 (split across h/w and sin/cos), "
        f"got {encoding_dim}"
    )
    assert min_period_m > 0 and max_period_m > min_period_m, (
        f"need 0 < min_period_m < max_period_m, got "
        f"min={min_period_m}, max={max_period_m}"
    )
    num_freqs = encoding_dim // 4

    # Physical meters from the patch center.
    h_meters = (
        torch.arange(grid_h, device=device, dtype=dtype) - (grid_h - 1) / 2.0
    ) * meters_per_token
    w_meters = (
        torch.arange(grid_w, device=device, dtype=dtype) - (grid_w - 1) / 2.0
    ) * meters_per_token

    # Fixed-absolute-period freq band: log-spaced from max_period (low freq)
    # to min_period (high freq). Same band regardless of patch geometry, which
    # is what makes the encoding resolution-aware.
    if num_freqs == 1:
        periods_m = torch.tensor([min_period_m], device=device, dtype=dtype)
    else:
        log_top = math.log2(max_period_m)
        log_bot = math.log2(min_period_m)
        # linspace from low freq (large period) to high freq (small period)
        log_periods = torch.linspace(
            log_top, log_bot, num_freqs, device=device, dtype=dtype
        )
        periods_m = 2.0**log_periods
    freqs = 2.0 * math.pi / periods_m  # (L,)

    angles_h = h_meters.unsqueeze(-1) * freqs  # (H, L)
    angles_w = w_meters.unsqueeze(-1) * freqs  # (W, L)

    sin_h = torch.sin(angles_h)  # (H, L)
    cos_h = torch.cos(angles_h)  # (H, L)
    sin_w = torch.sin(angles_w)  # (W, L)
    cos_w = torch.cos(angles_w)  # (W, L)

    # Broadcast (H, L) and (W, L) across the orthogonal axis.
    sin_h_grid = sin_h.unsqueeze(1).expand(grid_h, grid_w, num_freqs)  # (H, W, L)
    cos_h_grid = cos_h.unsqueeze(1).expand(grid_h, grid_w, num_freqs)
    sin_w_grid = sin_w.unsqueeze(0).expand(grid_h, grid_w, num_freqs)
    cos_w_grid = cos_w.unsqueeze(0).expand(grid_h, grid_w, num_freqs)

    # Layout: [sin_h_freqs, cos_h_freqs, sin_w_freqs, cos_w_freqs] -> 4L = encoding_dim.
    return torch.cat(
        [sin_h_grid, cos_h_grid, sin_w_grid, cos_w_grid], dim=-1
    )  # (H, W, 4L)
