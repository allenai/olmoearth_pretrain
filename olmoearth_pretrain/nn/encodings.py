"""A collection of functions for creating position encodings for the OlmoEarth Pretrain model.

These functions are based on the following repository:
https://github.com/bair-climate-initiative/scale-mae/blob/main/mae/util/pos_embed.py

They cover the following:
- 2D sinusoidal position encoding (for spatial data)
- 1D sinusoidal position encoding (for temporal data)
- Month encoding (for temporal data)
- Static temporal encoding (multi-frequency sincos of fractional year)
- Static lat/lon encoding (sphere-mapped multi-frequency sincos)
- Axial 2D RoPE and RoPE-Mixed for attention.
"""

import math

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


def init_2d_rope_mixed_freqs(
    head_dim: int,
    num_heads: int,
    base: float = 10.0,
    rotate: bool = True,
) -> torch.Tensor:
    """Initialize learnable 2D frequencies for RoPE-Mixed.

    Follows the per-head random-direction init from
    https://github.com/naver-ai/rope-vit (Heo et al., 2024). Each head receives
    ``head_dim // 2`` complex-pair 2D frequencies. Half of them point along a
    per-head random direction, the other half along the orthogonal direction,
    so each head covers two non-parallel axes in 2D frequency space.

    Args:
        head_dim: Per-head channel dimension. Must be divisible by 4.
        num_heads: Number of attention heads.
        base: Frequency base. The paper uses 10 for RoPE-Mixed in ViT-B.
        rotate: If True, randomize the per-head rotation angle.

    Returns:
        Tensor of shape ``(2, num_heads, head_dim // 2)``. The first axis
        indexes ``(row_freq, col_freq)`` for each complex pair.
    """
    if head_dim % 4 != 0:
        raise ValueError(
            f"RoPE-Mixed init requires head_dim divisible by 4, got {head_dim}"
        )
    mag = 1.0 / (
        base ** (torch.arange(0, head_dim, 4, dtype=torch.float32) / head_dim)
    )  # (head_dim // 4,)
    if rotate:
        angles = torch.rand(num_heads) * 2 * torch.pi
    else:
        angles = torch.zeros(num_heads)
    angles = angles.unsqueeze(-1)  # (num_heads, 1)
    freqs_row = torch.cat(
        [mag * torch.cos(angles), mag * torch.cos(angles + torch.pi / 2)],
        dim=-1,
    )
    freqs_col = torch.cat(
        [mag * torch.sin(angles), mag * torch.sin(angles + torch.pi / 2)],
        dim=-1,
    )
    return torch.stack([freqs_row, freqs_col], dim=0)


def apply_2d_rope_mixed(
    x: torch.Tensor,
    positions: torch.Tensor,
    freqs: torch.Tensor,
) -> torch.Tensor:
    """Apply RoPE-Mixed (learnable 2D frequencies) to attention q/k.

    Each complex feature pair is rotated by an angle of the form
    ``theta_row * row + theta_col * col``, where ``(theta_row, theta_col)`` is
    a learnable per-head, per-pair 2D frequency.

    Args:
        x: Attention tensor with shape ``(B, H, N, D)`` or packed
            ``(N, H, D)``.
        positions: Spatial coordinates with shape ``(B, N, 2)`` or packed
            ``(N, 2)``. Last dim is ``(row, col)``.
        freqs: Learnable 2D frequencies of shape ``(2, H, D // 2)``.
            ``freqs[0]`` is the row component, ``freqs[1]`` is the col
            component.
    """
    head_dim = x.shape[-1]
    if head_dim % 2 != 0:
        raise ValueError(f"RoPE head dimension must be even, got {head_dim}")
    if positions.shape[-1] != 2:
        raise ValueError(
            f"2D RoPE positions must end with size 2, got {positions.shape}"
        )
    if x.ndim not in (3, 4):
        raise ValueError(f"2D RoPE expects a 3D or 4D attention tensor, got {x.shape}")
    if freqs.ndim != 3 or freqs.shape[0] != 2:
        raise ValueError(
            f"RoPE-Mixed freqs must have shape (2, H, D/2), got {freqs.shape}"
        )
    if freqs.shape[-1] * 2 != head_dim:
        raise ValueError(
            f"RoPE-Mixed freqs last dim must equal head_dim // 2, "
            f"got freqs={freqs.shape}, head_dim={head_dim}"
        )

    dtype = x.dtype
    freqs_row = freqs[0].to(device=x.device, dtype=torch.float32)  # (H, D/2)
    freqs_col = freqs[1].to(device=x.device, dtype=torch.float32)  # (H, D/2)
    positions = positions.to(device=x.device, dtype=torch.float32)

    if x.ndim == 4:
        if positions.ndim != 3:
            raise ValueError(
                "unpacked RoPE-Mixed expects positions with shape "
                f"(B, N, 2), got {positions.shape}"
            )
        if freqs.shape[1] != x.shape[1]:
            raise ValueError(
                f"RoPE-Mixed freqs num_heads={freqs.shape[1]} does not match "
                f"attention num_heads={x.shape[1]}"
            )
        row_pos = positions[..., 0]  # (B, N)
        col_pos = positions[..., 1]  # (B, N)
        angles = (
            row_pos[:, None, :, None] * freqs_row[None, :, None, :]
            + col_pos[:, None, :, None] * freqs_col[None, :, None, :]
        )  # (B, H, N, D/2)
    else:
        if positions.ndim != 2:
            raise ValueError(
                "packed RoPE-Mixed expects positions with shape "
                f"(N, 2), got {positions.shape}"
            )
        if freqs.shape[1] != x.shape[1]:
            raise ValueError(
                f"RoPE-Mixed freqs num_heads={freqs.shape[1]} does not match "
                f"attention num_heads={x.shape[1]}"
            )
        row_pos = positions[..., 0]  # (N,)
        col_pos = positions[..., 1]  # (N,)
        angles = (
            row_pos[:, None, None] * freqs_row[None, :, :]
            + col_pos[:, None, None] * freqs_col[None, :, :]
        )  # (N, H, D/2)

    cos = torch.repeat_interleave(torch.cos(angles), repeats=2, dim=-1).to(dtype=dtype)
    sin = torch.repeat_interleave(torch.sin(angles), repeats=2, dim=-1).to(dtype=dtype)
    return (x * cos) + (rotate_half(x) * sin)


def get_static_temporal_encoding(
    timestamps: torch.Tensor, encoding_dim: int
) -> torch.Tensor:
    """Multi-frequency sinusoidal encoding of timestamps as fractional years.

    Computes ``frac_year = year + day_of_year/365.25 - 2020`` and applies
    sin/cos at geometric-spaced frequencies spanning ~128-year periods down to
    sub-daily resolution. The 1-cycle/year frequency naturally produces matching
    values for the same calendar day across years.

    No learnable parameters; deterministic; output dtype matches input.

    Args:
        timestamps: Tensor of shape ``(..., 3)`` where index 0 is day-of-month
            (1-31), index 1 is month (0-indexed, 0-11), index 2 is year.
        encoding_dim: Output dimension. Must be even.

    Returns:
        Tensor of shape ``(..., encoding_dim)``.
    """
    if encoding_dim % 2 != 0:
        raise ValueError(f"encoding_dim must be even, got {encoding_dim}")

    day = timestamps[..., 0].float()
    month = timestamps[..., 1].float()
    year = timestamps[..., 2].float()
    # average month length so this is independent of which year we're in.
    day_of_year = month * 30.4375 + day
    frac_year = year + day_of_year / 365.25 - 2020.0

    num_freqs = encoding_dim // 2
    # exponents chosen so 2^0 = 1 cycle/year and the band stretches from
    # ~128-year periods (exp=-7) to ~daily (exp=8.5; 2^8.5/year ~ 1/day).
    exponents = torch.linspace(-7.0, 8.5, num_freqs, device=timestamps.device)
    freqs = 2.0 * math.pi * (2.0**exponents)  # (num_freqs,)

    angles = frac_year.unsqueeze(-1) * freqs  # (..., num_freqs)
    sin = torch.sin(angles)
    cos = torch.cos(angles)
    return torch.cat([sin, cos], dim=-1)


def get_static_latlon_encoding(latlon: torch.Tensor, encoding_dim: int) -> torch.Tensor:
    """Multi-frequency sinusoidal encoding of tile-center latitude/longitude.

    Maps ``(lat, lon)`` to a point on the unit sphere ``(x, y, z)`` so longitude
    wrap-around and pole behavior are exact, then applies geometric-spaced
    sinusoidal frequencies on each axis. The trig is done in float64 internally
    so that ``lon=180`` and ``lon=-180`` are identical to machine precision at
    the highest frequencies (which would otherwise blow up in float32).

    Output is split equally across the three axes (x, y, z) and across sin/cos,
    so ``encoding_dim`` must be divisible by 6.

    Args:
        latlon: Tensor of shape ``(..., 2)``; index 0 is latitude in degrees
            [-90, 90], index 1 is longitude in degrees [-180, 180].
        encoding_dim: Output dimension. Must be divisible by 6.

    Returns:
        Tensor of shape ``(..., encoding_dim)``.
    """
    if encoding_dim % 6 != 0:
        raise ValueError(
            "encoding_dim must be divisible by 6 (split across x/y/z and "
            f"sin/cos); got {encoding_dim}"
        )

    in_dtype = latlon.dtype
    work = torch.float64

    lat_rad = latlon[..., 0].to(work) * (math.pi / 180.0)
    lon_rad = latlon[..., 1].to(work) * (math.pi / 180.0)

    cos_lat = torch.cos(lat_rad)
    x = cos_lat * torch.cos(lon_rad)
    y = cos_lat * torch.sin(lon_rad)
    z = torch.sin(lat_rad)
    xyz = torch.stack([x, y, z], dim=-1)  # (..., 3) in float64

    # Frequency band: lowest is one cycle per full sphere axis (period 2 along
    # the unit-axis); highest hits ~25 km on Earth at exp=9. linspace so the
    # band stays the same as we vary num_freqs.
    num_freqs = encoding_dim // 6
    if num_freqs == 1:
        exponents = torch.tensor([0.0], device=latlon.device, dtype=work)
    else:
        exponents = torch.linspace(
            0.0, 9.0, num_freqs, device=latlon.device, dtype=work
        )
    freqs = math.pi * (2.0**exponents)  # (num_freqs,)

    angles = xyz.unsqueeze(-1) * freqs  # (..., 3, num_freqs)
    sin = torch.sin(angles)
    cos = torch.cos(angles)
    # Flatten the (3, num_freqs) axes; sin then cos.
    flat_sin = sin.transpose(-1, -2).reshape(*xyz.shape[:-1], 3 * num_freqs)
    flat_cos = cos.transpose(-1, -2).reshape(*xyz.shape[:-1], 3 * num_freqs)
    out = torch.cat([flat_sin, flat_cos], dim=-1)
    return out.to(dtype=in_dtype)


SIMPLE_TEMPORAL_DIM = 4
# Brings the linear years-since-epoch channel to O(1), matching the
# unit-scale sin/cos channels it is fused with.
SIMPLE_TEMPORAL_YEAR_SCALE = 10.0
TEMPORAL_EPOCH_YEAR = 2020.0


def get_simple_temporal_encoding(timestamps: torch.Tensor) -> torch.Tensor:
    """Minimal 4-number temporal encoding: [frac_year/10, sin, cos, year_valid].

    Returns four channels per timestamp:
      * ``[0]`` ``(year + day_of_year/365.25 - 2020) / 10`` -- a linear,
        absolute measure of years since 2020, scaled to O(1). Zero when the
        year is unknown.
      * ``[1]`` ``sin(2*pi*day_of_year/365.25)`` -- the annual phase. Computed
        from day-of-year directly, which equals ``sin(2*pi*frac_year)``
        (integer years vanish under sin/cos) but stays numerically stable for
        the year-0 sentinel.
      * ``[2]`` the orthogonal ``cos`` annual-phase channel.
      * ``[3]`` ``year_valid`` -- 1.0 when the absolute year is known, 0.0
        when it is not.

    ``year == 0`` is the in-band "year unknown" sentinel: batch-level year
    dropout writes it during pretraining, eval tasks with fabricated dates can
    use it, and the zero-padded timestamps from variable-time eval collation
    hit it instead of producing a ``frac_year = -2020`` outlier. The explicit
    indicator channel exists because a zeroed year channel alone would be
    indistinguishable from a real date near Jan 2020 (the epoch).

    No learnable parameters; deterministic; output dtype matches input.

    Args:
        timestamps: Tensor of shape ``(..., 3)`` where index 0 is day-of-month
            (1-31), index 1 is month (0-indexed, 0-11), index 2 is year.

    Returns:
        Tensor of shape ``(..., 4)``.
    """
    day = timestamps[..., 0].float()
    month = timestamps[..., 1].float()
    year = timestamps[..., 2].float()
    day_of_year = month * 30.4375 + day

    year_valid = (year > 0).float()
    frac_year = year + day_of_year / 365.25 - TEMPORAL_EPOCH_YEAR
    scaled_year = year_valid * frac_year / SIMPLE_TEMPORAL_YEAR_SCALE

    phase = 2.0 * math.pi * day_of_year / 365.25
    return torch.stack(
        [scaled_year, torch.sin(phase), torch.cos(phase), year_valid], dim=-1
    )


def get_simple_latlon_encoding(latlon: torch.Tensor) -> torch.Tensor:
    """Minimal 3-number lat/lon encoding: unit-sphere (x, y, z).

    Maps ``(lat, lon)`` in degrees to a point on the unit sphere:
      * ``x = cos(lat) * cos(lon)``
      * ``y = cos(lat) * sin(lon)``
      * ``z = sin(lat)``

    Longitude wrap-around and pole behavior are exact by construction (no
    frequency expansion). Computed in float64 internally for precision, then
    cast back to the input dtype.

    Args:
        latlon: Tensor of shape ``(..., 2)``; index 0 is latitude in degrees
            [-90, 90], index 1 is longitude in degrees [-180, 180].

    Returns:
        Tensor of shape ``(..., 3)``.
    """
    in_dtype = latlon.dtype
    work = torch.float64
    lat_rad = latlon[..., 0].to(work) * (math.pi / 180.0)
    lon_rad = latlon[..., 1].to(work) * (math.pi / 180.0)
    cos_lat = torch.cos(lat_rad)
    x = cos_lat * torch.cos(lon_rad)
    y = cos_lat * torch.sin(lon_rad)
    z = torch.sin(lat_rad)
    return torch.stack([x, y, z], dim=-1).to(dtype=in_dtype)
