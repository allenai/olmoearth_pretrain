"""A collection of functions for creating position encodings for the OlmoEarth Pretrain model.

The absolute positon embedding functions are adapted from:
https://github.com/bair-climate-initiative/scale-mae/blob/main/mae/util/pos_embed.py

They cover the following:
- 2D sinusoidal position encoding (for spatial data)
- 1D sinusoidal position encoding (for temporal data)
- Month encoding (for temporal data)
"""

import warnings
from enum import StrEnum

import numpy as np
import torch


class PositionEncoding(StrEnum):
    """Supported position encoding modes.

    Covers both spatial-only encodings (``absolute``, the 2D RoPE variants) and
    spatiotemporal encodings (the 3D RoPE variants rotate over time as well as
    the two spatial axes).
    """

    ABSOLUTE = "absolute"
    AXIAL_2D_ROPE = "rope"
    MIXED_2D_ROPE = "rope_mixed"
    AXIAL_3D_ROPE = "rope_3d"
    MIXED_3D_ROPE = "rope_3d_mixed"
    NONE = "none"

    @classmethod
    def values(cls) -> tuple[str, ...]:
        """Return serialized config values accepted for position_encoding."""
        return tuple(encoding.value for encoding in cls)

    @classmethod
    def is_2d_rope(cls, value: str) -> bool:
        """Return whether ``value`` selects a 2D RoPE encoding."""
        return value in {cls.AXIAL_2D_ROPE, cls.MIXED_2D_ROPE}

    @classmethod
    def is_3d_rope(cls, value: str) -> bool:
        """Return whether ``value`` selects a 3D RoPE encoding."""
        return value in {cls.AXIAL_3D_ROPE, cls.MIXED_3D_ROPE}

    @classmethod
    def is_rope(cls, value: str) -> bool:
        """Return whether ``value`` selects any RoPE encoding."""
        return cls.is_2d_rope(value) or cls.is_3d_rope(value)


def resolve_position_encoding(
    position_encoding: str, spatial_pos_encoding: str | None
) -> str:
    """Reconcile ``position_encoding`` with the deprecated ``spatial_pos_encoding``.

    ``spatial_pos_encoding`` is the old name for ``position_encoding`` (renamed
    because the 3D RoPE variants encode time as well as space). It is still
    accepted on init/config for backwards compatibility with old checkpoints and
    callers, but emits a ``DeprecationWarning`` and takes precedence when set.

    Args:
        position_encoding: The canonical (new) value.
        spatial_pos_encoding: The deprecated alias, or ``None`` if not supplied.

    Returns:
        The position encoding mode to use.
    """
    if spatial_pos_encoding is not None:
        warnings.warn(
            "`spatial_pos_encoding` is deprecated and will be removed in a future "
            "release; use `position_encoding` instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        return spatial_pos_encoding
    return position_encoding


# Cumulative days at the start of each month (non-leap year). Index by 0-based
# month. Used to convert (day, month, year) timestamps to a calendar-day count.
_DAYS_BEFORE_MONTH = torch.tensor(
    [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334],
    dtype=torch.float32,
)


def timestamps_to_days(
    timestamps: torch.Tensor, anchor_year: int = 2000
) -> torch.Tensor:
    """Convert ``(day, month, year)`` timestamps to days-since-anchor.

    Helios timestamps are stored as ``(day, month - 1, year)`` (1-indexed day,
    0-indexed month, full Gregorian year). Returns an approximate count of
    calendar days since ``anchor_year-01-01`` using ``365.25`` days/year and
    non-leap-year cumulative month offsets. Off by at most ~1 day, which is
    well below the resolution RoPE rotation cares about.

    Args:
        timestamps: Long tensor with shape ``(..., 3)``. Last dim is
            ``(day, month, year)``.
        anchor_year: Reference year subtracted to keep magnitudes small.
            Defaults to 2000.

    Returns:
        Float tensor of shape ``timestamps.shape[:-1]`` containing days since
        ``anchor_year-01-01``.
    """
    if timestamps.shape[-1] != 3:
        raise ValueError(
            f"timestamps last dim must be 3 (day, month, year), got {timestamps.shape}"
        )
    day = timestamps[..., 0].to(torch.float32)
    month = timestamps[..., 1].to(torch.long)
    year = timestamps[..., 2].to(torch.float32)
    offsets = _DAYS_BEFORE_MONTH.to(timestamps.device)[month]
    return (year - anchor_year) * 365.25 + offsets + (day - 1.0)


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


def apply_2d_axial_rope(
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


def axial_3d_dim_split(head_dim: int, temporal_dim_frac: float) -> tuple[int, int, int]:
    """Compute (d_t, d_row, d_col) chunk sizes for axial 3D RoPE.

    Allocates ``round(head_dim * temporal_dim_frac / 2) * 2`` to the temporal
    chunk (rounded to nearest even) and splits the rest evenly between the
    two spatial axes. Each chunk size must be even because 1D RoPE rotates
    feature pairs.
    """
    if not 0.0 < temporal_dim_frac < 1.0:
        raise ValueError(
            f"temporal_dim_frac must be in (0, 1), got {temporal_dim_frac}"
        )
    d_t = round(head_dim * temporal_dim_frac / 2) * 2
    if d_t < 2:
        raise ValueError(
            f"axial 3D RoPE requires d_t >= 2, got d_t={d_t} for "
            f"head_dim={head_dim}, temporal_dim_frac={temporal_dim_frac}"
        )
    remaining = head_dim - d_t
    if remaining % 4 != 0:
        raise ValueError(
            f"axial 3D RoPE requires (head_dim - d_t) divisible by 4, got "
            f"head_dim={head_dim}, d_t={d_t}, remaining={remaining}"
        )
    d_spatial = remaining // 2
    return d_t, d_spatial, d_spatial


def apply_3d_axial_rope(
    x: torch.Tensor,
    positions: torch.Tensor,
    base: float = 10000.0,
    temporal_dim_frac: float = 0.25,
    temporal_base: float | None = None,
) -> torch.Tensor:
    """Apply axial 3D RoPE to attention query/key tensors.

    Splits the head dimension into three contiguous chunks
    ``(d_t, d_row, d_col)``. ``d_t`` is set by ``temporal_dim_frac`` (rounded
    to nearest even); the remainder is split evenly between row and col. Each
    chunk is rotated by 1D RoPE on its respective coordinate.

    Args:
        x: Attention tensor with shape ``(B, H, N, D)`` or packed shape
            ``(N, H, D)``.
        positions: Spatiotemporal coordinates with shape ``(B, N, 3)`` or
            packed ``(N, 3)``. Last coord dim is ``(t, row, col)``.
        base: RoPE frequency base for the spatial axes.
        temporal_dim_frac: Fraction of ``head_dim`` allocated to the temporal
            chunk (default 0.25, matching the existing additive 1/4 split).
        temporal_base: Optional separate frequency base for the temporal
            axis. ``None`` (default) reuses ``base``. Useful when temporal
            coordinates live on a very different scale (e.g. days) than
            spatial patch indices.
    """
    head_dim = x.shape[-1]
    if positions.shape[-1] != 3:
        raise ValueError(
            f"3D RoPE positions must end with size 3, got {positions.shape}"
        )
    if x.ndim not in (3, 4):
        raise ValueError(f"3D RoPE expects a 3D or 4D attention tensor, got {x.shape}")

    d_t, d_row, d_col = axial_3d_dim_split(head_dim, temporal_dim_frac)

    if x.ndim == 4:
        if positions.ndim != 3:
            raise ValueError(
                "unpacked 3D RoPE expects positions with shape "
                f"(B, N, 3), got {positions.shape}"
            )
        t_pos = positions[:, None, :, 0]
        row_pos = positions[:, None, :, 1]
        col_pos = positions[:, None, :, 2]
    else:
        if positions.ndim != 2:
            raise ValueError(
                "packed 3D RoPE expects positions with shape "
                f"(N, 3), got {positions.shape}"
            )
        t_pos = positions[:, None, 0]
        row_pos = positions[:, None, 1]
        col_pos = positions[:, None, 2]

    x_t = x[..., :d_t]
    x_row = x[..., d_t : d_t + d_row]
    x_col = x[..., d_t + d_row :]

    t_base = base if temporal_base is None else temporal_base
    x_t = apply_1d_rope(x_t, t_pos, t_base)
    x_row = apply_1d_rope(x_row, row_pos, base)
    x_col = apply_1d_rope(x_col, col_pos, base)
    return torch.cat([x_t, x_row, x_col], dim=-1)


def init_3d_mixed_rope_freqs(
    head_dim: int,
    num_heads: int,
    base: float = 10.0,
    rotate: bool = True,
) -> torch.Tensor:
    """Initialize learnable 3D frequencies for RoPE-Mixed (t, row, col).

    Mirrors :func:`init_2d_mixed_rope_freqs` but in 3D: each head receives
    ``head_dim // 2`` 3D frequency vectors, organized as two groups of
    ``head_dim // 4`` pairs along two orthogonal random directions in R^3.
    Magnitudes follow the geometric series ``1 / base^(2k/head_dim)``.

    When ``rotate=False`` the directions default to ``(1, 0, 0)`` (temporal
    axis) and ``(0, 1, 0)`` (row axis), giving a deterministic axial init.

    Returns:
        Tensor of shape ``(3, num_heads, head_dim // 2)`` where the leading
        axis indexes ``(t_freq, row_freq, col_freq)``.
    """
    if head_dim % 4 != 0:
        raise ValueError(
            f"RoPE-Mixed init requires head_dim divisible by 4, got {head_dim}"
        )
    mag = 1.0 / (
        base ** (torch.arange(0, head_dim, 4, dtype=torch.float32) / head_dim)
    )  # (head_dim // 4,)

    if rotate:
        d1 = torch.randn(num_heads, 3)
        d1 = d1 / d1.norm(dim=-1, keepdim=True)
        v = torch.randn(num_heads, 3)
        v = v - (v * d1).sum(dim=-1, keepdim=True) * d1
        d2 = v / v.norm(dim=-1, keepdim=True)
    else:
        d1 = torch.tensor([1.0, 0.0, 0.0]).expand(num_heads, 3).clone()
        d2 = torch.tensor([0.0, 1.0, 0.0]).expand(num_heads, 3).clone()

    # First D/4 pairs along d1, next D/4 pairs along d2.
    # freqs[axis, h, :] gathers the per-pair frequency for that axis.
    freqs_per_axis = []
    for axis in range(3):
        first = mag.unsqueeze(0) * d1[:, axis : axis + 1]  # (H, D/4)
        second = mag.unsqueeze(0) * d2[:, axis : axis + 1]  # (H, D/4)
        freqs_per_axis.append(torch.cat([first, second], dim=-1))  # (H, D/2)
    return torch.stack(freqs_per_axis, dim=0)  # (3, H, D/2)


def apply_3d_mixed_rope(
    x: torch.Tensor,
    positions: torch.Tensor,
    freqs: torch.Tensor,
) -> torch.Tensor:
    """Apply RoPE-Mixed (learnable 3D frequencies) to attention q/k.

    Each complex feature pair is rotated by an angle of the form
    ``theta_t * t + theta_row * row + theta_col * col``, where the
    3-vector ``(theta_t, theta_row, theta_col)`` is a learnable per-head,
    per-pair frequency.

    Args:
        x: Attention tensor with shape ``(B, H, N, D)`` or packed
            ``(N, H, D)``.
        positions: Coordinates with shape ``(B, N, 3)`` or packed ``(N, 3)``.
            Last dim is ``(t, row, col)``.
        freqs: Learnable 3D frequencies of shape ``(3, H, D // 2)``.
    """
    head_dim = x.shape[-1]
    if head_dim % 2 != 0:
        raise ValueError(f"RoPE head dimension must be even, got {head_dim}")
    if positions.shape[-1] != 3:
        raise ValueError(
            f"3D RoPE positions must end with size 3, got {positions.shape}"
        )
    if x.ndim not in (3, 4):
        raise ValueError(f"3D RoPE expects a 3D or 4D attention tensor, got {x.shape}")
    if freqs.ndim != 3 or freqs.shape[0] != 3:
        raise ValueError(
            f"RoPE-Mixed freqs must have shape (3, H, D/2), got {freqs.shape}"
        )
    if freqs.shape[-1] * 2 != head_dim:
        raise ValueError(
            f"RoPE-Mixed freqs last dim must equal head_dim // 2, "
            f"got freqs={freqs.shape}, head_dim={head_dim}"
        )

    dtype = x.dtype
    freqs_t = freqs[0].to(device=x.device, dtype=torch.float32)  # (H, D/2)
    freqs_row = freqs[1].to(device=x.device, dtype=torch.float32)
    freqs_col = freqs[2].to(device=x.device, dtype=torch.float32)
    positions = positions.to(device=x.device, dtype=torch.float32)

    if x.ndim == 4:
        if positions.ndim != 3:
            raise ValueError(
                "unpacked RoPE-Mixed expects positions with shape "
                f"(B, N, 3), got {positions.shape}"
            )
        if freqs.shape[1] != x.shape[1]:
            raise ValueError(
                f"RoPE-Mixed freqs num_heads={freqs.shape[1]} does not match "
                f"attention num_heads={x.shape[1]}"
            )
        t_pos = positions[..., 0]
        row_pos = positions[..., 1]
        col_pos = positions[..., 2]
        angles = (
            t_pos[:, None, :, None] * freqs_t[None, :, None, :]
            + row_pos[:, None, :, None] * freqs_row[None, :, None, :]
            + col_pos[:, None, :, None] * freqs_col[None, :, None, :]
        )  # (B, H, N, D/2)
    else:
        if positions.ndim != 2:
            raise ValueError(
                "packed RoPE-Mixed expects positions with shape "
                f"(N, 3), got {positions.shape}"
            )
        if freqs.shape[1] != x.shape[1]:
            raise ValueError(
                f"RoPE-Mixed freqs num_heads={freqs.shape[1]} does not match "
                f"attention num_heads={x.shape[1]}"
            )
        t_pos = positions[..., 0]
        row_pos = positions[..., 1]
        col_pos = positions[..., 2]
        angles = (
            t_pos[:, None, None] * freqs_t[None, :, :]
            + row_pos[:, None, None] * freqs_row[None, :, :]
            + col_pos[:, None, None] * freqs_col[None, :, :]
        )  # (N, H, D/2)

    cos = torch.repeat_interleave(torch.cos(angles), repeats=2, dim=-1).to(dtype=dtype)
    sin = torch.repeat_interleave(torch.sin(angles), repeats=2, dim=-1).to(dtype=dtype)
    return (x * cos) + (rotate_half(x) * sin)


def init_2d_mixed_rope_freqs(
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


def apply_2d_mixed_rope(
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
