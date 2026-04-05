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

import math
from enum import StrEnum

import numpy as np
import torch
from torch import nn


class TimestampEncodingMode(StrEnum):
    """Mode for encoding temporal information."""

    LEGACY = "legacy"
    UNIFIED = "unified"
    LEARNED = "learned"
    STATIC = "static"


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


def timestamps_to_learned_input(timestamps: torch.Tensor) -> torch.Tensor:
    """Convert [day, month, year] timestamps to a 3-float learned-encoding input.

    Produces [fractional_year_since_2020, sin(2*pi*frac), cos(2*pi*frac)]
    where fractional_year_since_2020 is a continuous real number (Jan 1 2020 = 0,
    Jan 1 2021 = 1, etc.) and the sin/cos pair captures yearly cyclical seasonality.

    Args:
        timestamps: Tensor of shape (B, T, 3) where [..., 0] is day (1-31),
            [..., 1] is month (0-indexed, 0-11), [..., 2] is year.

    Returns:
        Tensor of shape (B, T, 3).
    """
    day = timestamps[..., 0].float()
    month = timestamps[..., 1].float()
    year = timestamps[..., 2].float()

    day_of_year = month * 30.4375 + day
    fractional_year = year + day_of_year / 365.25 - 2020.0

    sin_val = torch.sin(2 * np.pi * fractional_year)
    cos_val = torch.cos(2 * np.pi * fractional_year)

    return torch.stack([fractional_year, sin_val, cos_val], dim=-1)


class TimestampMLP(nn.Module):
    """Learned timestamp encoding via a small MLP.

    Takes raw [day, month, year] timestamps, converts them to a 3-float
    representation (fractional year + sin/cos yearly cycle), and maps them
    through a 2-layer MLP to produce timestamp embeddings.

    Args:
        output_dim: Output embedding dimension (should be 2 * embedding_dim_per_embedding_type).
        hidden_dim: Hidden layer dimension. Default: 64.
        activation: Activation module between the two linear layers.
            Default: nn.GELU().
    """

    def __init__(
        self,
        output_dim: int,
        hidden_dim: int = 64,
        activation: nn.Module | None = None,
    ):
        """Initialize the TimestampMLP."""
        super().__init__()
        if activation is None:
            activation = nn.GELU()
        self.mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            activation,
            nn.Linear(hidden_dim, output_dim),
        )
        # Preserve default PyTorch init — skip the parent model's xavier_uniform_ init
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                m._skip_custom_init = True  # type: ignore[attr-defined]

    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        """Produce learned timestamp embeddings.

        Args:
            timestamps: (B, T, 3) raw [day, month, year] tensor.

        Returns:
            (B, T, output_dim) learned timestamp embedding.
        """
        x = timestamps_to_learned_input(timestamps)
        # Match the dtype of the MLP weights (e.g. bfloat16 under FSDP)
        x = x.to(dtype=next(self.mlp.parameters()).dtype)
        return self.mlp(x)


def latlon_to_learned_input(latlon: torch.Tensor, num_freqs: int = 20) -> torch.Tensor:
    """Convert lat/lon in degrees to a multi-frequency spherical encoding.

    Maps lat/lon to a point on the unit sphere (x, y, z), then applies
    multi-frequency sinusoidal encoding to each coordinate. This provides
    a discontinuity-free representation at multiple spatial scales.

    Output features: [x, y, z, sin(2^0·π·x), cos(2^0·π·x), ...,
    sin(2^(L-1)·π·z), cos(2^(L-1)·π·z)] for a total of 3 + 6*num_freqs.

    Args:
        latlon: Tensor of shape (..., 2) where [..., 0] is latitude in degrees
            [-90, 90] and [..., 1] is longitude in degrees [-180, 180].
        num_freqs: Number of frequency bands (L). Default: 20.

    Returns:
        Tensor of shape (..., 3 + 6 * num_freqs).
    """
    lat_rad = latlon[..., 0] * (math.pi / 180.0)
    lon_rad = latlon[..., 1] * (math.pi / 180.0)

    # Unit sphere coordinates
    cos_lat = torch.cos(lat_rad)
    x = cos_lat * torch.cos(lon_rad)
    y = cos_lat * torch.sin(lon_rad)
    z = torch.sin(lat_rad)

    xyz = torch.stack([x, y, z], dim=-1)  # (..., 3)

    # Multi-frequency encoding
    # freqs: [2^0, 2^1, ..., 2^(L-1)] * pi
    freqs = (
        2.0 ** torch.arange(num_freqs, device=latlon.device, dtype=latlon.dtype)
    ) * math.pi
    # (..., 3, 1) * (L,) -> (..., 3, L)
    angles = xyz.unsqueeze(-1) * freqs
    # (..., 3, L) -> (..., 6L) interleaved sin/cos
    encoded = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # (..., 3, 2L)
    encoded = encoded.flatten(-2)  # (..., 6L)

    return torch.cat([xyz, encoded], dim=-1)  # (..., 3 + 6L)


def compute_per_token_latlon(
    latlon: torch.Tensor,
    grid_h: int,
    grid_w: int,
    meters_per_token: float,
) -> torch.Tensor:
    """Compute per-token lat/lon from tile center and grid geometry.

    Args:
        latlon: Tile center coordinates (B, 2) in degrees [lat, lon].
        grid_h: Number of tokens in height dimension.
        grid_w: Number of tokens in width dimension.
        meters_per_token: Spatial extent of each token in meters
            (input_res * patch_size).

    Returns:
        Per-token lat/lon tensor of shape (B, H, W, 2) in degrees.
    """
    METERS_PER_DEG = 111320.0

    lat_center = latlon[:, 0]  # (B,)
    lon_center = latlon[:, 1]  # (B,)

    device = latlon.device
    h_offsets = torch.arange(grid_h, device=device) - (grid_h - 1) / 2.0  # (H,)
    w_offsets = torch.arange(grid_w, device=device) - (grid_w - 1) / 2.0  # (W,)

    # Latitude offsets (increasing h = south = decreasing lat)
    lat_offset_deg = -h_offsets * meters_per_token / METERS_PER_DEG  # (H,)

    # Longitude offsets (vary per batch element due to cos(lat) correction)
    cos_lat = torch.cos(lat_center * (math.pi / 180.0)).clamp(min=1e-6)  # (B,)
    lon_offset_deg = (
        w_offsets[None, :] * meters_per_token / (METERS_PER_DEG * cos_lat[:, None])
    )  # (B, W)

    # Broadcast to (B, H, W)
    token_lat = lat_center[:, None, None] + lat_offset_deg[None, :, None]  # (B, H, 1)
    token_lon = lon_center[:, None, None] + lon_offset_deg[:, None, :]  # (B, 1, W)

    return torch.stack(
        [token_lat.expand(-1, grid_h, grid_w), token_lon.expand(-1, grid_h, grid_w)],
        dim=-1,
    )  # (B, H, W, 2)


class LatLonMLP(nn.Module):
    """Learned geographic encoding via a small MLP.

    Takes lat/lon coordinates in degrees, maps them to the unit sphere,
    applies multi-frequency sinusoidal encoding for multi-scale resolution,
    and maps through a 2-layer MLP to produce geographic embeddings.

    Args:
        output_dim: Output embedding dimension (should be embedding_dim_per_embedding_type).
        hidden_dim: Hidden layer dimension. Default: 64.
        num_freqs: Number of frequency bands for the positional encoding. Default: 20.
        activation: Activation module between the two linear layers.
            Default: nn.GELU().
    """

    def __init__(
        self,
        output_dim: int,
        hidden_dim: int = 128,
        num_freqs: int = 20,
        activation: nn.Module | None = None,
    ):
        """Initialize the LatLonMLP."""
        super().__init__()
        self.num_freqs = num_freqs
        if activation is None:
            activation = nn.GELU()
        input_dim = 3 + 6 * num_freqs
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, output_dim),
        )
        # Preserve default PyTorch init — skip the parent model's xavier_uniform_ init
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                m._skip_custom_init = True  # type: ignore[attr-defined]

    def forward(self, latlon: torch.Tensor) -> torch.Tensor:
        """Produce learned geographic embeddings.

        Args:
            latlon: (..., 2) lat/lon in degrees.

        Returns:
            (..., output_dim) learned geographic embedding.
        """
        x = latlon_to_learned_input(latlon, self.num_freqs)
        # Match the dtype of the MLP weights (e.g. bfloat16 under FSDP)
        x = x.to(dtype=next(self.mlp.parameters()).dtype)
        return self.mlp(x)


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
    # Geometric progression: 2^(-7) to 2^8.5 cycles/year
    # Low = 128-year period, High ≈ daily resolution
    exponents = torch.linspace(-7.0, 8.5, num_freqs, device=timestamps.device)
    freqs = 2.0 * math.pi * (2.0**exponents)  # (num_freqs,)

    # (B, T) x (num_freqs,) -> (B, T, num_freqs)
    angles = frac_year.unsqueeze(-1) * freqs
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


def get_static_spatial_encoding(
    latlon: torch.Tensor, encoding_dim: int
) -> torch.Tensor:
    """Static multi-frequency sinusoidal geographic encoding.

    Maps lat/lon to the unit sphere (x, y, z), then applies geometric-spaced
    sinusoidal frequencies to each coordinate. Handles longitude wrap-around
    naturally via the sphere mapping. Resolves positions down to ~30cm.

    Args:
        latlon: Tensor of shape (..., 2) where [..., 0] is latitude and
            [..., 1] is longitude, both in degrees.
        encoding_dim: Output encoding dimension.

    Returns:
        Tensor of shape (..., encoding_dim).
    """
    lat_rad = latlon[..., 0] * (math.pi / 180.0)
    lon_rad = latlon[..., 1] * (math.pi / 180.0)

    cos_lat = torch.cos(lat_rad)
    x = cos_lat * torch.cos(lon_rad)
    y = cos_lat * torch.sin(lon_rad)
    z = torch.sin(lat_rad)
    xyz = torch.stack([x, y, z], dim=-1)  # (..., 3)

    num_freqs = encoding_dim // 6
    remainder = encoding_dim - num_freqs * 6

    # Geometric progression: 2^0 to 2^21 (planet-scale to ~30cm)
    max_exp = min(21, max(num_freqs - 1, 0))
    exponents = torch.linspace(0.0, max_exp, num_freqs, device=latlon.device)
    freqs = math.pi * (2.0**exponents)  # (num_freqs,)

    # (..., 3, 1) * (num_freqs,) -> (..., 3, num_freqs)
    angles = xyz.unsqueeze(-1) * freqs
    encoded = torch.cat(
        [torch.sin(angles), torch.cos(angles)], dim=-1
    )  # (..., 3, 2*num_freqs)
    result = encoded.flatten(-2)  # (..., 6*num_freqs)

    # Fill remainder dims with raw (x, y, z) or zero-pad
    if remainder > 0:
        pad = (
            xyz[..., :remainder]
            if remainder <= 3
            else torch.cat(
                [
                    xyz,
                    torch.zeros(*xyz.shape[:-1], remainder - 3, device=latlon.device),
                ],
                dim=-1,
            )
        )
        result = torch.cat([result, pad], dim=-1)

    return result
