"""Pixel-resolution MIM targets, subsampled to one pixel per token cell.

With per-pixel location queries (``Encoder.forward(location_query_patch_size=...)``) the
location queries can sit at pixel resolution while the MIM decode queries and targets sit at the sampled
patch size. These helpers move the queries and targets to pixel resolution WITHOUT
changing the number of decode queries: every token cell draws one pixel uniformly at
random, the decoder query of each masked token in that cell is placed at that pixel's
center (the decoder's only spatial signal is RoPE), and the target is the frozen
projection of that single pixel (the projection-only target applied at
``patch_size=1``).

The draw is per (sample, token cell) and shared across modalities, timesteps and
band sets, so all queries of a cell predict the same pixel. The decoder has no
self-attention between queries, so decoding this subset is exact for the drawn
pixels: only which pixel each masked token is scored on changes.
"""

import torch
from torch import Tensor

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample


def spatial_token_grid(
    sample: MaskedOlmoEarthSample, patch_size: int
) -> tuple[int, int]:
    """``(h_p, w_p)`` token grid shared by every spatial modality of ``sample``.

    A modality stored on a finer pixel grid (``image_tile_size_factor = f > 1``, e.g.
    ``naip_10``) tokenizes ``p * f`` of its own pixels per token, so it shares the base
    grid. Raises if a spatial modality is on a coarser grid (``f < 1``) or if the
    spatial modalities disagree.
    """
    grid: tuple[int, int] | None = None
    for name in sample.modalities:
        spec = Modality.get(name)
        if not spec.is_spatial:
            continue
        factor = spec.image_tile_size_factor
        if factor < 1:
            raise ValueError(
                f"pixel targets need image_tile_size_factor >= 1; {name} has {factor}"
            )
        token_px = patch_size * factor
        height, width = getattr(sample, name).shape[1:3]
        if height % token_px or width % token_px:
            raise ValueError(
                f"{name} is {height}x{width}, not a multiple of patch_size * "
                f"image_tile_size_factor = {token_px}"
            )
        modality_grid = (height // token_px, width // token_px)
        if grid is not None and modality_grid != grid:
            raise ValueError(
                f"spatial modalities disagree on the token grid: {grid} vs "
                f"{modality_grid} ({name})"
            )
        grid = modality_grid
    if grid is None:
        raise ValueError("pixel targets need at least one spatial modality")
    return grid


def sample_pixel_offsets(
    batch_size: int,
    grid: tuple[int, int],
    patch_size: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Draw one pixel per token cell, uniformly: ``[B, h_p, w_p, 2]`` int64 (row, col)."""
    return torch.randint(
        0,
        patch_size,
        (batch_size, grid[0], grid[1], 2),
        device=device,
        generator=generator,
    )


def offsets_to_query_shift(offsets: Tensor, patch_size: int) -> Tensor:
    """Pixel offsets -> the query shift in patch units: ``(o + 0.5) / p - 0.5``.

    Matches ``AttentionPooler.build_location_query_positions`` at location-query patch size 1: pixel
    ``o`` of patch ``i`` has its center at ``i + (o + 0.5) / p - 0.5`` patch units, so a
    shifted query lands exactly on its per-pixel location query's coordinate (at a coarser
    location-query patch size it lands inside that query's footprint).
    """
    return (offsets.to(torch.float32) + 0.5) / patch_size - 0.5


def _gather_block(
    field: Tensor, offsets: Tensor, patch_size: int, factor: int
) -> Tensor:
    """Keep the ``factor x factor`` fine-pixel block under each cell's drawn base pixel.

    For a modality on a ``factor``-times finer pixel grid: the drawn base pixel
    ``(i * p + r, j * p + c)`` covers the fine pixels starting at
    ``((i * p + r) * f, (j * p + c) * f)``; that block is kept, so
    ``[B, H * f, W * f, ...]`` -> ``[B, h_p * f, w_p * f, ...]`` (one ``f x f`` block
    per cell, i.e. one token per cell when tokenized at ``patch_size=1``).
    """
    batch_size, h_p, w_p, _ = offsets.shape
    device = offsets.device
    f = factor
    within = torch.arange(f, device=device)
    rows = (
        (torch.arange(h_p, device=device) * patch_size).view(1, h_p, 1, 1, 1)
        + offsets[..., 0].view(batch_size, h_p, 1, w_p, 1)
    ) * f + within.view(1, 1, f, 1, 1)
    cols = (
        (torch.arange(w_p, device=device) * patch_size).view(1, 1, 1, w_p, 1)
        + offsets[..., 1].view(batch_size, h_p, 1, w_p, 1)
    ) * f + within.view(1, 1, 1, 1, f)
    batch_index = torch.arange(batch_size, device=device).view(batch_size, 1, 1, 1, 1)
    out = field[batch_index, rows, cols]  # [B, h_p, f, w_p, f, ...]
    return out.reshape(batch_size, h_p * f, w_p * f, *out.shape[5:])


def gather_pixels(
    sample: MaskedOlmoEarthSample, offsets: Tensor, patch_size: int
) -> MaskedOlmoEarthSample:
    """Keep only the drawn pixel of each token cell, for every spatial modality.

    Each spatial field (and its mask) goes from ``[B, H, W, ...]`` to
    ``[B, h_p, w_p, ...]``, cell ``(i, j)`` holding pixel
    ``(i * p + offsets[..., 0], j * p + offsets[..., 1])``. Projected at
    ``patch_size=1`` this gives one target per token cell, on the same grid (and
    with the same token layout) as the patch-size targets it replaces. A modality on
    a finer pixel grid (``image_tile_size_factor = f > 1``, e.g. ``naip_10``) keeps
    the ``f x f`` block under the drawn base pixel instead (see
    :func:`_gather_block`), which is again one token per cell at ``patch_size=1``.
    Non-spatial modalities and the timestamps are passed through unchanged.
    """
    batch_size, h_p, w_p, _ = offsets.shape
    device = offsets.device
    rows = (
        torch.arange(h_p, device=device).view(1, h_p, 1) * patch_size + offsets[..., 0]
    )
    cols = (
        torch.arange(w_p, device=device).view(1, 1, w_p) * patch_size + offsets[..., 1]
    )
    batch_index = torch.arange(batch_size, device=device).view(batch_size, 1, 1)
    updates = {}
    for name in sample.modalities:
        if not Modality.get(name).is_spatial:
            continue
        mask_name = sample.get_masked_modality_name(name)
        factor = Modality.get(name).image_tile_size_factor
        for field_name in (name, mask_name):
            value = getattr(sample, field_name)
            if value is None:
                continue
            if factor > 1:
                updates[field_name] = _gather_block(value, offsets, patch_size, factor)
            else:
                updates[field_name] = value[batch_index, rows, cols]
    return sample._replace(**updates)
