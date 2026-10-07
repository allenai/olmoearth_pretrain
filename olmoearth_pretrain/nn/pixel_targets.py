"""Pixel-resolution MIM targets, subsampled to one pixel per token cell.

With per-pixel Perceiver latents (``Encoder.forward(latent_patch_size=...)``) the
latents can sit at pixel resolution while the MIM queries and targets sit at the sampled
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

    Raises if a spatial modality is stored on a different pixel grid
    (``image_tile_size_factor != 1``) or if the spatial modalities disagree.
    """
    grid: tuple[int, int] | None = None
    for name in sample.modalities:
        spec = Modality.get(name)
        if not spec.is_spatial:
            continue
        if spec.image_tile_size_factor != 1:
            raise ValueError(
                f"pixel targets need every spatial modality on the base pixel grid; "
                f"{name} has image_tile_size_factor={spec.image_tile_size_factor}"
            )
        height, width = getattr(sample, name).shape[1:3]
        if height % patch_size or width % patch_size:
            raise ValueError(
                f"{name} is {height}x{width}, not a multiple of patch_size={patch_size}"
            )
        modality_grid = (height // patch_size, width // patch_size)
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

    Matches ``Perceiver.build_register_positions`` at latent patch size 1: pixel
    ``o`` of patch ``i`` has its center at ``i + (o + 0.5) / p - 0.5`` patch units, so a
    shifted query lands exactly on its per-pixel latent's coordinate (at a coarser
    latent patch size it lands inside that latent's footprint).
    """
    return (offsets.to(torch.float32) + 0.5) / patch_size - 0.5


def gather_pixels(
    sample: MaskedOlmoEarthSample, offsets: Tensor, patch_size: int
) -> MaskedOlmoEarthSample:
    """Keep only the drawn pixel of each token cell, for every spatial modality.

    Each spatial field (and its mask) goes from ``[B, H, W, ...]`` to
    ``[B, h_p, w_p, ...]``, cell ``(i, j)`` holding pixel
    ``(i * p + offsets[..., 0], j * p + offsets[..., 1])``. Projected at
    ``patch_size=1`` this gives one target per token cell, on the same grid (and
    with the same token layout) as the patch-size targets it replaces. Non-spatial
    modalities and the timestamps are passed through unchanged.
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
        updates[name] = getattr(sample, name)[batch_index, rows, cols]
        mask = getattr(sample, mask_name)
        if mask is not None:
            updates[mask_name] = mask[batch_index, rows, cols]
    return sample._replace(**updates)
