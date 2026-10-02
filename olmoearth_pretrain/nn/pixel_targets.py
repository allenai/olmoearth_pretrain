"""Pixel-resolution MIM targets, subsampled to one pixel per token cell.

With per-pixel Perceiver latents (``PerceiverConfig.pixel_latents``) the latents can sit
at pixel resolution while the MIM queries and targets sit at the sampled patch size. These
helpers move the queries and targets to pixel resolution WITHOUT changing the
number of decode queries: every token cell draws one pixel uniformly at random, the
decoder query of each masked token in that cell is placed at that pixel's center
(the decoder's only spatial signal is RoPE), and the target is the frozen projection
of that single pixel (the projection-only target applied at ``patch_size=1``).

Two draws, both uniform over the cell's pixels and redrawn every step:

* ``shared``: one draw per (sample, token cell), shared by every token stacked on
  that cell (all timesteps, band sets and modalities), so the cell's masked tokens
  predict one pixel's time series.
* ``independent``: one draw per token -- per (sample, cell, timestep, modality) for
  multitemporal modalities and per (sample, cell, modality) for static ones -- so the
  tokens of a cell may point at different pixels. Needs one band set per modality
  (the band sets of a token share its pixels).

Either way there is one query and one target per masked token. The decoder has no
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


def sample_independent_pixel_offsets(
    sample: MaskedOlmoEarthSample,
    patch_size: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> dict[str, Tensor]:
    """One draw per token: ``{modality: offsets}`` for every spatial modality.

    Multitemporal modalities get ``[B, h_p, w_p, T, 2]`` (a pixel per timestep),
    static ones ``[B, h_p, w_p, 2]``; int64 ``(row, col)`` within the cell.
    """
    grid = spatial_token_grid(sample, patch_size)
    offsets = {}
    for name in sample.modalities:
        spec = Modality.get(name)
        if not spec.is_spatial:
            continue
        shape: tuple[int, ...] = (sample.batch_size, *grid)
        if spec.is_multitemporal:
            shape = (*shape, getattr(sample, name).shape[3])
        offsets[name] = torch.randint(
            0, patch_size, (*shape, 2), device=device, generator=generator
        )
    return offsets


def offsets_to_query_shift(offsets: Tensor, patch_size: int) -> Tensor:
    """Pixel offsets -> the query shift in patch units: ``(o + 0.5) / p - 0.5``.

    Matches ``joint_latent.build_pixel_latent_positions`` at stride 1: pixel ``o`` of
    patch ``i`` has its center at ``i + (o + 0.5) / p - 0.5`` patch units, so a
    shifted query lands exactly on its per-pixel latent's coordinate (at a coarser
    latent stride it lands inside that latent's footprint).
    """
    return (offsets.to(torch.float32) + 0.5) / patch_size - 0.5


def _gather(field: Tensor, offsets: Tensor, patch_size: int) -> Tensor:
    """``[B, H, W, ...]`` -> ``[B, h_p, w_p, ...]`` at the drawn pixels.

    ``offsets`` is ``[B, h_p, w_p, 2]`` (one pixel per cell for every timestep) or
    ``[B, h_p, w_p, T, 2]`` (a pixel per cell and timestep; ``field`` then has its
    timestep axis at dim 3).
    """
    batch_size, h_p, w_p = offsets.shape[:3]
    device = offsets.device
    if offsets.ndim == 4:
        index_shape: tuple[int, ...] = (batch_size, h_p, w_p)
        tail: tuple[Tensor, ...] = ()
    else:
        timesteps = offsets.shape[3]
        index_shape = (batch_size, h_p, w_p, timesteps)
        tail = (torch.arange(timesteps, device=device).view(1, 1, 1, timesteps),)
    pad = (1,) * (len(index_shape) - 3)
    rows = (
        torch.arange(h_p, device=device).view(1, h_p, 1, *pad) * patch_size
        + offsets[..., 0]
    )
    cols = (
        torch.arange(w_p, device=device).view(1, 1, w_p, *pad) * patch_size
        + offsets[..., 1]
    )
    batch_index = torch.arange(batch_size, device=device).view(batch_size, 1, 1, *pad)
    return field[(batch_index, rows, cols, *tail)]


def gather_pixels(
    sample: MaskedOlmoEarthSample,
    offsets: Tensor | dict[str, Tensor],
    patch_size: int,
) -> MaskedOlmoEarthSample:
    """Keep only the drawn pixel of each token, for every spatial modality.

    Each spatial field (and its mask) goes from ``[B, H, W, ...]`` to
    ``[B, h_p, w_p, ...]``, cell ``(i, j)`` holding pixel
    ``(i * p + offsets[..., 0], j * p + offsets[..., 1])``. ``offsets`` is one
    ``[B, h_p, w_p, 2]`` tensor shared by every modality (the ``shared`` draw) or a
    ``{modality: offsets}`` dict (the ``independent`` draw, see
    :func:`sample_independent_pixel_offsets`). Projected at ``patch_size=1`` this
    gives one target per token, on the same grid (and with the same token layout) as
    the patch-size targets it replaces. Non-spatial modalities and the timestamps are
    passed through unchanged.
    """
    updates = {}
    for name in sample.modalities:
        if not Modality.get(name).is_spatial:
            continue
        modality_offsets = offsets[name] if isinstance(offsets, dict) else offsets
        mask_name = sample.get_masked_modality_name(name)
        updates[name] = _gather(getattr(sample, name), modality_offsets, patch_size)
        mask = getattr(sample, mask_name)
        if mask is not None:
            updates[mask_name] = _gather(mask, modality_offsets, patch_size)
    return sample._replace(**updates)
