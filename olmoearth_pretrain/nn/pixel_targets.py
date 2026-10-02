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

* ``pooled``: per (sample, modality), draw as many targets as there are masked
  tokens, uniformly and without replacement from ALL masked pixels (every pixel of
  every masked token's footprint, at that token's timestep). A token's footprint can
  then get zero, one or several target pixels. Queries no longer map one-to-one onto
  tokens, so this mode decodes a flat list (:class:`PooledPixelQueries`, decoded by
  ``Predictor.forward_pooled``).

In every mode there is one query and one target per masked token in total. The decoder has no
self-attention between queries, so decoding this subset is exact for the drawn
pixels: only which pixel each masked token is scored on changes.
"""

from dataclasses import dataclass

import torch
from torch import Tensor

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue


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


@dataclass
class PooledPixelQueries:
    """The pooled pixel targets of ONE modality, as ``Q`` slots per sample.

    Slot ``q`` of sample ``b`` (where ``valid[b, q]``) asks for pixel
    ``pixel[b, q]`` of the token at ``token_index[b, q] = (i, j, t)``: the pixel
    ``(i * p + row, j * p + col)`` at timestep ``t`` (``t = 0`` for static maps).
    Sample ``b`` has as many valid slots as it has masked tokens of this modality.
    """

    token_index: Tensor  # [B, Q, 3] int64 (i, j, t)
    pixel: Tensor  # [B, Q, 2] int64 (row, col) inside the footprint
    valid: Tensor  # [B, Q] bool


def token_decode_mask(
    sample: MaskedOlmoEarthSample, name: str, patch_size: int
) -> Tensor:
    """``[B, h_p, w_p, T]`` bool: which tokens of ``name`` the decoder predicts.

    Reads band set 0's mask at each token's top-left pixel, as the patch embedding
    does (one band set per modality is required by the pooled draw).
    """
    mask = getattr(sample, sample.get_masked_modality_name(name))
    return mask[:, ::patch_size, ::patch_size, :, 0] == MaskValue.DECODER.value


def sample_pooled_pixel_queries(
    sample: MaskedOlmoEarthSample,
    patch_size: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> dict[str, PooledPixelQueries]:
    """Pooled draw for every spatial modality (see the module docstring)."""
    pixels_per_token = patch_size * patch_size
    out = {}
    for name in sample.modalities:
        if not Modality.get(name).is_spatial:
            raise ValueError(
                f"pixel_target_draw='pooled' supports spatial modalities only; got "
                f"{name}"
            )
        decode = token_decode_mask(sample, name, patch_size).to(device)
        batch_size, h_p, w_p, timesteps = decode.shape
        counts = decode.flatten(1).sum(dim=1)  # [B] targets per sample
        # At least one slot, so a modality with nothing masked still has a (fully
        # invalid) entry and predictions and targets list the same modalities.
        num_slots = max(int(counts.max().item()) if counts.numel() else 0, 1)
        # Uniform scores over every (token, pixel) unit; non-decoded tokens sort last.
        scores = torch.rand(
            (batch_size, h_p, w_p, timesteps, pixels_per_token),
            device=device,
            generator=generator,
        )
        scores = scores.masked_fill(~decode.unsqueeze(-1), -1.0).flatten(1)
        # Top-`count` scores per sample = `count` units drawn without replacement.
        unit = scores.topk(num_slots, dim=1).indices  # [B, Q]
        pixel_flat = unit % pixels_per_token
        token_flat = unit // pixels_per_token
        t = token_flat % timesteps
        j = (token_flat // timesteps) % w_p
        i = token_flat // (timesteps * w_p)
        out[name] = PooledPixelQueries(
            token_index=torch.stack([i, j, t], dim=-1),
            pixel=torch.stack(
                [pixel_flat // patch_size, pixel_flat % patch_size], dim=-1
            ),
            valid=torch.arange(num_slots, device=device).unsqueeze(0)
            < counts.unsqueeze(1),
        )
    return out


def gather_pooled_pixels(
    sample: MaskedOlmoEarthSample,
    queries: dict[str, PooledPixelQueries],
    patch_size: int,
) -> MaskedOlmoEarthSample:
    """The drawn pixels as a ``[B, Q, 1, 1, ...]`` field per modality.

    Laid out so the patch embedding at ``patch_size=1`` returns one target per slot,
    in the ``[B, Q, 1, 1, band sets, D]`` layout ``Predictor.forward_pooled`` emits.
    Invalid slots hold an arbitrary pixel; the decoded mask leaves them out of the
    loss.
    """
    updates = {}
    for name, q in queries.items():
        batch_index = torch.arange(q.valid.shape[0], device=q.valid.device).view(-1, 1)
        rows = q.token_index[..., 0] * patch_size + q.pixel[..., 0]
        cols = q.token_index[..., 1] * patch_size + q.pixel[..., 1]
        t = q.token_index[..., 2]
        mask_name = sample.get_masked_modality_name(name)
        for field_name in (name, mask_name):
            field = getattr(sample, field_name)
            if field is None:
                continue
            picked = field[batch_index, rows, cols, t]  # [B, Q, ...]
            updates[field_name] = picked.unsqueeze(2).unsqueeze(3)
    return sample._replace(**updates)
