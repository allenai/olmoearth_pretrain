"""Pixel-resolution MIM targets: decode queries placed on single pixels.

With per-pixel Perceiver latents (``PerceiverConfig.pixel_latents``) the latents can sit
at pixel resolution while the MIM queries and targets sit at the sampled patch size.
Pixel targets move the queries and targets to pixel resolution WITHOUT changing the
number of decode queries: each query names a masked token and one pixel of that
token's footprint, its 2D RoPE coordinate is moved to that pixel's center (the
decoder's only spatial signal), and its target is the frozen projection of that single
pixel (the projection-only target applied at ``patch_size=1``).

Every draw produces the same object, :class:`PixelQueries` (``Q`` slots per sample and
modality), decoded by ``Predictor.forward_pixel_queries`` and paired with its targets
by :func:`gather_query_pixels`. The draws differ only in which pixels the slots name;
all are uniform and redrawn every step, and all give one slot per masked token in
total:

* ``shared``: one pixel per (sample, token cell), shared by every token stacked on
  that cell (all timesteps and modalities), so the cell's masked tokens predict one
  pixel's time series.
* ``independent``: one pixel per masked token, so the tokens of a cell may point at
  different pixels.
* ``pooled``: per (sample, modality), as many pixels as masked tokens, drawn without
  replacement from ALL masked pixels (every pixel of every masked token's footprint,
  at that token's timestep). A token's footprint can then get zero, one or several
  target pixels.

The decoder has no self-attention between queries, so decoding a slot is exact for its
pixel: only which pixel each query is scored on changes. Every modality must be
spatial, on the base pixel grid, with one band set (a token's band sets would
otherwise have to share a slot).
"""

from dataclasses import dataclass

import torch
from torch import Tensor

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue

PIXEL_TARGET_DRAWS = ("shared", "independent", "pooled")


@dataclass
class PixelQueries:
    """The pixel queries of ONE modality, as ``Q`` slots per sample.

    Slot ``q`` of sample ``b`` (where ``valid[b, q]``) asks for pixel
    ``pixel[b, q]`` of the token at ``token_index[b, q] = (i, j, t)``: the pixel
    ``(i * p + row, j * p + col)`` at timestep ``t`` (``t = 0`` for static maps).
    Sample ``b`` has as many valid slots as it has masked tokens of this modality.
    """

    token_index: Tensor  # [B, Q, 3] int64 (i, j, t)
    pixel: Tensor  # [B, Q, 2] int64 (row, col) inside the footprint
    valid: Tensor  # [B, Q] bool


def pixel_center_shift(pixel: Tensor, patch_size: int) -> Tensor:
    """Pixel offsets in a footprint -> the query shift in patch units.

    ``(o + 0.5) / p - 0.5`` per axis: pixel ``o`` of patch ``i`` has its center at
    ``i + (o + 0.5) / p - 0.5`` patch units, the stride-1 coordinate of
    ``Perceiver.build_pixel_latent_positions``, so a shifted query lands exactly on
    its per-pixel latent (at a coarser latent stride, inside that latent's footprint).
    """
    return (pixel.to(torch.float32) + 0.5) / patch_size - 0.5


def spatial_token_grid(
    sample: MaskedOlmoEarthSample, patch_size: int
) -> tuple[int, int]:
    """``(h_p, w_p)`` token grid shared by every modality of ``sample``.

    Raises on a non-spatial modality, on a spatial modality stored on a different pixel
    grid (``image_tile_size_factor != 1``), or if the modalities disagree.
    """
    grid: tuple[int, int] | None = None
    for name in sample.modalities:
        spec = Modality.get(name)
        if not spec.is_spatial:
            raise ValueError(
                f"pixel targets support spatial modalities only; got {name}"
            )
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


def token_decode_mask(
    sample: MaskedOlmoEarthSample, name: str, patch_size: int
) -> Tensor:
    """``[B, h_p, w_p, T]`` bool: which tokens of ``name`` the decoder predicts.

    Reads band set 0's mask at each token's top-left pixel, as the patch embedding
    does (one band set per modality is required).
    """
    mask = getattr(sample, sample.get_masked_modality_name(name))
    return mask[:, ::patch_size, ::patch_size, :, 0] == MaskValue.DECODER.value


def _decoded_token_slots(decode: Tensor) -> tuple[Tensor, Tensor]:
    """One slot per decoded token, in row-major ``(i, j, t)`` order.

    Returns ``token_index [B, Q, 3]`` and ``valid [B, Q]``, ``Q`` = the largest
    per-sample count (at least 1, so every modality has an entry).
    """
    batch_size, _, w_p, timesteps = decode.shape
    flat = decode.flatten(1)  # [B, h_p * w_p * T]
    counts = flat.sum(dim=1)
    num_slots = max(int(counts.max().item()) if counts.numel() else 0, 1)
    # Stable sort puts the decoded tokens first, in their original order.
    token_flat = torch.sort((~flat).to(torch.uint8), dim=1, stable=True).indices[
        :, :num_slots
    ]
    t = token_flat % timesteps
    j = (token_flat // timesteps) % w_p
    i = token_flat // (timesteps * w_p)
    valid = torch.arange(num_slots, device=decode.device).unsqueeze(
        0
    ) < counts.unsqueeze(1)
    return torch.stack([i, j, t], dim=-1), valid


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


def shared_pixel_queries(
    sample: MaskedOlmoEarthSample,
    patch_size: int,
    offsets: Tensor,
) -> dict[str, PixelQueries]:
    """One slot per decoded token, at its cell's pixel in ``offsets`` ``[B, h_p, w_p, 2]``."""
    out = {}
    for name in sample.modalities:
        decode = token_decode_mask(sample, name, patch_size).to(offsets.device)
        token_index, valid = _decoded_token_slots(decode)
        batch_index = torch.arange(decode.shape[0], device=decode.device).view(-1, 1)
        pixel = offsets[batch_index, token_index[..., 0], token_index[..., 1]]
        out[name] = PixelQueries(token_index=token_index, pixel=pixel, valid=valid)
    return out


def sample_pixel_queries(
    sample: MaskedOlmoEarthSample,
    patch_size: int,
    draw: str,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> dict[str, PixelQueries]:
    """Draw the pixel queries of every modality (see the module docstring)."""
    if draw not in PIXEL_TARGET_DRAWS:
        raise ValueError(
            f"pixel target draw must be one of {PIXEL_TARGET_DRAWS}, got {draw!r}"
        )
    grid = spatial_token_grid(sample, patch_size)
    if draw == "shared":
        offsets = sample_pixel_offsets(
            sample.batch_size, grid, patch_size, device=device, generator=generator
        )
        return shared_pixel_queries(sample, patch_size, offsets)

    pixels_per_token = patch_size * patch_size
    out = {}
    for name in sample.modalities:
        decode = token_decode_mask(sample, name, patch_size).to(device)
        if draw == "independent":
            token_index, valid = _decoded_token_slots(decode)
            pixel = torch.randint(
                0,
                patch_size,
                (*valid.shape, 2),
                device=device,
                generator=generator,
            )
            out[name] = PixelQueries(token_index=token_index, pixel=pixel, valid=valid)
            continue
        # Pooled: `count` (token, pixel) units per sample, without replacement.
        batch_size, h_p, w_p, timesteps = decode.shape
        counts = decode.flatten(1).sum(dim=1)
        num_slots = max(int(counts.max().item()) if counts.numel() else 0, 1)
        # Uniform scores over every (token, pixel) unit; non-decoded tokens sort last,
        # so the top-`count` scores of a sample are `count` units drawn uniformly.
        scores = torch.rand(
            (batch_size, h_p, w_p, timesteps, pixels_per_token),
            device=device,
            generator=generator,
        )
        scores = scores.masked_fill(~decode.unsqueeze(-1), -1.0).flatten(1)
        unit = scores.topk(num_slots, dim=1).indices  # [B, Q]
        pixel_flat = unit % pixels_per_token
        token_flat = unit // pixels_per_token
        t = token_flat % timesteps
        j = (token_flat // timesteps) % w_p
        i = token_flat // (timesteps * w_p)
        out[name] = PixelQueries(
            token_index=torch.stack([i, j, t], dim=-1),
            pixel=torch.stack(
                [pixel_flat // patch_size, pixel_flat % patch_size], dim=-1
            ),
            valid=torch.arange(num_slots, device=device).unsqueeze(0)
            < counts.unsqueeze(1),
        )
    return out


def gather_query_pixels(
    sample: MaskedOlmoEarthSample,
    queries: dict[str, PixelQueries],
    patch_size: int,
) -> MaskedOlmoEarthSample:
    """The queried pixels as a ``[B, Q, 1, 1, ...]`` field per modality.

    Laid out so the patch embedding at ``patch_size=1`` returns one target per slot,
    in the ``[B, Q, 1, 1, band sets, D]`` layout ``Predictor.forward_pixel_queries``
    emits. Invalid slots hold an arbitrary pixel; the decoded mask leaves them out of
    the loss.
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
