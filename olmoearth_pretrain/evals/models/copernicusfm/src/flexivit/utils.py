# type: ignore
import collections.abc
import math
from itertools import repeat
from typing import Any

import torch
import torch.nn.functional as F


def to_2tuple(x: Any) -> tuple:
    if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
        return tuple(x)
    return tuple(repeat(x, 2))


def resize_abs_pos_embed(
    pos_embed: torch.Tensor,
    new_size: tuple[int, int],
    old_size: int | tuple[int, int] | None = None,
    num_prefix_tokens: int = 1,
    interpolation: str = "bicubic",
    antialias: bool = True,
) -> torch.Tensor:
    """Resize absolute position embeddings to a target resolution via interpolation

    Modified from: https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/pos_embed.py

    Args:
        pos_embed: Position embeddings tensor of size [b, n, d]
        new_size: Target [height, width] of embedding
        old_size: Original [height, width] of embedding
        num_prefix_tokens: Number of non-spatial prefix tokens (eg. cls)
        interpolation: Resize interpolation type
        antialias: Whether to apply antialiasing resizing
    Returns:
        Resized pos_embed of size [b, n', d]
    """
    new_size = to_2tuple(new_size)
    new_ntok = new_size[0] * new_size[1]

    if not old_size:
        old_size = int(math.sqrt(pos_embed.shape[1] - num_prefix_tokens))  # type:ignore
    old_size = to_2tuple(old_size)

    # Return if no resize necessary
    if new_size == old_size:
        return pos_embed

    if num_prefix_tokens:
        posemb_prefix, pos_embed = (
            pos_embed[:, :num_prefix_tokens],
            pos_embed[:, num_prefix_tokens:],
        )
    else:
        posemb_prefix, pos_embed = None, pos_embed

    # Interpolate position embedding
    pos_embed = pos_embed.reshape(1, old_size[0], old_size[1], -1).permute(0, 3, 1, 2)
    pos_embed = F.interpolate(
        pos_embed, size=new_size, mode=interpolation, antialias=antialias
    )
    pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(1, new_ntok, -1)

    # Add back extra prefix tokens
    if posemb_prefix is not None:
        pos_embed = torch.cat([posemb_prefix, pos_embed], dim=1)

    return pos_embed
