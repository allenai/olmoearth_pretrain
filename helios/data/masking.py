"""Masking module."""

import random
from typing import NamedTuple

import numpy as np
import torch
from einops import rearrange, repeat


class MaskedOutput(NamedTuple):
    """Contains the masked output for a collate function.

    A mask can take 3 values:

    0: seen by the encoder (i.e. makes the key and value tokens in the decoder).
    1: not seen by the encoder, and ignored by the decoder.
    2: not seen by the encoder, and processed by the decoder (the decoder's query values).
    """

    space_time_x: torch.Tensor
    space_x: torch.Tensor
    time_x: torch.Tensor
    static_x: torch.Tensor
    space_time_mask: torch.Tensor
    space_mask: torch.Tensor
    time_mask: torch.Tensor
    static_mask: torch.Tensor
    time_info: torch.Tensor


# seed the random number generator, will want to move this into a seeding function
random.seed(42)


def create_token_mask(
    total_tokens: int,
    tokens_the_decoder_will_unmask: int,
    tokens_the_encoder_will_encode: int,
    batch_size: int,
) -> torch.Tensor:
    """Creates a token mask for a batch of inputs.

    0: seen by the encoder (i.e. makes the key and value tokens in the decoder)
    1: not seen by the encoder, and ignored by the decoder
    2: not seen by the encoder, and processed by the decoder (the decoder's query values)
    """
    flat_tokens = np.concatenate(
        (
            np.ones(
                total_tokens
                - (tokens_the_encoder_will_encode + tokens_the_decoder_will_unmask),
                dtype=np.int_,
            ),
            np.ones(tokens_the_decoder_will_unmask, dtype=np.int_) * 2,
            np.zeros(
                tokens_the_encoder_will_encode,
                dtype=np.int_,
            ),
        )
    )
    batch_flat_tokens = repeat(flat_tokens, "t -> b t", b=batch_size)
    # hopefully this will allow for reproducibility, since random is seeded
    # TODO: We may want to creat this generator once and reuse globally
    rng = np.random.default_rng(random.randint(0, 100))
    batch_flat_tokens = rng.permuted(batch_flat_tokens, axis=1)
    return batch_flat_tokens


def apply_random_masking(
    space_time_x: torch.Tensor,
    space_x: torch.Tensor,
    time_x: torch.Tensor,
    static_x: torch.Tensor,
    encode_ratio: float,
    decode_ratio: float,
    time_info: torch.Tensor,
    patch_size: int,
    ignore_band_groups: list[str] | None = None,
) -> MaskedOutput:
    """Masks out random tokens (blocks of of pxpx1x1).

    e.g. if mask_ratio=0.25, h = w = 8 and p=2, then a mask (for one timestep)
    and channel group) might be
    [0 0 1 1]
    [0 0 1 1]
    [0 0 0 0]
    [0 0 0 0]
    Operates over batches where each item in the batch is independently masked
    """
    if ignore_band_groups is not None:
        raise NotImplementedError("Ignore band groups not implemented")

    # calculate how many channels are going to be used
    b, h, w, t, num_channels_space_time = space_time_x.shape
    num_channels_space = space_x.shape[-1]
    num_channels_time = time_x.shape[-1]
    num_channels_static = static_x.shape[-1]

    h_patch = int(h / patch_size)
    w_patch = int(w / patch_size)

    # Calculate the number of tokens for each data variation type
    num_space_time_tokens = h_patch * w_patch * t * num_channels_space_time
    num_space_tokens = h_patch * w_patch * num_channels_space
    num_time_tokens = t * num_channels_time
    num_static_tokens = num_channels_static

    total_tokens = (
        num_space_time_tokens + num_space_tokens + num_time_tokens + num_static_tokens
    )
    tokens_the_decoder_will_unmask = int(total_tokens * decode_ratio)
    tokens_the_encoder_will_encode = int(total_tokens * encode_ratio)
    token_mask = create_token_mask(
        total_tokens,
        tokens_the_decoder_will_unmask,
        tokens_the_encoder_will_encode,
        b,
    )

    # get the space time mask tokens
    space_time_mask_tokens = token_mask[:, :num_space_time_tokens]
    space_time_mask_tokens = rearrange(
        space_time_mask_tokens,
        "b (h w t c) -> b h w t c",
        h=h_patch,
        w=w_patch,
        t=t,
        c=num_channels_space_time,
    )
    space_time_mask = torch.from_numpy(
        np.repeat(
            np.repeat(space_time_mask_tokens, repeats=patch_size, axis=1),
            repeats=patch_size,
            axis=2,
        )
    )

    # get the space mask tokens
    print(token_mask.shape)
    print(num_space_time_tokens)
    print(num_time_tokens)
    print(num_static_tokens)
    space_tokens = token_mask[
        :, num_space_time_tokens : -(num_time_tokens + num_static_tokens)
    ]
    space_tokens = rearrange(
        space_tokens,
        "b (h w c) -> b h w c",
        h=h_patch,
        w=w_patch,
        c=num_channels_space,
    )
    space_mask = torch.from_numpy(
        np.repeat(
            np.repeat(space_tokens, repeats=patch_size, axis=1),
            repeats=patch_size,
            axis=2,
        )
    )
    # get the time mask tokens
    time_tokens = token_mask[
        :, -(num_time_tokens + num_static_tokens) : -num_static_tokens
    ]
    time_mask = rearrange(time_tokens, "b (t c) -> b t c", t=t, c=num_channels_time)
    time_mask_t = torch.from_numpy(time_mask)

    static_tokens = token_mask[:, -num_static_tokens:]
    static_mask = torch.from_numpy(static_tokens)
    return MaskedOutput(
        space_time_x,
        space_x,
        time_x,
        static_x,
        space_time_mask,
        space_mask,
        time_mask_t,
        static_mask,
        time_info,
    )
