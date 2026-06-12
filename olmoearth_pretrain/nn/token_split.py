"""Shared helpers for splitting decoder and encoder token streams."""

from typing import NamedTuple

import torch
from torch import Tensor

from olmoearth_pretrain.datatypes import MaskValue


class SplitTokensByMaskResult(NamedTuple):
    """Token groups and bookkeeping produced by split_tokens_by_mask."""

    tokens_to_decode: Tensor
    unmasked_tokens: Tensor
    tokens_to_decode_mask: Tensor
    unmasked_tokens_mask: Tensor
    indices: Tensor
    seqlens_tokens_to_decode: Tensor
    seqlens_unmasked_tokens: Tensor
    max_length_of_decoded_tokens: Tensor
    max_length_of_unmasked_tokens: Tensor


def split_tokens_by_mask(tokens: Tensor, mask: Tensor) -> SplitTokensByMaskResult:
    """Split token sequences into decoder and online-encoder groups.

    Missing tokens are sorted into the unused middle section with
    ``TARGET_ENCODER_ONLY`` tokens. The input mask is not modified.

    Args:
        tokens: Tokens to split of shape [B, T, D].
        mask: Mask of shape [B, T].

    Returns:
        Split token groups, their masks, sequence lengths, and indices for
        restoring the original token ordering.
    """
    original_mask_dtype = mask.dtype
    normalized_mask = torch.where(
        mask == MaskValue.MISSING.value,
        torch.full_like(mask, MaskValue.TARGET_ENCODER_ONLY.value),
        mask,
    )

    sorted_mask, indices = torch.sort(
        normalized_mask.int(), dim=1, descending=True, stable=True
    )
    sorted_tokens = tokens.gather(1, indices[:, :, None].expand_as(tokens))

    decoder_mask = sorted_mask == MaskValue.DECODER.value
    online_encoder_mask = sorted_mask == MaskValue.ONLINE_ENCODER.value

    seqlens_unmasked_tokens = online_encoder_mask.sum(dim=-1)
    max_length_of_unmasked_tokens = seqlens_unmasked_tokens.max()
    max_unmasked_len = int(max_length_of_unmasked_tokens.item())
    seqlens_tokens_to_decode = decoder_mask.sum(dim=-1)
    max_length_of_decoded_tokens = seqlens_tokens_to_decode.max()
    max_decode_len = int(max_length_of_decoded_tokens.item())

    tokens_to_decode = sorted_tokens[:, :max_decode_len]
    tokens_to_decode_mask = decoder_mask[:, :max_decode_len].to(original_mask_dtype)

    if max_unmasked_len > 0:
        unmasked_tokens = sorted_tokens[:, -max_unmasked_len:]
        unmasked_tokens_mask = online_encoder_mask[:, -max_unmasked_len:].to(
            original_mask_dtype
        )
    else:
        unmasked_tokens = sorted_tokens[:, :0]
        unmasked_tokens_mask = online_encoder_mask[:, :0].to(original_mask_dtype)

    return SplitTokensByMaskResult(
        tokens_to_decode=tokens_to_decode,
        unmasked_tokens=unmasked_tokens,
        tokens_to_decode_mask=tokens_to_decode_mask,
        unmasked_tokens_mask=unmasked_tokens_mask,
        indices=indices,
        seqlens_tokens_to_decode=seqlens_tokens_to_decode,
        seqlens_unmasked_tokens=seqlens_unmasked_tokens,
        max_length_of_decoded_tokens=max_length_of_decoded_tokens,
        max_length_of_unmasked_tokens=max_length_of_unmasked_tokens,
    )


def combine_split_tokens(
    tokens_to_decode: Tensor,
    unmasked_tokens: Tensor,
    tokens_to_decode_mask: Tensor,
    unmasked_tokens_mask: Tensor,
    indices: Tensor,
) -> Tensor:
    """Recombine split decode/context token groups into original token order."""
    batch_size, num_tokens = indices.shape
    embedding_dim = tokens_to_decode.shape[-1]
    tokens = torch.zeros(
        (batch_size, num_tokens, embedding_dim),
        dtype=tokens_to_decode.dtype,
        device=tokens_to_decode.device,
    )

    num_unmasked_tokens = unmasked_tokens.shape[1]
    if num_unmasked_tokens > 0:
        tokens[:, -num_unmasked_tokens:] = (
            unmasked_tokens * unmasked_tokens_mask.unsqueeze(-1)
        )

    num_tokens_to_decode = tokens_to_decode.shape[1]
    if num_tokens_to_decode > 0:
        tokens[:, :num_tokens_to_decode] += (
            tokens_to_decode * tokens_to_decode_mask.unsqueeze(-1)
        )

    return tokens.scatter(1, indices[:, :, None].expand_as(tokens), tokens)
