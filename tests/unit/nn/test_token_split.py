"""Tests for shared token split helpers."""

import torch

from olmoearth_pretrain.datatypes import MaskValue
from olmoearth_pretrain.nn.token_split import combine_split_tokens, split_tokens_by_mask


def test_split_tokens_by_mask_does_not_mutate_mask() -> None:
    """Splitting should not rewrite missing values in the caller's mask."""
    tokens = torch.tensor([[1, 2, 3]], dtype=torch.float32).unsqueeze(-1)
    mask = torch.tensor(
        [
            [
                MaskValue.MISSING.value,
                MaskValue.DECODER.value,
                MaskValue.ONLINE_ENCODER.value,
            ]
        ]
    )
    original_mask = mask.clone()

    split = split_tokens_by_mask(tokens, mask)

    assert torch.equal(mask, original_mask)
    assert split.tokens_to_decode.shape == (1, 1, 1)
    assert split.unmasked_tokens.shape == (1, 1, 1)


def test_combine_split_tokens_handles_no_decode_tokens() -> None:
    """Combining should work when every non-missing token is context."""
    tokens = torch.tensor([[1, 2, 3]], dtype=torch.float32).unsqueeze(-1)
    mask = torch.full((1, 3), MaskValue.ONLINE_ENCODER.value)

    split = split_tokens_by_mask(tokens, mask)
    combined = combine_split_tokens(
        tokens_to_decode=split.tokens_to_decode,
        unmasked_tokens=split.unmasked_tokens,
        tokens_to_decode_mask=split.tokens_to_decode_mask,
        unmasked_tokens_mask=split.unmasked_tokens_mask,
        indices=split.indices,
    )

    assert split.tokens_to_decode.shape == (1, 0, 1)
    assert torch.equal(combined, tokens)


def test_combine_split_tokens_handles_no_unmasked_tokens() -> None:
    """Combining should work when every non-missing token is decoded."""
    tokens = torch.tensor([[1, 2, 3]], dtype=torch.float32).unsqueeze(-1)
    mask = torch.tensor(
        [[MaskValue.DECODER.value, MaskValue.MISSING.value, MaskValue.DECODER.value]]
    )

    split = split_tokens_by_mask(tokens, mask)
    combined = combine_split_tokens(
        tokens_to_decode=split.tokens_to_decode,
        unmasked_tokens=split.unmasked_tokens,
        tokens_to_decode_mask=split.tokens_to_decode_mask,
        unmasked_tokens_mask=split.unmasked_tokens_mask,
        indices=split.indices,
    )

    expected = tokens.clone()
    expected[mask == MaskValue.MISSING.value] = 0

    assert split.unmasked_tokens.shape == (1, 0, 1)
    assert torch.equal(combined, expected)
