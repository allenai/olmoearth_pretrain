"""Bandset merge/unmerge modules for cross-channel learning.

When using multi-bandset tokenization (e.g. 3 bandsets for Sentinel-2), these modules
allow merging bandset tokens into one before the encoder transformer and unmerging
back to per-bandset tokens in the decoder. This gives cross-spectral learning via the
merge projection while keeping the transformer sequence length short.
"""

import logging

import torch
import torch.nn as nn
from torch import Tensor

from olmoearth_pretrain.datatypes import MaskValue

logger = logging.getLogger(__name__)


class BandsetMerge(nn.Module):
    """Merge multiple bandset tokens into one via learned linear projection.

    Takes tokens of shape ``[..., num_bandsets, D]`` and produces ``[..., 1, D]``.
    Initialized to approximate mean pooling for stable early training.

    Non-ENCODER bandsets are zeroed out before projection to prevent information
    leakage (e.g. when some bandsets are assigned DECODER role by the masking
    strategy).  The result is rescaled so that the effective magnitude matches
    what the projection would produce with all bandsets active.
    """

    def __init__(self, embedding_size: int, num_bandsets: int) -> None:
        """Initialize bandset merge module."""
        super().__init__()
        self.num_bandsets = num_bandsets
        self.embedding_size = embedding_size
        self.proj = nn.Linear(num_bandsets * embedding_size, embedding_size)
        self._init_as_mean_pool()

    def _init_as_mean_pool(self) -> None:
        """Initialize weights to approximate mean pooling for stable early training."""
        with torch.no_grad():
            self.proj.weight.zero_()
            self.proj.bias.zero_()
            for i in range(self.num_bandsets):
                start = i * self.embedding_size
                end = start + self.embedding_size
                self.proj.weight[:, start:end] = (
                    torch.eye(self.embedding_size) / self.num_bandsets
                )

    def forward(self, tokens: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """Merge bandset tokens into one.

        Non-ENCODER bandsets are zeroed out before the learned projection.
        Output is rescaled by ``num_bandsets / num_active`` so that magnitude
        stays consistent regardless of how many bandsets are active.

        Args:
            tokens: ``[B, P_H, P_W, T, num_bandsets, D]``
            mask: ``[B, P_H, P_W, T, num_bandsets]``

        Returns:
            Tuple of (merged_tokens ``[..., 1, D]``, merged_mask ``[..., 1]``).
        """
        # Zero out non-ENCODER bandsets to prevent information leakage
        encoder_mask = mask == MaskValue.ONLINE_ENCODER.value  # [..., num_bandsets]
        tokens = tokens * encoder_mask.unsqueeze(-1).to(tokens.dtype)

        # Normalize by number of active bandsets for consistent magnitude
        active_count = (
            encoder_mask.sum(dim=-1, keepdim=True).clamp(min=1).to(tokens.dtype)
        )
        scale = self.num_bandsets / active_count  # [..., 1]

        tokens_cat = tokens.flatten(-2)  # [..., num_bandsets * D]
        # Cast to match proj weight dtype (FSDP may store weights in bfloat16
        # while composite_encodings produces float32 intermediates)
        merged = self.proj(tokens_cat.to(self.proj.weight.dtype))  # [..., D]
        merged = merged * scale  # rescale
        merged = merged.unsqueeze(-2)  # [..., 1, D]

        # Merged mask: ONLINE_ENCODER if any bandset is ENCODER, else take min
        has_encoder = encoder_mask.any(dim=-1, keepdim=True)  # [..., 1]
        merged_mask = torch.where(
            has_encoder,
            torch.full_like(mask[..., 0:1], MaskValue.ONLINE_ENCODER.value),
            mask.min(dim=-1, keepdim=True).values,
        )
        return merged, merged_mask


class BandsetUnmerge(nn.Module):
    """Expand one merged token back to multiple bandset tokens.

    Takes tokens of shape ``[..., 1, D]`` and produces ``[..., num_bandsets, D]``.
    """

    def __init__(self, input_size: int, num_bandsets: int) -> None:
        """Initialize bandset unmerge module."""
        super().__init__()
        self.num_bandsets = num_bandsets
        self.output_size = input_size
        self.proj = nn.Linear(input_size, num_bandsets * input_size)

    def forward(self, tokens: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """Unmerge one token back to multiple bandset tokens.

        Args:
            tokens: ``[B, P_H, P_W, T, 1, D]``
            mask: Either ``[..., 1]`` (merged mask, will be expanded) or
                ``[..., num_bandsets]`` (original pre-merge mask, used as-is).

        Returns:
            Tuple of (unmerged_tokens ``[..., num_bandsets, D]``,
            unmerged_mask ``[..., num_bandsets]``).
        """
        squeezed = tokens.squeeze(-2)  # [..., D]
        # Cast to match proj weight dtype (see BandsetMerge.forward for rationale)
        expanded = self.proj(
            squeezed.to(self.proj.weight.dtype)
        )  # [..., num_bandsets * D]
        *spatial, _ = expanded.shape
        unmerged = expanded.reshape(*spatial, self.num_bandsets, self.output_size)
        # If mask already has num_bandsets, use as-is; otherwise expand
        if mask.shape[-1] == self.num_bandsets:
            unmerged_mask = mask
        else:
            unmerged_mask = mask.expand(
                *mask.shape[:-1], self.num_bandsets
            ).contiguous()
        return unmerged, unmerged_mask
