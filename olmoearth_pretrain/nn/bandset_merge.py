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

logger = logging.getLogger(__name__)


class BandsetMerge(nn.Module):
    """Merge multiple bandset tokens into one via learned linear projection.

    Takes tokens of shape ``[..., num_bandsets, D]`` and produces ``[..., 1, D]``.
    Initialized to approximate mean pooling for stable early training.
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

        Args:
            tokens: ``[B, P_H, P_W, T, num_bandsets, D]``
            mask: ``[B, P_H, P_W, T, num_bandsets]``

        Returns:
            Tuple of (merged_tokens ``[..., 1, D]``, merged_mask ``[..., 1]``).
            Mask is taken from the first bandset (assumes uniform masking across bandsets).
        """
        tokens_cat = tokens.flatten(-2)  # [..., num_bandsets * D]
        merged = self.proj(tokens_cat)  # [..., D]
        merged = merged.unsqueeze(-2)  # [..., 1, D]
        # Uniform masking: all bandsets have same mask, take first
        merged_mask = mask[..., 0:1]  # [..., 1]
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
            mask: ``[B, P_H, P_W, T, 1]``

        Returns:
            Tuple of (unmerged_tokens ``[..., num_bandsets, D]``,
            unmerged_mask ``[..., num_bandsets]``).
        """
        squeezed = tokens.squeeze(-2)  # [..., D]
        expanded = self.proj(squeezed)  # [..., num_bandsets * D]
        *spatial, _ = expanded.shape
        unmerged = expanded.reshape(*spatial, self.num_bandsets, self.output_size)
        # Expand mask: [..., 1] -> [..., num_bandsets]
        unmerged_mask = mask.expand(*mask.shape[:-1], self.num_bandsets).contiguous()
        return unmerged, unmerged_mask
