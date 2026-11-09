"""Contrastive alignment of the location-token grid with grid-caption embeddings.

Each crop overlaps a few caption cells (see data/captions.py). The head averages the
location tokens over each (sample, cell), projects them to the text dim and runs a
symmetric InfoNCE against those cells' caption embeddings across the microbatch. Captions
nearly identical to the positive ("almost all open water") are not used as negatives.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from olmoearth_pretrain.config import Config
from olmoearth_pretrain.data.caption_embeddings import CaptionEmbeddings
from olmoearth_pretrain.data.captions import GRID, patch_caption_keys
from olmoearth_pretrain.data.constants import MISSING_VALUE


@dataclass
class TextAlignmentHeadConfig(Config):
    """Configuration for the text alignment head.

    Args:
        embeddings_dir: Output of caption_embeddings (embeddings.npy and meta.json).
        text_dim: Matryoshka dim the caption embeddings are truncated to.
        weight: Loss weight.
        init_temperature: Starting softmax temperature; learned from there.
        same_text_threshold: Captions at least this similar to the positive are not
            negatives.
    """

    embeddings_dir: str
    text_dim: int = 768
    weight: float = 1.0
    init_temperature: float = 0.07
    same_text_threshold: float = 0.95

    def build(self, embedding_dim: int) -> TextAlignmentHead:
        """Build the head for a location-token grid of width ``embedding_dim``."""
        return TextAlignmentHead(
            embeddings=CaptionEmbeddings(self.embeddings_dir, self.text_dim),
            embedding_dim=embedding_dim,
            weight=self.weight,
            init_temperature=self.init_temperature,
            same_text_threshold=self.same_text_threshold,
        )


class TextAlignmentHead(nn.Module):
    """Linear projection of pooled location tokens and a learned temperature."""

    def __init__(
        self,
        embeddings: CaptionEmbeddings,
        embedding_dim: int,
        weight: float = 1.0,
        init_temperature: float = 0.07,
        same_text_threshold: float = 0.95,
    ):
        """Initialize the head."""
        super().__init__()
        self.embeddings = embeddings
        self.proj = nn.Linear(embedding_dim, embeddings.dim)
        self.log_scale = nn.Parameter(torch.tensor(math.log(1 / init_temperature)))
        self.weight = weight
        self.same_text_threshold = same_text_threshold

    def forward(self, pooled: Tensor, text: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        """Weighted symmetric InfoNCE between pooled cells [N, D] and captions [N, dim]."""
        image = F.normalize(self.proj(pooled).float(), dim=-1)
        text = text.float()
        n = len(text)
        scale = self.log_scale.float().exp().clamp(max=100.0)
        logits = scale * image @ text.T
        same = (text @ text.T >= self.same_text_threshold) & ~torch.eye(
            n, dtype=torch.bool, device=text.device
        )
        logits = logits.masked_fill(same, float("-inf"))
        target = torch.arange(n, device=text.device)
        loss = (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target)) / 2
        metrics = {
            "text/loss": loss.detach(),
            "text/top1": (logits.argmax(dim=1) == target).float().mean().detach(),
            "text/temperature": (1 / scale).detach(),
            "text/cells": torch.tensor(float(n), device=text.device),
        }
        return self.weight * loss, metrics


def pool_caption_cells(
    location_tokens: Tensor, caption_cell: Tensor, embeddings: CaptionEmbeddings
) -> tuple[Tensor, Tensor]:
    """Mean location token and caption embedding of each (sample, cell) in the batch.

    Args:
        location_tokens: [B, h, w, D] location-token grid.
        caption_cell: [B, H, W, 1, 2] raster from the batch, H a multiple of h.
        embeddings: Caption embeddings to look the cells up in.

    Returns:
        [N, D] pooled tokens and [N, dim] unit caption embeddings, one row per cell
        that has a caption.
    """
    b, h, w, _ = location_tokens.shape
    keys = patch_caption_keys(caption_cell, caption_cell.shape[1] // h)
    assert keys.shape[1:3] == (h, w), (keys.shape, location_tokens.shape)
    valid = keys[..., 1] != MISSING_VALUE
    sample = torch.arange(b, device=keys.device)[:, None, None].expand(b, h, w)
    group = sample[valid] * GRID * GRID + keys[..., 1][valid].long()
    groups, inverse = group.unique(return_inverse=True)

    tokens = location_tokens[valid]
    pooled = tokens.new_zeros(len(groups), tokens.shape[-1])
    pooled.index_add_(0, inverse, tokens)
    pooled = pooled / torch.bincount(inverse, minlength=len(groups))[:, None]

    window = keys[..., 0][valid].new_zeros(len(groups))
    window[inverse] = keys[..., 0][valid]
    cell = (groups % (GRID * GRID)).to(window.dtype)
    text, has_text = embeddings.lookup(torch.stack([window, cell], dim=-1))
    has_text = has_text.to(pooled.device)
    return pooled[has_text], text.to(pooled.device)[has_text]


def text_alignment_loss(
    head: TextAlignmentHead, location_tokens: Tensor, caption_cell: Tensor, pixels: int
) -> tuple[Tensor, dict[str, Tensor]]:
    """The head's loss on a batch, or zero when it has fewer than two captioned cells.

    Args:
        head: The text alignment head.
        location_tokens: [B, N, D] flattened location-token grid, rows first.
        caption_cell: [B, H, W, 1, 2] raster from the batch.
        pixels: Pixels per location token along each side.
    """
    b, _, d = location_tokens.shape
    grid = location_tokens.reshape(
        b, caption_cell.shape[1] // pixels, caption_cell.shape[2] // pixels, d
    )
    pooled, text = pool_caption_cells(grid, caption_cell, head.embeddings)
    n = len(text)
    if n < 2:
        # Run the head anyway so data-parallel gradient reductions line up across ranks.
        pooled = location_tokens.new_zeros(2, d)
        text = F.normalize(text.new_ones(2, head.embeddings.dim), dim=-1)
        loss, _ = head(pooled, text)
        return loss * 0.0, {"text/cells": torch.tensor(float(n), device=text.device)}
    return head(pooled, text)
