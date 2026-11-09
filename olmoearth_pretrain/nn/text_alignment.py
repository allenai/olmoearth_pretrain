"""Contrastive alignment of the location-token grid with text.

Captions: each crop overlaps a few caption cells (see data/captions.py). The head
averages the location tokens over each (sample, cell), projects them to the text dim and
runs a symmetric InfoNCE against those cells' caption embeddings across the microbatch.

Labels (optional): pixels carry label ids (see data/text_labels.py). The tokens of each
(sample, label) are averaged the same way and classified among the microbatch's labels
plus random distractors from the vocabulary, through the same projection, so captions
and labels share one text space.

Either way, texts nearly identical to the positive ("almost all open water", "water"
and "open water") are not used as negatives.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from olmoearth_pretrain.config import Config
from olmoearth_pretrain.data.caption_embeddings import CaptionEmbeddings
from olmoearth_pretrain.data.captions import GRID, patch_caption_keys
from olmoearth_pretrain.data.constants import MISSING_VALUE
from olmoearth_pretrain.data.text_labels import CHANNELS


@dataclass
class TextAlignmentHeadConfig(Config):
    """Configuration for the text alignment head.

    Args:
        embeddings_dir: Output of caption_embeddings (embeddings.npy and meta.json).
        text_dim: Matryoshka dim the caption embeddings are truncated to.
        weight: Loss weight.
        init_temperature: Starting softmax temperature; learned from there.
        same_text_threshold: Texts at least this similar to the positive are not
            negatives.
        label_vocab_dir: Embedded label vocabulary (names.json, embeddings.npy) for the
            text_label targets; None trains on captions only.
        label_weight: Weight of the label loss.
        num_label_distractors: Vocabulary labels drawn as extra negatives per microbatch.
    """

    embeddings_dir: str
    text_dim: int = 768
    weight: float = 1.0
    init_temperature: float = 0.07
    same_text_threshold: float = 0.95
    label_vocab_dir: str | None = None
    label_weight: float = 0.1
    num_label_distractors: int = 1024

    def build(self, embedding_dim: int) -> TextAlignmentHead:
        """Build the head for a location-token grid of width ``embedding_dim``."""
        return TextAlignmentHead(
            embeddings=CaptionEmbeddings(self.embeddings_dir, self.text_dim),
            embedding_dim=embedding_dim,
            weight=self.weight,
            init_temperature=self.init_temperature,
            same_text_threshold=self.same_text_threshold,
            labels=(
                LabelEmbeddings(self.label_vocab_dir, self.text_dim)
                if self.label_vocab_dir is not None
                else None
            ),
            label_weight=self.label_weight,
            num_label_distractors=self.num_label_distractors,
        )


class LabelEmbeddings:
    """A label vocabulary's embeddings, held in memory: [labels, dim] unit vectors."""

    def __init__(self, path: str, dim: int | None = None):
        """Read ``embeddings.npy`` and ``meta.json``; ``dim`` truncates (Matryoshka)."""
        self.meta = json.load(open(os.path.join(path, "meta.json")))
        table = np.load(os.path.join(path, "embeddings.npy")).astype(np.float32)
        table = table[:, : dim or table.shape[1]]
        self.table = torch.from_numpy(
            table / np.linalg.norm(table, axis=1, keepdims=True)
        )
        self.dim = self.table.shape[1]

    def __len__(self) -> int:
        """The number of labels."""
        return len(self.table)

    def get(self, ids: Tensor) -> Tensor:
        """Embeddings of ``ids`` on their device."""
        if self.table.device != ids.device:
            self.table = self.table.to(ids.device)
        return self.table[ids]


class TextAlignmentHead(nn.Module):
    """Linear projection of pooled location tokens and a learned temperature."""

    def __init__(
        self,
        embeddings: CaptionEmbeddings,
        embedding_dim: int,
        weight: float = 1.0,
        init_temperature: float = 0.07,
        same_text_threshold: float = 0.95,
        labels: LabelEmbeddings | None = None,
        label_weight: float = 0.1,
        num_label_distractors: int = 1024,
    ):
        """Initialize the head."""
        super().__init__()
        self.embeddings = embeddings
        self.labels = labels
        if labels is not None and labels.dim != embeddings.dim:
            raise ValueError(f"label dim {labels.dim} != caption dim {embeddings.dim}")
        self.proj = nn.Linear(embedding_dim, embeddings.dim)
        self.log_scale = nn.Parameter(torch.tensor(math.log(1 / init_temperature)))
        self.weight = weight
        self.same_text_threshold = same_text_threshold
        self.label_weight = label_weight
        self.num_label_distractors = num_label_distractors

    def forward(
        self, pooled: Tensor, text: Tensor, targets: Tensor | None = None
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Weighted loss and metrics for pooled tokens [N, D].

        Without ``targets``, ``text`` [N, dim] holds each row's caption: symmetric
        InfoNCE. With ``targets`` [N], ``text`` [K, dim] holds candidate labels and
        each row is classified among them.
        """
        image = F.normalize(self.proj(pooled).float(), dim=-1)
        text = text.float()
        scale = self.log_scale.float().exp().clamp(max=100.0)
        if targets is not None:
            return self._label_loss(scale * image @ text.T, text, targets)
        n = len(text)
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

    def _label_loss(
        self, logits: Tensor, candidates: Tensor, targets: Tensor
    ) -> tuple[Tensor, dict[str, Tensor]]:
        same = (candidates[targets] @ candidates.T >= self.same_text_threshold) & (
            torch.arange(len(candidates), device=targets.device) != targets[:, None]
        )
        logits = logits.masked_fill(same, float("-inf"))
        loss = F.cross_entropy(logits, targets)
        metrics = {
            "text/label_loss": loss.detach(),
            "text/label_top1": (logits.argmax(dim=1) == targets)
            .float()
            .mean()
            .detach(),
            "text/labels": torch.tensor(float(len(targets)), device=targets.device),
        }
        return self.label_weight * loss, metrics


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


def pool_labels(location_tokens: Tensor, text_label: Tensor) -> tuple[Tensor, Tensor]:
    """Mean location token of each (sample, label) in the batch, and the label ids.

    Args:
        location_tokens: [B, h, w, D] location-token grid.
        text_label: [B, H, W, 1, CHANNELS] label ids, H a multiple of h. A token takes
            the labels at its centre pixel, in every channel.

    Returns:
        [G, D] pooled tokens and [G] label ids.
    """
    b, h, w, d = location_tokens.shape
    keys = patch_caption_keys(text_label, text_label.shape[1] // h)
    assert keys.shape[1:3] == (h, w), (keys.shape, location_tokens.shape)
    sample = torch.arange(b, device=keys.device)[:, None, None, None]
    sample = sample.expand(b, h, w, CHANNELS)
    token = torch.arange(b * h * w, device=keys.device).reshape(b, h, w, 1)
    token = token.expand(b, h, w, CHANNELS)
    valid = keys != MISSING_VALUE
    label = keys[valid].long()
    # Labels are below 2**24, so (sample, label) packs into one int64.
    groups, inverse = (sample[valid] * (1 << 24) + label).unique(return_inverse=True)
    tokens = location_tokens.reshape(b * h * w, d)[token[valid]]
    pooled = tokens.new_zeros(len(groups), d)
    pooled.index_add_(0, inverse, tokens)
    pooled = pooled / torch.bincount(inverse, minlength=len(groups))[:, None]
    return pooled, groups % (1 << 24)


def label_alignment_loss(
    head: TextAlignmentHead, location_tokens: Tensor, text_label: Tensor, pixels: int
) -> tuple[Tensor, dict[str, Tensor]]:
    """The head's label loss on a batch, or zero when no token has a label.

    Args:
        head: The text alignment head, built with a label vocabulary.
        location_tokens: [B, N, D] flattened location-token grid, rows first.
        text_label: [B, H, W, 1, CHANNELS] label ids from the batch.
        pixels: Pixels per location token along each side.
    """
    assert head.labels is not None
    b, _, d = location_tokens.shape
    grid = location_tokens.reshape(
        b, text_label.shape[1] // pixels, text_label.shape[2] // pixels, d
    )
    pooled, ids = pool_labels(grid, text_label)
    if len(ids) == 0:
        # Run the head anyway so data-parallel gradient reductions line up across ranks.
        candidates = head.labels.get(torch.arange(2, device=location_tokens.device))
        targets = torch.arange(2, device=location_tokens.device)
        loss, _ = head(location_tokens.new_zeros(2, d), candidates, targets)
        return loss * 0.0, {
            "text/labels": torch.tensor(0.0, device=location_tokens.device)
        }
    distractors = torch.randint(
        len(head.labels), (head.num_label_distractors,), device=ids.device
    )
    candidates = torch.cat([ids, distractors]).unique()
    targets = torch.searchsorted(candidates, ids)
    return head(pooled, head.labels.get(candidates), targets)
