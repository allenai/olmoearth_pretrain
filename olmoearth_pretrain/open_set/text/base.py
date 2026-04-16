"""Generic text encoder interface.

The decoder consumes both a sequence of per-token embeddings (for
cross-attention) and a single pooled embedding (for the final dot-product
classification head). Concrete encoders return both.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


@dataclass(frozen=True)
class TextEncoding:
    """Output of a text encoder for a batch of strings.

    Attributes:
        tokens: ``[N, L, D]`` — per-token embeddings, padded to a common length.
        pooled: ``[N, D]`` — single embedding per string (e.g. the SigLIP CLS-equivalent).
        attention_mask: ``[N, L]`` — 1 for real tokens, 0 for padding. Used by the
            decoder's cross-attention to ignore pad positions.
    """

    tokens: torch.Tensor
    pooled: torch.Tensor
    attention_mask: torch.Tensor


class TextEncoder(Protocol):
    """Frozen encoder mapping strings to embedding sequences + pooled vectors."""

    @property
    def name(self) -> str:
        """Stable identifier used as part of the on-disk cache key."""
        ...

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of the per-token / pooled embeddings."""
        ...

    def encode(self, texts: list[str]) -> TextEncoding:
        """Encode a list of strings. Implementations must run in inference mode."""
        ...
