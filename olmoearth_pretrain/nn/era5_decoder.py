"""Lightweight time-query decoder for ERA5 reconstruction (objective B).

Given N latent tokens from the :class:`Era5DailyEncoder`, this module
produces a T-step reconstruction of the raw ERA5 input by attending from
learned per-timestep queries back into the encoder output.

Architecture::

    Encoder tokens  H ∈ R^{B × N × D}
                       ↑ cross-attention
    Time queries    Q ∈ R^{B × T × D}   (learned positional + day-of-year)
                       ↓
    Decoded tokens  R ∈ R^{B × T × D}
                       ↓ MLP
    Reconstruction  X̂ ∈ R^{B × T × V}

The decoder is intentionally simple so the encoder bears the
representational burden.  If it underfits, increase ``depth`` or
``mlp_ratio``.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor, nn

from olmoearth_pretrain.config import Config
from olmoearth_pretrain.data.constants import (
    MAX_ERA5L_DAY_10_SEQUENCE_LENGTH,
    Modality,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cross-attention decoder block
# ---------------------------------------------------------------------------


class _CrossAttentionBlock(nn.Module):
    """Pre-norm cross-attention block followed by a small FFN."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_dim = int(d_model * mlp_ratio)
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        queries: Tensor,
        memory: Tensor,
        memory_key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Cross-attend from *queries* into *memory*.

        Args:
            queries: ``[B, T, D]``
            memory: ``[B, N, D]`` encoder tokens.
            memory_key_padding_mask: ``[B, N]`` bool, True = ignore.
        """
        q = self.norm_q(queries)
        kv = self.norm_kv(memory)
        attn_out, _ = self.cross_attn(
            q,
            kv,
            kv,
            key_padding_mask=memory_key_padding_mask,
            need_weights=False,
        )
        queries = queries + attn_out
        queries = queries + self.ffn(self.norm_ffn(queries))
        return queries


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class Era5TimeQueryDecoderConfig(Config):
    """Configuration for :class:`Era5TimeQueryDecoder`.

    Args:
        embedding_size: Must match the encoder's ``embedding_size``.
        depth: Number of cross-attention blocks.
        num_heads: Attention heads per block.
        mlp_ratio: FFN hidden-dim multiplier.
        max_sequence_length: Max T for the learned positional table.
        num_output_channels: V — number of reconstructed bands.
        add_day_of_year_features: Inject sin/cos day-of-year into queries.
        dropout: Dropout in FFN and attention.
    """

    embedding_size: int = 384
    depth: int = 2
    num_heads: int = 6
    mlp_ratio: float = 4.0
    max_sequence_length: int = MAX_ERA5L_DAY_10_SEQUENCE_LENGTH
    num_output_channels: int = Modality.ERA5L_DAY_10.num_bands
    add_day_of_year_features: bool = True
    dropout: float = 0.0
    extras: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate the decoder configuration."""
        if self.embedding_size <= 0:
            raise ValueError("embedding_size must be > 0")
        if self.depth < 1:
            raise ValueError("depth must be >= 1")
        if self.embedding_size % self.num_heads != 0:
            raise ValueError("embedding_size must be divisible by num_heads")

    def build(self) -> Era5TimeQueryDecoder:
        """Build the decoder module from this config."""
        self.validate()
        return Era5TimeQueryDecoder(config=self)


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------


class Era5TimeQueryDecoder(nn.Module):
    """Time-query cross-attention decoder for ERA5 reconstruction.

    Produces ``X̂ [B, T, V]`` from encoder tokens ``[B, N, D]``.
    """

    def __init__(self, config: Era5TimeQueryDecoderConfig) -> None:
        """Initialize the decoder from its config."""
        super().__init__()
        config.validate()
        self.config = config
        d = config.embedding_size
        T = config.max_sequence_length
        V = config.num_output_channels

        # Learned per-timestep position embedding
        self.pos_embed = nn.Embedding(T, d)

        # Optional day-of-year sin/cos -> d projection
        self.add_day_of_year = config.add_day_of_year_features
        self.doy_proj: nn.Linear | None = (
            nn.Linear(2, d) if self.add_day_of_year else None
        )

        # Cross-attention blocks
        self.blocks = nn.ModuleList(
            [
                _CrossAttentionBlock(
                    d_model=d,
                    num_heads=config.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    dropout=config.dropout,
                )
                for _ in range(config.depth)
            ]
        )

        # MLP head: LN -> D -> 2D -> GELU -> 2D -> V
        self.head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, 2 * d),
            nn.GELU(),
            nn.Linear(2 * d, V),
        )

    def forward(
        self,
        tokens: Tensor,
        token_ignore_mask: Tensor,
        timestamps: Tensor,
        seq_len: int | None = None,
    ) -> Tensor:
        """Decode encoder tokens into a per-timestep reconstruction.

        Args:
            tokens: ``[B, N_total, D]`` full encoder output (may include
                CLS / prior tokens — those are attended to as well).
            token_ignore_mask: ``[B, N_total]`` bool, True = ignored.
            timestamps: ``[B, T, 3]``  ``[day-of-year, month0, year]``.
            seq_len: Override for the output sequence length (defaults to
                ``timestamps.shape[1]``).

        Returns:
            ``X̂ [B, T, V]`` reconstructed ERA5.
        """
        b = tokens.shape[0]
        T = seq_len or timestamps.shape[1]

        # Build time queries [B, T, D]
        pos_ids = torch.arange(T, device=tokens.device)
        queries = self.pos_embed(pos_ids).unsqueeze(0).expand(b, -1, -1)

        if self.doy_proj is not None:
            doy = timestamps[:, :T, 0].float()
            angle = 2.0 * math.pi * (doy - 1.0) / 365.0
            doy_feat = torch.stack([torch.sin(angle), torch.cos(angle)], dim=-1)
            queries = queries + self.doy_proj(doy_feat)

        # Cross-attend into encoder tokens
        for block in self.blocks:
            queries = block(queries, tokens, memory_key_padding_mask=token_ignore_mask)

        # Project to output channels
        return self.head(queries)

    def apply_compile(self) -> None:
        """Apply torch.compile for parity with the encoder."""
        self.compile(dynamic=True)
