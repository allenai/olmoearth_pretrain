"""Dedicated daily ERA5-Land time-series encoder.

This module implements a standalone patch-based transformer encoder for daily
ERA5-Land sequences. It is **separate** from `nn/flexi_vit.py` so that we can
keep the existing FlexiViT image encoder frozen (e.g. for objective C) while
training a new encoder over the daily climate sequence.

The encoder (adapted from rslearn's proven `PatchTransformerEncoder`):

* Tokenizes multi-day patches via Conv1D (kernel_size, stride), reducing the
  366-step annual sequence to ~50 tokens.
* Adds optional day-of-year / relative-position time features, encoded once
  per patch token (center date) and projected into the token space.
* Runs a pre-norm transformer stack.
* Pools the result with mean / cls / attention / gated / cls_mean_concat.

The forward pass exposes both the per-token sequence and a single pooled
embedding.  It also accepts an optional `prior_tokens` argument so that, for
objective C, the frozen FlexiViT image embedding at time ``T`` can be passed
in as additional prior context that the encoder attends to alongside the ERA5
sequence; the supervised objective A does not use this field.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from olmoearth_pretrain.config import Config
from olmoearth_pretrain.data.constants import (
    MAX_ERA5L_DAY_10_SEQUENCE_LENGTH,
    Modality,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pooling enum
# ---------------------------------------------------------------------------


class Era5Pooling(StrEnum):
    """Pooling strategy applied to the encoder output."""

    MEAN = "mean"
    CLS = "cls"
    ATTN = "attention"
    GATED = "gated"
    CLS_MEAN_CONCAT = "cls_mean_concat"


# ---------------------------------------------------------------------------
# Building blocks (ported from rslearn's transformer_encoder.py)
# ---------------------------------------------------------------------------


class _StochasticDepth(nn.Module):
    """Drop residual branches stochastically during training."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        """Apply stochastic depth."""
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class _TransformerBlock(nn.Module):
    """Pre-norm Transformer encoder block."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        mlp_ratio: int | float,
        dropout: float,
        attention_dropout: float,
        drop_path: float,
    ) -> None:
        super().__init__()
        hidden_dim = int(d_model * mlp_ratio)
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.drop_path1 = _StochasticDepth(drop_path)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )
        self.drop_path2 = _StochasticDepth(drop_path)

    def forward(self, x: Tensor, key_padding_mask: Tensor | None = None) -> Tensor:
        """Forward pass with optional key_padding_mask (True = ignore)."""
        attn_input = self.norm1(x)
        attn_out, _ = self.attn(
            attn_input,
            attn_input,
            attn_input,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + self.drop_path1(attn_out)
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class Era5DailyEncoderConfig(Config):
    """Configuration for `Era5DailyEncoder`.

    Args:
        embedding_size: Width of the transformer hidden state (d_model).
        depth: Number of transformer blocks.
        num_heads: Number of attention heads in each block.
        mlp_ratio: Ratio of MLP hidden dim to `embedding_size`.
        max_sequence_length: Maximum number of daily timesteps the encoder
            can ingest.
        modality_name: Modality the encoder consumes (defaults to
            ``era5l_day_10``); used to derive ``in_channels``.
        pooling: Pooling strategy for the global embedding.
        patch_kernel_size: Temporal kernel (in timesteps) of the Conv1D
            patch embedding.
        patch_stride: Stride (in timesteps) of the Conv1D patch embedding.
        dropout: Dropout probability in MLP and token dropout.
        attention_dropout: Dropout inside multi-head attention.
        drop_path_rate: Stochastic depth rate (linearly increased per layer).
        add_day_of_year_features: Add sin/cos day-of-year features to tokens.
        add_relative_position_features: Add sin/cos relative index features.
        use_mask_embed: When True, create a learned per-band mask embedding
            ``[1, 1, V]`` and accept an optional ``corruption_mask`` in
            ``forward()``.  Masked positions are replaced with the learned
            embedding instead of being zeroed.
        use_conv_stem: When True, replace the single Conv1D patch embedding
            with a two-layer stem (Conv1D + GroupNorm + GELU + 1x1 Conv1D)
            that has enough nonlinear capacity to gate out masked channels.
        extras: Reserved for objective C plumbing.
    """

    embedding_size: int = 384
    depth: int = 8
    num_heads: int = 6
    mlp_ratio: float = 4.0
    max_sequence_length: int = MAX_ERA5L_DAY_10_SEQUENCE_LENGTH
    modality_name: str = "era5l_day_10"
    pooling: str = Era5Pooling.MEAN.value
    patch_kernel_size: int = 14
    patch_stride: int = 7
    dropout: float = 0.1
    attention_dropout: float = 0.0
    drop_path_rate: float = 0.0
    add_day_of_year_features: bool = True
    add_relative_position_features: bool = False
    use_mask_embed: bool = False
    use_conv_stem: bool = False
    extras: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate the configuration."""
        if self.modality_name not in Modality.names():
            raise ValueError(f"Unknown modality: {self.modality_name}")
        if self.embedding_size % self.num_heads != 0:
            raise ValueError("embedding_size must be divisible by num_heads")
        if self.pooling not in {p.value for p in Era5Pooling}:
            raise ValueError(f"Unknown pooling: {self.pooling}")
        if self.max_sequence_length < 1:
            raise ValueError("max_sequence_length must be >= 1")
        if self.patch_kernel_size < 1:
            raise ValueError("patch_kernel_size must be >= 1")
        if self.patch_stride < 1:
            raise ValueError("patch_stride must be >= 1")
        if self.drop_path_rate < 0.0 or self.drop_path_rate >= 1.0:
            raise ValueError("drop_path_rate must be in [0, 1)")

    def build(self) -> Era5DailyEncoder:
        """Build the corresponding encoder."""
        self.validate()
        return Era5DailyEncoder(config=self)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


class Era5DailyEncoder(nn.Module):
    """Patch-based transformer encoder for daily ERA5-Land sequences.

    Accepts raw tensors ``[B, T, C]`` of daily values, per-step timestamps
    ``[B, T, 3]``, and a boolean padding mask ``[B, T]`` (True = padded).
    Returns a dict with ``pooled`` embedding and full token sequence.
    """

    def __init__(self, config: Era5DailyEncoderConfig) -> None:
        """Build encoder layers from *config*."""
        super().__init__()
        config.validate()
        self.config = config
        modality = Modality.get(config.modality_name)
        in_channels = modality.num_bands
        d_model = config.embedding_size

        self.embedding_size = d_model
        self.patch_kernel_size = config.patch_kernel_size
        self.patch_stride = config.patch_stride
        self.pooling = Era5Pooling(config.pooling)
        self.add_day_of_year_features = config.add_day_of_year_features
        self.add_relative_position_features = config.add_relative_position_features

        # Learned per-band mask embedding for reconstruction masking
        if config.use_mask_embed:
            self.mask_embed = nn.Parameter(torch.zeros(1, 1, in_channels))
            nn.init.trunc_normal_(self.mask_embed, std=0.02)
        else:
            self.mask_embed = None

        # Patch embedding: [B, C, T] -> [B, d_model, N]
        if config.use_conv_stem:
            mid = d_model // 2
            self.patch_embed: nn.Module = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    mid,
                    kernel_size=config.patch_kernel_size,
                    stride=config.patch_stride,
                ),
                nn.GroupNorm(1, mid),
                nn.GELU(),
                nn.Conv1d(mid, d_model, kernel_size=1),
            )
        else:
            self.patch_embed = nn.Conv1d(
                in_channels=in_channels,
                out_channels=d_model,
                kernel_size=config.patch_kernel_size,
                stride=config.patch_stride,
            )
        self.token_dropout = nn.Dropout(config.dropout)

        # Time features (day-of-year sin/cos, optional relative pos sin/cos)
        self.num_time_features = 0
        if config.add_day_of_year_features:
            self.num_time_features += 2
        if config.add_relative_position_features:
            self.num_time_features += 2
        self.time_embed: nn.Linear | None = (
            nn.Linear(self.num_time_features, d_model)
            if self.num_time_features > 0
            else None
        )

        # CLS token (only for cls / cls_mean_concat pooling)
        if self.pooling in {Era5Pooling.CLS, Era5Pooling.CLS_MEAN_CONCAT}:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        else:
            self.cls_token = None

        # Attention pooling query
        if self.pooling == Era5Pooling.ATTN:
            self.attn_query = nn.Parameter(torch.zeros(d_model))
            nn.init.normal_(self.attn_query, std=0.02)
        else:
            self.attn_query = None

        # Gated pooling projection
        self.gate_proj: nn.Linear | None = (
            nn.Linear(d_model, 1) if self.pooling == Era5Pooling.GATED else None
        )

        # Transformer blocks with linearly increasing stochastic depth
        drop_rates = torch.linspace(
            0.0, config.drop_path_rate, steps=config.depth
        ).tolist()
        self.blocks = nn.ModuleList(
            [
                _TransformerBlock(
                    d_model=d_model,
                    num_heads=config.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    dropout=config.dropout,
                    attention_dropout=config.attention_dropout,
                    drop_path=drop_rates[i],
                )
                for i in range(config.depth)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        era5: Tensor,
        timestamps: Tensor,
        ignore_mask: Tensor,
        prior_tokens: Tensor | None = None,
        corruption_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Encode a batch of daily ERA5 sequences.

        Args:
            era5: ``[B, T, C]`` daily ERA5 values (already normalized).
            timestamps: ``[B, T, 3]`` per-step ``[day-of-year, month0, year]``.
            ignore_mask: ``[B, T]`` boolean tensor. ``True`` positions are
                ignored by attention / pooling.
            prior_tokens: Optional ``[B, K, D]`` prior tokens prepended to
                the sequence (used by objective C). They are always
                considered valid (never ignored).
            corruption_mask: Optional ``[B, T, C]`` boolean tensor. ``True``
                positions are replaced with the learned ``mask_embed``
                before patchifying.  Only effective when ``use_mask_embed``
                is enabled.

        Returns:
            Dict with keys:

            * ``tokens``  : ``[B, N, D]`` per-token sequence after the
              transformer stack.
            * ``pooled``  : ``[B, D]`` global embedding.
            * ``ignore_mask`` : ``[B, N]`` boolean mask aligned with
              ``tokens`` (``True`` = ignored position).
            * ``num_prior_tokens`` / ``num_cls_tokens`` / ``num_data_tokens`` :
              ints stored as 0-d tensors so callers can slice the sequence.
        """
        if era5.ndim != 3:
            raise ValueError(f"Expected [B, T, C] era5, got shape {tuple(era5.shape)}")
        if timestamps.ndim != 3 or timestamps.shape[-1] != 3:
            raise ValueError(
                f"Expected [B, T, 3] timestamps, got shape {tuple(timestamps.shape)}"
            )
        if ignore_mask.ndim != 2:
            raise ValueError(
                f"Expected [B, T] ignore_mask, got shape {tuple(ignore_mask.shape)}"
            )
        b, t, _ = era5.shape
        if ignore_mask.shape != (b, t):
            raise ValueError(
                f"ignore_mask shape {tuple(ignore_mask.shape)} does not "
                f"match era5 (B, T) = ({b}, {t})"
            )

        # Replace corrupted positions with learned mask embedding
        if corruption_mask is not None and self.mask_embed is not None:
            era5 = torch.where(corruption_mask, self.mask_embed.expand_as(era5), era5)

        # valid_mask: True = real data (used internally)
        valid_mask = ~ignore_mask

        # Zero out padded timesteps before conv
        seq = era5 * valid_mask.unsqueeze(-1)

        # --- Patch embedding via Conv1D ---
        tokens, token_mask = self._patchify(seq, valid_mask)

        # --- Time features (per-token center-date calendar encoding) ---
        patch_time = self._build_patch_time_features(timestamps, valid_mask)
        if patch_time is not None:
            tokens = tokens + patch_time

        # --- Prepend prior_tokens and/or CLS ---
        num_prior = 0
        num_cls = 0
        if prior_tokens is not None:
            if prior_tokens.ndim != 3 or prior_tokens.shape[0] != b:
                raise ValueError(
                    f"Expected [B, K, D] prior_tokens with B={b}, got shape "
                    f"{tuple(prior_tokens.shape)}"
                )
            if prior_tokens.shape[-1] != self.embedding_size:
                raise ValueError(
                    f"prior_tokens last dim {prior_tokens.shape[-1]} does not "
                    f"match encoder embedding_size {self.embedding_size}"
                )
            num_prior = prior_tokens.shape[1]
            prior_valid = torch.ones(
                b, num_prior, dtype=torch.bool, device=tokens.device
            )
            tokens = torch.cat([prior_tokens, tokens], dim=1)
            token_mask = torch.cat([prior_valid, token_mask], dim=1)

        if self.cls_token is not None:
            num_cls = 1
            cls = self.cls_token.expand(b, -1, -1)
            cls_valid = torch.ones(b, 1, dtype=torch.bool, device=tokens.device)
            tokens = torch.cat([cls, tokens], dim=1)
            token_mask = torch.cat([cls_valid, token_mask], dim=1)

        # Guard against fully-padded sequences producing NaN in attention
        empty_rows = ~token_mask.any(dim=1)
        if empty_rows.any():
            token_mask = token_mask.clone()
            token_mask[empty_rows, 0] = True

        # --- Transformer blocks ---
        tokens = self.token_dropout(tokens)
        key_padding_mask = ~token_mask  # MHA convention: True = ignore
        for block in self.blocks:
            tokens = block(tokens, key_padding_mask=key_padding_mask)
        tokens = self.final_norm(tokens)

        # --- Pooling ---
        full_ignore_mask = ~token_mask
        num_data = tokens.shape[1] - num_prior - num_cls
        pooled = self._pool(
            tokens=tokens,
            token_mask=token_mask,
            num_prior=num_prior,
            num_cls=num_cls,
        )

        return {
            "tokens": tokens,
            "pooled": pooled,
            "ignore_mask": full_ignore_mask,
            "num_prior_tokens": torch.tensor(num_prior, device=tokens.device),
            "num_cls_tokens": torch.tensor(num_cls, device=tokens.device),
            "num_data_tokens": torch.tensor(num_data, device=tokens.device),
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _patchify(self, seq: Tensor, valid_mask: Tensor) -> tuple[Tensor, Tensor]:
        """Conv1D patch embedding + derive per-token validity mask.

        Args:
            seq: ``[B, T, C]`` (padded positions already zeroed).
            valid_mask: ``[B, T]`` bool, True = valid.

        Returns:
            tokens ``[B, N, D]`` and token_mask ``[B, N]`` (True = valid).
        """
        b, t, _ = seq.shape
        kernel = self.patch_kernel_size
        stride = self.patch_stride

        # Pad time dimension so conv covers all timesteps cleanly
        pad_t = self._patch_pad_amount(t)
        if pad_t > 0:
            seq = F.pad(seq, (0, 0, 0, pad_t), value=0.0)
            valid_mask = F.pad(valid_mask, (0, pad_t), value=False)

        # Conv1D expects [B, C, T]
        tokens = self.patch_embed(seq.transpose(1, 2)).transpose(1, 2)

        # Token is valid if at least one timestep in its receptive field is valid
        patch_mask = valid_mask.unfold(dimension=1, size=kernel, step=stride)
        token_mask = patch_mask.any(dim=-1)  # [B, N]

        return tokens, token_mask

    def _patch_pad_amount(self, t: int) -> int:
        """Timesteps to right-pad so the Conv1D patching covers ``t`` cleanly."""
        kernel = self.patch_kernel_size
        stride = self.patch_stride
        if t < kernel:
            return kernel - t
        remainder = (t - kernel) % stride
        return (stride - remainder) % stride

    def _build_patch_time_features(
        self, timestamps: Tensor, valid_mask: Tensor
    ) -> Tensor | None:
        """Encode one representative timestamp per patch token.

        Rather than averaging per-day sin/cos features, we derive a single
        representative date for each patch and encode that. For day-of-year we
        take the mask-aware mean day-of-year of the days in the patch's
        receptive field (its center date); for relative position we use the
        token's own index. Both are mapped through sin/cos and projected.

        Returns ``[B, N, D]`` additive contribution to token embeddings, or
        None if no time features are configured.
        """
        if self.time_embed is None:
            return None

        kernel = self.patch_kernel_size
        stride = self.patch_stride
        b, t = valid_mask.shape

        pad_t = self._patch_pad_amount(t)
        t_padded = t + pad_t
        num_tokens = (t_padded - kernel) // stride + 1

        feats: list[Tensor] = []

        if self.add_day_of_year_features:
            doy = timestamps[..., 0].float()  # [B, T]
            weights = valid_mask.float()
            if pad_t > 0:
                doy = F.pad(doy, (0, pad_t), value=0.0)
                weights = F.pad(weights, (0, pad_t), value=0.0)
            # Mask-aware mean day-of-year within each patch -> [B, N]
            patch_doy = doy.unfold(dimension=1, size=kernel, step=stride)
            patch_w = weights.unfold(dimension=1, size=kernel, step=stride)
            denom = patch_w.sum(dim=-1).clamp(min=1.0)
            center_doy = (patch_doy * patch_w).sum(dim=-1) / denom  # [B, N]
            angle = 2.0 * math.pi * (center_doy - 1.0) / 365.0
            feats.extend([torch.sin(angle), torch.cos(angle)])

        if self.add_relative_position_features:
            idx = torch.arange(
                num_tokens, device=valid_mask.device, dtype=torch.float32
            )
            rel = idx / max(num_tokens - 1, 1)
            rel_angle = 2.0 * math.pi * rel
            feats.extend(
                [
                    torch.sin(rel_angle).unsqueeze(0).expand(b, -1),
                    torch.cos(rel_angle).unsqueeze(0).expand(b, -1),
                ]
            )

        patch_time = torch.stack(feats, dim=-1)  # [B, N, F]
        return self.time_embed(patch_time)  # [B, N, D]

    @staticmethod
    def _masked_softmax(scores: Tensor, mask: Tensor) -> Tensor:
        """Mask-aware softmax (mask True = valid position)."""
        fill_value = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(~mask, fill_value)
        weights = torch.softmax(scores, dim=-1)
        weights = weights * mask.float()
        denom = weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        return weights / denom

    def _pool(
        self,
        tokens: Tensor,
        token_mask: Tensor,
        num_prior: int,
        num_cls: int,
    ) -> Tensor:
        """Pool the token sequence into a single per-sample embedding.

        Pooling operates over *data* tokens only (prior + CLS excluded for
        mean/attention/gated). For CLS, the CLS token itself is returned.
        """
        start = num_cls + num_prior
        data_tokens = tokens[:, start:, :]
        data_mask = token_mask[:, start:]  # True = valid

        if self.pooling == Era5Pooling.MEAN:
            weights = data_mask.float()
            denom = weights.sum(dim=1, keepdim=True).clamp(min=1.0)
            return (data_tokens * weights.unsqueeze(-1)).sum(dim=1) / denom

        if self.pooling == Era5Pooling.ATTN:
            assert self.attn_query is not None
            scale = self.embedding_size**-0.5
            scores = (data_tokens * self.attn_query).sum(dim=-1) * scale
            weights = self._masked_softmax(scores, data_mask)
            return (data_tokens * weights.unsqueeze(-1)).sum(dim=1)

        if self.pooling == Era5Pooling.GATED:
            assert self.gate_proj is not None
            gates = torch.sigmoid(self.gate_proj(data_tokens)).squeeze(-1)
            gates = gates * data_mask.float()
            denom = gates.sum(dim=1, keepdim=True).clamp(min=1e-6)
            return (data_tokens * gates.unsqueeze(-1)).sum(dim=1) / denom

        if self.pooling == Era5Pooling.CLS:
            return tokens[:, 0, :]

        if self.pooling == Era5Pooling.CLS_MEAN_CONCAT:
            cls_vec = tokens[:, 0, :]
            weights = data_mask.float()
            denom = weights.sum(dim=1, keepdim=True).clamp(min=1.0)
            mean_vec = (data_tokens * weights.unsqueeze(-1)).sum(dim=1) / denom
            return torch.cat([cls_vec, mean_vec], dim=1)

        raise ValueError(f"Unsupported pooling mode: {self.pooling}")

    def apply_compile(self) -> None:
        """Apply torch.compile to the model."""
        self.compile(dynamic=True)
