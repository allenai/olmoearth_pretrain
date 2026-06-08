"""Dedicated daily ERA5-Land time-series encoder.

This module implements a standalone transformer encoder for daily ERA5-Land
sequences. It is **separate** from `nn/flexi_vit.py` so that we can keep the
existing FlexiViT image encoder frozen (e.g. for objective C) while training a
new encoder over the daily climate sequence (up to 366 days).

The encoder:

* Tokenizes each daily timestep with a per-band-set linear projection (mirrors
  the non-spatial branch of `MultiModalPatchEmbeddings`).
* Adds composite additive encodings: per-band-set channel embedding,
  sin-cos position over the day index, sin-cos position over day-of-year,
  and a 12-bin month embedding (reuses `get_month_encoding_table`).
* Runs a transformer stack reusing `nn.attention.Block`.
* Pools the result with mean / cls / attention pooling.

The forward pass exposes both the per-token sequence and a single pooled
embedding. It also accepts an optional `prior_tokens` argument so that, for
objective C, the frozen FlexiViT image embedding at time ``T`` can be passed
in as additional prior context that the encoder attends to alongside the
ERA5 sequence; the supervised objective A does not use this field.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import torch
from einops import repeat
from torch import Tensor, nn

from olmoearth_pretrain.config import Config
from olmoearth_pretrain.data.constants import (
    MAX_ERA5L_DAY_10_SEQUENCE_LENGTH,
    Modality,
    ModalitySpec,
)
from olmoearth_pretrain.nn.attention import Block
from olmoearth_pretrain.nn.encodings import (
    get_1d_sincos_pos_encoding,
    get_month_encoding_table,
)

logger = logging.getLogger(__name__)


class Era5Pooling(StrEnum):
    """Pooling strategy applied to the encoder output."""

    MEAN = "mean"
    CLS = "cls"
    ATTN = "attn"


@dataclass
class Era5DailyEncoderConfig(Config):
    """Configuration for `Era5DailyEncoder`.

    Args:
        embedding_size: Width of the transformer hidden state.
        depth: Number of transformer blocks.
        num_heads: Number of attention heads in each block.
        mlp_ratio: Ratio of MLP hidden dim to `embedding_size`.
        max_sequence_length: Maximum number of daily timesteps the encoder
            can ingest (drives the size of the position / day-of-year tables).
        modality_name: Modality the encoder consumes (defaults to
            `era5l_day_10`).
        pooling: Pooling strategy for the global embedding.
        drop_path: Stochastic-depth rate.
        attn_drop: Dropout rate inside attention.
        proj_drop: Dropout rate on transformer projection outputs.
        qkv_bias: Whether QKV projections include a bias.
        qk_norm: Whether to apply norm to Q,K before attention.
        use_flash_attn: Use flash attention kernels if available.
        learnable_channel_embeddings: Use a learnable per-band-set channel
            embedding instead of zeros.
    """

    embedding_size: int = 384
    depth: int = 8
    num_heads: int = 6
    mlp_ratio: float = 4.0
    max_sequence_length: int = MAX_ERA5L_DAY_10_SEQUENCE_LENGTH
    modality_name: str = "era5l_day_10"
    pooling: str = Era5Pooling.MEAN.value
    drop_path: float = 0.1
    attn_drop: float = 0.0
    proj_drop: float = 0.0
    qkv_bias: bool = True
    qk_norm: bool = False
    use_flash_attn: bool = False
    learnable_channel_embeddings: bool = True
    init_values: float | None = None
    # Reserved for future / objective C plumbing. No effect on the architecture
    # itself; the forward signature already takes a `prior_tokens` argument.
    extras: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate the configuration."""
        if self.modality_name not in Modality.names():
            raise ValueError(f"Unknown modality: {self.modality_name}")
        if self.embedding_size % 4 != 0:
            raise ValueError(
                "embedding_size must be divisible by 4 (the encoder splits it "
                "across channel / time / day-of-year / month embeddings)."
            )
        if self.pooling not in {p.value for p in Era5Pooling}:
            raise ValueError(f"Unknown pooling: {self.pooling}")
        if self.max_sequence_length < 1:
            raise ValueError("max_sequence_length must be >= 1")

    def build(self) -> Era5DailyEncoder:
        """Build the corresponding encoder."""
        self.validate()
        return Era5DailyEncoder(config=self)


class _Era5PatchEmbed(nn.Module):
    """Per-band-set linear patch embedding for the time-only ERA5 modality."""

    def __init__(self, modality: ModalitySpec, embedding_size: int) -> None:
        super().__init__()
        self.modality = modality
        self.embedding_size = embedding_size
        if modality.is_spatial:
            raise ValueError(
                f"Era5DailyEncoder expects a non-spatial modality, got "
                f"{modality.name} (is_spatial=True)."
            )
        self.per_bandset_embeddings = nn.ModuleList(
            [
                nn.Linear(len(band_set.bands), embedding_size)
                for band_set in modality.band_sets
            ]
        )
        self._bandset_band_indices: list[list[int]] = modality.bandsets_as_indices()

    def forward(self, x: Tensor) -> Tensor:
        """Patchify a time-only ERA5 sample.

        Args:
            x: Tensor of shape ``[B, T, C]`` where ``C`` is the total number of
                bands across band-sets.

        Returns:
            Tensor of shape ``[B, T, num_band_sets, D]``.
        """
        if x.ndim != 3:
            raise ValueError(f"Expected [B, T, C] tensor, got shape {tuple(x.shape)}")
        per_bandset = []
        for embed, idxs in zip(self.per_bandset_embeddings, self._bandset_band_indices):
            per_bandset.append(embed(x[..., idxs]))
        # [B, T, num_band_sets, D]
        return torch.stack(per_bandset, dim=-2)


class _Era5CompositeEncodings(nn.Module):
    """Additive composite encodings for the daily ERA5 encoder.

    The embedding budget is split into four equal quarters following the
    convention used in `CompositeEncodings` (channel / time / day-of-year /
    month). Spatial encodings are omitted since the modality is non-spatial.
    """

    def __init__(
        self,
        modality: ModalitySpec,
        embedding_size: int,
        max_sequence_length: int,
        learnable_channel_embeddings: bool = True,
    ) -> None:
        super().__init__()
        self.modality = modality
        self.embedding_size = embedding_size
        self.max_sequence_length = max_sequence_length
        n = embedding_size // 4
        self.embedding_dim_per_embedding_type = n
        # Sin-cos position encoding over the temporal index.
        self.pos_embed = nn.Parameter(
            get_1d_sincos_pos_encoding(torch.arange(max_sequence_length), n),
            requires_grad=False,
        )
        # Sin-cos day-of-year (1..366) -> table of length 367 so that
        # `dayofyear` from datetime can be used directly (1-indexed).
        self.day_of_year_embed = nn.Parameter(
            get_1d_sincos_pos_encoding(torch.arange(367), n),
            requires_grad=False,
        )
        # Month encoding (0..11).
        month_tab = get_month_encoding_table(n)
        self.month_embed = nn.Embedding.from_pretrained(month_tab, freeze=True)
        # Per-band-set channel embedding (learnable by default, zeros otherwise).
        num_bandsets = modality.num_band_sets
        if learnable_channel_embeddings:
            channel_embed = nn.Parameter(torch.zeros(num_bandsets, n))
        else:
            channel_embed = nn.Parameter(
                torch.zeros(num_bandsets, n), requires_grad=False
            )
        self.channel_embed = channel_embed

    def forward(self, tokens: Tensor, timestamps: Tensor) -> Tensor:
        """Apply additive encodings.

        Args:
            tokens: ``[B, T, num_band_sets, D]`` per-band-set tokens.
            timestamps: ``[B, T, 3]`` with columns ``[day, month0, year]``
                (months are zero-indexed, matching the convention used by
                `get_timestamps` in `evals/datasets/rslearn_dataset.py`).

        Returns:
            Tokens with composite encodings added (same shape as input).
        """
        if tokens.ndim != 4:
            raise ValueError(
                f"Expected [B, T, num_band_sets, D] tokens, got shape {tuple(tokens.shape)}"
            )
        b, t, b_s, d = tokens.shape
        if t > self.max_sequence_length:
            raise ValueError(
                f"Sequence length {t} exceeds encoder max_sequence_length "
                f"{self.max_sequence_length}"
            )
        n = self.embedding_dim_per_embedding_type
        device = tokens.device
        embed = torch.zeros_like(tokens)
        # Channel embedding (broadcasts over batch + time).
        channel = repeat(self.channel_embed, "b_s d -> b t b_s d", b=b, t=t).to(device)
        embed[..., :n] = embed[..., :n] + channel
        # Temporal position embedding (broadcasts over batch + band-sets).
        time_embed = repeat(self.pos_embed[:t], "t d -> b t b_s d", b=b, b_s=b_s).to(
            device
        )
        embed[..., n : 2 * n] = embed[..., n : 2 * n] + time_embed
        # Day-of-year embedding from timestamps[:, :, 0] (1..366).
        # We clamp to [0, 366] defensively so that out-of-range indices
        # (e.g. padded zeros from collation) don't blow up the gather.
        day_idx = timestamps[..., 0].clamp(min=0, max=366).long()
        doy_embed = self.day_of_year_embed.to(device)[day_idx]  # [B, T, n]
        doy_embed = repeat(doy_embed, "b t d -> b t b_s d", b_s=b_s)
        embed[..., 2 * n : 3 * n] = embed[..., 2 * n : 3 * n] + doy_embed
        # Month embedding from timestamps[:, :, 1] (0..11).
        month_idx = timestamps[..., 1].clamp(min=0, max=11).long()
        month_embed = self.month_embed(month_idx)  # [B, T, n]
        month_embed = repeat(month_embed, "b t d -> b t b_s d", b_s=b_s)
        embed[..., 3 * n : 4 * n] = embed[..., 3 * n : 4 * n] + month_embed
        return tokens + embed


class Era5DailyEncoder(nn.Module):
    """Dedicated transformer encoder over daily ERA5-Land sequences.

    See the module docstring for design context. The encoder takes a daily
    sequence ``[B, T, C]`` (T <= `config.max_sequence_length`), per-step
    timestamps ``[B, T, 3]`` (day-of-year, month-0-indexed, year), and a
    boolean padding mask ``[B, T]`` where ``True`` marks padding positions
    to ignore. It returns the per-token sequence, a pooled embedding, and
    the per-token (post-prior-prepend) validity mask for downstream
    objectives that need it (e.g. masked reconstruction in objective B).
    """

    def __init__(self, config: Era5DailyEncoderConfig) -> None:
        """Build encoder layers from *config*."""
        super().__init__()
        config.validate()
        self.config = config
        self.modality = Modality.get(config.modality_name)
        self.embedding_size = config.embedding_size
        self.max_sequence_length = config.max_sequence_length
        self.pooling = Era5Pooling(config.pooling)
        self.num_bandsets = self.modality.num_band_sets

        self.patch_embed = _Era5PatchEmbed(
            modality=self.modality, embedding_size=config.embedding_size
        )
        self.encodings = _Era5CompositeEncodings(
            modality=self.modality,
            embedding_size=config.embedding_size,
            max_sequence_length=config.max_sequence_length,
            learnable_channel_embeddings=config.learnable_channel_embeddings,
        )

        if self.pooling == Era5Pooling.CLS:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embedding_size))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        else:
            self.cls_token = None

        # Stochastic depth schedule across blocks (linear).
        dpr = [x.item() for x in torch.linspace(0.0, config.drop_path, config.depth)]
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=config.embedding_size,
                    num_heads=config.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    qkv_bias=config.qkv_bias,
                    qk_norm=config.qk_norm,
                    drop=config.proj_drop,
                    attn_drop=config.attn_drop,
                    drop_path=dpr[i],
                    init_values=config.init_values,
                    use_flash_attn=config.use_flash_attn,
                )
                for i in range(config.depth)
            ]
        )
        self.norm = nn.LayerNorm(config.embedding_size)

        if self.pooling == Era5Pooling.ATTN:
            # Single learnable attention-pool query.
            self.attn_query = nn.Parameter(torch.zeros(1, 1, config.embedding_size))
            nn.init.trunc_normal_(self.attn_query, std=0.02)
            self.attn_pool = nn.MultiheadAttention(
                embed_dim=config.embedding_size,
                num_heads=config.num_heads,
                batch_first=True,
            )
        else:
            self.attn_query = None
            self.attn_pool = None

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0.0)

    def forward(
        self,
        era5: Tensor,
        timestamps: Tensor,
        padding_mask: Tensor,
        prior_tokens: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Encode a batch of daily ERA5 sequences.

        Args:
            era5: ``[B, T, C]`` daily ERA5 values (already normalized).
            timestamps: ``[B, T, 3]`` per-step ``[day-of-year, month0, year]``.
            padding_mask: ``[B, T]`` boolean tensor. ``True`` positions are
                padded and ignored by attention / pooling.
            prior_tokens: Optional ``[B, K, D]`` prior tokens prepended to
                the sequence (used by objective C). They are always
                considered valid (not padded) and receive zero positional
                encodings.

        Returns:
            Dict with keys:

            * ``tokens``  : ``[B, N, D]`` per-token sequence after the
              transformer stack. ``N = K + T * num_band_sets`` (or
              ``N = 1 + K + T * num_band_sets`` when pooling is ``cls``).
            * ``pooled``  : ``[B, D]`` global embedding.
            * ``padding_mask`` : ``[B, N]`` boolean mask aligned with
              ``tokens`` (``True`` = padded position).
            * ``num_prior_tokens`` / ``num_cls_tokens`` / ``num_data_tokens`` :
              ints stored as 0-d tensors so callers can slice the sequence.
        """
        if era5.ndim != 3:
            raise ValueError(f"Expected [B, T, C] era5, got shape {tuple(era5.shape)}")
        if timestamps.ndim != 3 or timestamps.shape[-1] != 3:
            raise ValueError(
                f"Expected [B, T, 3] timestamps, got shape {tuple(timestamps.shape)}"
            )
        if padding_mask.ndim != 2:
            raise ValueError(
                f"Expected [B, T] padding_mask, got shape {tuple(padding_mask.shape)}"
            )
        b, t, _ = era5.shape
        if padding_mask.shape != (b, t):
            raise ValueError(
                f"padding_mask shape {tuple(padding_mask.shape)} does not match "
                f"era5 (B, T) = ({b}, {t})"
            )

        # [B, T, num_band_sets, D] -> add encodings -> flatten over band-sets.
        tokens = self.patch_embed(era5)
        tokens = self.encodings(tokens, timestamps)
        b_s = tokens.shape[-2]
        tokens = tokens.reshape(b, t * b_s, self.embedding_size)
        # Expand the [B, T] padding mask to [B, T*num_band_sets] (all band-sets
        # at a padded timestep are themselves padded).
        data_mask = padding_mask.unsqueeze(-1).expand(b, t, b_s).reshape(b, t * b_s)

        num_prior = 0
        num_cls = 0
        prepended = []
        prepended_masks = []
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
            prepended.append(prior_tokens)
            prepended_masks.append(
                torch.zeros(b, num_prior, dtype=torch.bool, device=tokens.device)
            )
        if self.cls_token is not None:
            num_cls = 1
            prepended.append(self.cls_token.expand(b, -1, -1).to(tokens.device))
            prepended_masks.append(
                torch.zeros(b, num_cls, dtype=torch.bool, device=tokens.device)
            )
        if prepended:
            tokens = torch.cat([*prepended, tokens], dim=1)
            full_mask = torch.cat([*prepended_masks, data_mask], dim=1)
        else:
            full_mask = data_mask

        # Build a [B, N] attention mask: True = participates in attention.
        attn_mask = (~full_mask).to(tokens.dtype)
        for block in self.blocks:
            tokens = block(tokens, attn_mask=attn_mask.bool())
        tokens = self.norm(tokens)

        pooled = self._pool(
            tokens=tokens,
            full_mask=full_mask,
            num_prior=num_prior,
            num_cls=num_cls,
        )

        return {
            "tokens": tokens,
            "pooled": pooled,
            "padding_mask": full_mask,
            "num_prior_tokens": torch.tensor(num_prior, device=tokens.device),
            "num_cls_tokens": torch.tensor(num_cls, device=tokens.device),
            "num_data_tokens": torch.tensor(t * b_s, device=tokens.device),
        }

    def _pool(
        self,
        tokens: Tensor,
        full_mask: Tensor,
        num_prior: int,
        num_cls: int,
    ) -> Tensor:
        """Pool the token sequence into a single per-sample embedding.

        The pooling always operates over the *data* tokens (T*num_band_sets) —
        prior tokens and CLS are excluded for `mean` / `attn` so that the
        pooled vector summarizes the daily ERA5 sequence regardless of prior
        context. For `cls`, the CLS token itself is returned (it has already
        attended to prior + data via the transformer stack).
        """
        if self.pooling == Era5Pooling.CLS:
            return tokens[:, num_prior, :]
        # Data tokens come after prior + cls.
        start = num_prior + num_cls
        data_tokens = tokens[:, start:, :]
        data_mask = full_mask[:, start:]
        if self.pooling == Era5Pooling.MEAN:
            valid = (~data_mask).to(tokens.dtype).unsqueeze(-1)
            count = valid.sum(dim=1).clamp(min=1.0)
            return (data_tokens * valid).sum(dim=1) / count
        if self.pooling == Era5Pooling.ATTN:
            assert self.attn_query is not None
            assert self.attn_pool is not None
            q = self.attn_query.expand(tokens.shape[0], -1, -1)
            # MultiheadAttention treats True in key_padding_mask as positions
            # to IGNORE — exactly the convention of our `padding_mask`.
            out, _ = self.attn_pool(
                query=q, key=data_tokens, value=data_tokens, key_padding_mask=data_mask
            )
            return out.squeeze(1)
        raise ValueError(f"Unknown pooling: {self.pooling}")

    def apply_compile(self) -> None:
        """Apply torch.compile to the model."""
        self.compile(dynamic=True)
