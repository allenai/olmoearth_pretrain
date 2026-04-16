"""Text-conditioned decoder built from existing OlmoEarth attention blocks.

Reuses :class:`olmoearth_pretrain.nn.attention.Block` (which already supports
``cross_attn=True``) to compose a stack of (self-attn over image tokens) +
(cross-attn from image tokens to text tokens) + FFN layers.

We deliberately do **not** introduce a new attention implementation — the
existing ``Block`` already handles flash attention, qk_norm, layer scale,
drop path, and FSDP/compile, and matches the rest of the repo.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from olmoearth_pretrain.config import Config
from olmoearth_pretrain.nn.attention import Block


class _DecoderLayer(nn.Module):
    """One decoder layer: self-attn over image tokens, then cross-attn to text."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        qkv_bias: bool,
        qk_norm: bool,
        drop: float,
        attn_drop: float,
        drop_path: float,
        use_flash_attn: bool,
        norm_layer: type[nn.Module] = nn.LayerNorm,
    ) -> None:
        """Initialize the decoder layer."""
        super().__init__()
        # Self-attention over image tokens (with its own MLP).
        self.self_block = Block(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            norm_layer=norm_layer,
            cross_attn=False,
            use_flash_attn=use_flash_attn,
        )
        # Cross-attention: image tokens query, text tokens supply K/V.
        self.cross_block = Block(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            norm_layer=norm_layer,
            cross_attn=True,
            use_flash_attn=use_flash_attn,
        )

    def forward(
        self,
        image_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        text_attn_mask: torch.Tensor | None = None,
        image_attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run one self-attn + cross-attn pass.

        Args:
            image_tokens: ``[B, N, D]``.
            text_tokens: ``[B, L, D]`` — text K/V.
            text_attn_mask: ``[B, L]`` — True for valid text positions. The
                underlying ``sdpa`` accepts this directly when not using
                flash attention.
            image_attn_mask: ``[B, N]`` — True for valid image positions.

        Returns:
            ``[B, N, D]`` — refined image tokens.
        """
        image_tokens = self.self_block(x=image_tokens, attn_mask=image_attn_mask)
        image_tokens = self.cross_block(
            x=image_tokens, y=text_tokens, attn_mask=text_attn_mask
        )
        return image_tokens


@dataclass
class CrossAttnDecoderConfig(Config):
    """Configuration for :class:`CrossAttnDecoder`.

    Attributes:
        dim: Decoder hidden size.
        depth: Number of (self-attn + cross-attn) layers.
        num_heads: Multi-head attention heads.
        mlp_ratio: Hidden / input ratio in each block's MLP.
        qkv_bias: Whether to add bias to QKV projections.
        qk_norm: Whether to LayerNorm Q and K (matches encoder default).
        drop: Hidden / projection dropout rate.
        attn_drop: Attention dropout rate.
        drop_path: Stochastic depth rate.
        use_flash_attn: Use flash attention if available.
    """

    dim: int = 512
    depth: int = 4
    num_heads: int = 8
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    qk_norm: bool = False
    drop: float = 0.0
    attn_drop: float = 0.0
    drop_path: float = 0.0
    use_flash_attn: bool = False

    def build(self, image_dim: int, text_dim: int) -> CrossAttnDecoder:
        """Build the decoder, projecting from encoder/text dims to ``dim``."""
        return CrossAttnDecoder(self, image_dim=image_dim, text_dim=text_dim)


class CrossAttnDecoder(nn.Module):
    """Stack of self-attn + cross-attn layers conditioned on text tokens.

    Image and text inputs are first projected to ``config.dim``. The output
    is the refined image-token sequence, ready to be turned into per-pixel
    logits by the open-set model's classification head.
    """

    def __init__(
        self,
        config: CrossAttnDecoderConfig,
        image_dim: int,
        text_dim: int,
    ) -> None:
        """Initialize the decoder."""
        super().__init__()
        self.config = config
        self.image_proj = (
            nn.Identity()
            if image_dim == config.dim
            else nn.Linear(image_dim, config.dim)
        )
        self.text_proj = (
            nn.Identity() if text_dim == config.dim else nn.Linear(text_dim, config.dim)
        )

        self.layers = nn.ModuleList(
            [
                _DecoderLayer(
                    dim=config.dim,
                    num_heads=config.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    qkv_bias=config.qkv_bias,
                    qk_norm=config.qk_norm,
                    drop=config.drop,
                    attn_drop=config.attn_drop,
                    drop_path=config.drop_path,
                    use_flash_attn=config.use_flash_attn,
                )
                for _ in range(config.depth)
            ]
        )
        self.norm = nn.LayerNorm(config.dim)

    @property
    def output_dim(self) -> int:
        """Dimensionality of refined image tokens."""
        return self.config.dim

    def forward(
        self,
        image_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        text_attn_mask: torch.Tensor | None = None,
        image_attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the full decoder."""
        image_tokens = self.image_proj(image_tokens)
        text_tokens = self.text_proj(text_tokens)
        for layer in self.layers:
            image_tokens = layer(
                image_tokens=image_tokens,
                text_tokens=text_tokens,
                text_attn_mask=text_attn_mask,
                image_attn_mask=image_attn_mask,
            )
        return self.norm(image_tokens)
