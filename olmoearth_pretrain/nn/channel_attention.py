"""Channel attention for aggregating multi-modality bandset tokens.

Inspired by Panopticon (https://arxiv.org/abs/2503.10845), this module uses
cross-attention to fuse per-modality bandset tokens into a single token per
spatial-temporal position.

Also contains `AggregatedPredictor`, a decoder designed to work with
aggregated encoder outputs, reconstructing per-modality bandset tokens
via cross-attention against the aggregated context.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor

from olmoearth_pretrain.config import Config
from olmoearth_pretrain.data.constants import (
    BASE_GSD,
    ModalitySpec,
    get_modality_specs_from_names,
)
from olmoearth_pretrain.datatypes import MaskValue
from olmoearth_pretrain.nn.tokenization import TokenizationConfig

if TYPE_CHECKING:
    from olmoearth_pretrain.nn.flexi_vit import TokensAndMasks

logger = logging.getLogger(__name__)


@dataclass
class ChnAttnConfig(Config):
    """Configuration for channel attention aggregation.

    Args:
        enabled: Whether to use channel attention aggregation.
        attn_dim: Dimension for cross-attention. If None, defaults to embedding_size.
        num_heads: Number of attention heads.
        aggregation_mode: "all" fuses all modalities into 1 token per (h,w,t).
            "per_modality" fuses each modality's bandsets into 1 token,
            yielding N_modality tokens per (h,w,t).
    """

    enabled: bool = False
    attn_dim: int | None = None
    num_heads: int = 8
    aggregation_mode: str = "all"  # "all" | "per_modality"

    def validate(self) -> None:
        """Validate the configuration."""
        if self.aggregation_mode not in ("all", "per_modality"):
            raise ValueError(
                f"aggregation_mode must be 'all' or 'per_modality', got {self.aggregation_mode}"
            )


class CrossAttention(nn.Module):
    """Multi-head cross-attention: Q attends to K/V.

    No query projection since query is a learned parameter.
    """

    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False):
        """Initialize CrossAttention."""
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.scale = head_dim**-0.5

        self.proj_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_v = nn.Linear(dim, dim, bias=qkv_bias)

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Forward pass.

        Args:
            q: (B, Nq, D)
            k: (B, Nkv, D)
            v: (B, Nkv, D)
            key_padding_mask: (B, Nkv), True means IGNORE this key position.
        """
        B, Nq, D = q.shape
        Nkv = k.shape[1]

        # Cast to weight dtype for mixed-precision compatibility
        weight_dtype = self.proj_k.weight.dtype
        q = q.to(weight_dtype)
        k = k.to(weight_dtype)
        v = v.to(weight_dtype)

        q = q.reshape(B, Nq, self.num_heads, D // self.num_heads).permute(0, 2, 1, 3)
        q = q * self.scale

        k = (
            self.proj_k(k)
            .reshape(B, Nkv, self.num_heads, D // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        v = (
            self.proj_v(v)
            .reshape(B, Nkv, self.num_heads, D // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        attn = q @ k.transpose(-2, -1)  # (B, num_heads, Nq, Nkv)
        if key_padding_mask is not None:
            # key_padding_mask: (B, Nkv) -> (B, 1, 1, Nkv)
            attn = attn.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, Nq, D)
        return x


class ChnAttn(nn.Module):
    """Channel attention module that aggregates bandset tokens via cross-attention.

    For each (h,w,t) position, gathers bandset tokens from one or more modalities
    and fuses them into a reduced set of tokens using learned queries.

    Args:
        embedding_size: The embedding dimension of the backbone.
        attn_dim: Dimension for the cross-attention. If different from
            embedding_size, linear projections are added.
        num_heads: Number of attention heads.
        aggregation_mode: "all" or "per_modality".
        supported_modality_names: List of modality names this module handles.
        num_bandsets_per_modality: Dict mapping modality name to number of bandsets.
    """

    def __init__(
        self,
        embedding_size: int,
        attn_dim: int,
        num_heads: int,
        aggregation_mode: str,
        supported_modality_names: list[str],
        num_bandsets_per_modality: dict[str, int],
    ):
        """Initialize ChnAttn."""
        super().__init__()
        self.embedding_size = embedding_size
        self.attn_dim = attn_dim
        self.aggregation_mode = aggregation_mode
        self.supported_modality_names = supported_modality_names
        self.num_bandsets_per_modality = num_bandsets_per_modality

        # Projections if attn_dim != embedding_size
        self.proj_in = (
            nn.Linear(embedding_size, attn_dim)
            if attn_dim != embedding_size
            else nn.Identity()
        )
        self.proj_out = (
            nn.Linear(attn_dim, embedding_size)
            if attn_dim != embedding_size
            else nn.Identity()
        )

        self.cross_attn = CrossAttention(
            dim=attn_dim, num_heads=num_heads, qkv_bias=True
        )

        if aggregation_mode == "all":
            # Single learned query for all modalities
            self.query = nn.Parameter(torch.randn(1, 1, attn_dim) * 0.02)
        elif aggregation_mode == "per_modality":
            # One learned query per modality
            num_modalities = len(supported_modality_names)
            self.query = nn.Parameter(torch.randn(1, num_modalities, attn_dim) * 0.02)
        else:
            raise ValueError(f"Unknown aggregation_mode: {aggregation_mode}")

    def forward(
        self,
        tokens_dict: dict[str, Tensor],
        supported_modality_names: list[str],
    ) -> tuple[Tensor, Tensor]:
        """Aggregate bandset tokens via cross-attention.

        Args:
            tokens_dict: Dict with modality tokens (B, H, W, T, bandsets, D) and
                masks (B, H, W, T, bandsets). Only modality keys (no _mask suffix)
                and their corresponding masks are used.
            supported_modality_names: Modalities to process.

        Returns:
            aggregated_tokens: (B, H, W, T, N_out, D) where N_out=1 for "all"
                mode or N_modalities for "per_modality" mode.
            aggregated_mask: (B, H, W, T, N_out)
        """
        # Collect tokens and masks from present modalities
        all_tokens = []
        all_masks = []
        modalities_present = []

        for modality in supported_modality_names:
            if modality not in tokens_dict:
                continue
            modality_tokens = tokens_dict[modality]  # (B, H, W, T, bs, D)
            mask_key = f"{modality}_mask"
            modality_mask = tokens_dict[mask_key]  # (B, H, W, T, bs)
            all_tokens.append(modality_tokens)
            all_masks.append(modality_mask)
            modalities_present.append(modality)

        if len(all_tokens) == 0:
            raise ValueError("No modality tokens found for channel attention")

        # Get spatial dims from first modality
        ref = all_tokens[0]
        B, H, W, T = ref.shape[:4]
        D = ref.shape[-1]

        if self.aggregation_mode == "all":
            return self._forward_all(all_tokens, all_masks, B, H, W, T, D)
        else:
            return self._forward_per_modality(
                all_tokens, all_masks, modalities_present, B, H, W, T, D
            )

    def _forward_all(
        self,
        all_tokens: list[Tensor],
        all_masks: list[Tensor],
        B: int,
        H: int,
        W: int,
        T: int,
        D: int,
    ) -> tuple[Tensor, Tensor]:
        """All modalities fused into 1 token per (h,w,t)."""
        # Concat all bandsets: (B, H, W, T, total_bs, D)
        cat_tokens = torch.cat(all_tokens, dim=-2)
        cat_masks = torch.cat(all_masks, dim=-1)

        # Flatten spatial dims: (B*H*W*T, total_bs, D)
        flat_tokens = rearrange(cat_tokens, "b h w t n d -> (b h w t) n d")
        flat_masks = rearrange(cat_masks, "b h w t n -> (b h w t) n")

        # Project to attn_dim
        flat_tokens = self.proj_in(flat_tokens)

        # Build key_padding_mask: True where we should IGNORE
        # Only attend to ONLINE_ENCODER tokens
        ignore_mask = flat_masks != MaskValue.ONLINE_ENCODER.value

        # Expand query: (1, 1, attn_dim) -> (B*H*W*T, 1, attn_dim)
        BN = flat_tokens.shape[0]
        query = self.query.expand(BN, -1, -1)

        # Aggregated mask: ONLINE_ENCODER if any input was ONLINE_ENCODER at this position
        has_encoder = (cat_masks == MaskValue.ONLINE_ENCODER.value).any(
            dim=-1, keepdim=True
        )
        agg_mask = torch.where(
            has_encoder,
            torch.full_like(
                has_encoder, MaskValue.ONLINE_ENCODER.value, dtype=cat_masks.dtype
            ),
            torch.full_like(
                has_encoder, MaskValue.MISSING.value, dtype=cat_masks.dtype
            ),
        )

        # Cross-attend
        out = self.cross_attn(
            query, flat_tokens, flat_tokens, key_padding_mask=ignore_mask
        )

        # Project back
        out = self.proj_out(out)

        # Reshape: (B*H*W*T, 1, D) -> (B, H, W, T, 1, D)
        out = rearrange(out, "(b h w t) n d -> b h w t n d", b=B, h=H, w=W, t=T)

        # Zero out positions where all inputs were MISSING to prevent NaN
        # propagation from softmax over all-masked keys.
        # Use torch.where since nan * 0 = nan in IEEE 754.
        out = torch.where(
            has_encoder.unsqueeze(-1),
            out,
            torch.zeros_like(out),
        )

        return out, agg_mask

    def _forward_per_modality(
        self,
        all_tokens: list[Tensor],
        all_masks: list[Tensor],
        modalities_present: list[str],
        B: int,
        H: int,
        W: int,
        T: int,
        D: int,
    ) -> tuple[Tensor, Tensor]:
        """Each modality's bandsets fused to 1 token, yielding N_modality tokens per (h,w,t)."""
        out_tokens = []
        out_masks = []

        for i, (mod_tokens, mod_masks) in enumerate(zip(all_tokens, all_masks)):
            # mod_tokens: (B, H, W, T_i, bs, D), mod_masks: (B, H, W, T_i, bs)
            # Each modality may have a different T dimension
            B_i, H_i, W_i, T_i = mod_tokens.shape[:4]
            flat_tokens = rearrange(mod_tokens, "b h w t n d -> (b h w t) n d")
            flat_masks = rearrange(mod_masks, "b h w t n -> (b h w t) n")

            flat_tokens = self.proj_in(flat_tokens)
            ignore_mask = flat_masks != MaskValue.ONLINE_ENCODER.value

            BN = flat_tokens.shape[0]
            # Use the i-th query for this modality
            query_i = self.query[:, i : i + 1, :].expand(BN, -1, -1)

            # Mask: ONLINE_ENCODER if any bandset was ONLINE_ENCODER
            has_encoder = (mod_masks == MaskValue.ONLINE_ENCODER.value).any(
                dim=-1, keepdim=True
            )
            mod_agg_mask = torch.where(
                has_encoder,
                torch.full_like(
                    has_encoder, MaskValue.ONLINE_ENCODER.value, dtype=mod_masks.dtype
                ),
                torch.full_like(
                    has_encoder, MaskValue.MISSING.value, dtype=mod_masks.dtype
                ),
            )

            out = self.cross_attn(
                query_i, flat_tokens, flat_tokens, key_padding_mask=ignore_mask
            )
            out = self.proj_out(out)
            out = rearrange(
                out, "(b h w t) n d -> b h w t n d", b=B_i, h=H_i, w=W_i, t=T_i
            )

            # Zero out positions where all inputs were MISSING
            out = torch.where(
                has_encoder.unsqueeze(-1),
                out,
                torch.zeros_like(out),
            )

            out_tokens.append(out)
            out_masks.append(mod_agg_mask)

        # Pad T dimension to max_T across modalities so we can cat along
        # the modality (dim=-2) axis.
        max_T = max(t.shape[3] for t in out_tokens)
        padded_tokens = []
        padded_masks = []
        for out, mod_agg_mask in zip(out_tokens, out_masks):
            T_i = out.shape[3]
            if T_i < max_T:
                pad_t = max_T - T_i
                # out: (B, H, W, T_i, 1, D) -> pad T dim
                out = torch.nn.functional.pad(out, (0, 0, 0, 0, pad_t, 0))
                # mod_agg_mask: (B, H, W, T_i, 1) -> pad T dim with MISSING
                mask_pad = torch.full(
                    (*mod_agg_mask.shape[:3], pad_t, mod_agg_mask.shape[4]),
                    MaskValue.MISSING.value,
                    dtype=mod_agg_mask.dtype,
                    device=mod_agg_mask.device,
                )
                mod_agg_mask = torch.cat([mask_pad, mod_agg_mask], dim=3)
            padded_tokens.append(out)
            padded_masks.append(mod_agg_mask)

        # Stack: (B, H, W, max_T, N_modalities, D)
        agg_tokens = torch.cat(padded_tokens, dim=-2)
        agg_masks = torch.cat(padded_masks, dim=-1)

        return agg_tokens, agg_masks


class AggregatedPredictor(nn.Module):
    """Decoder for aggregated encoder outputs.

    Constructs per-modality/bandset mask tokens for DECODER-masked positions,
    cross-attends them against aggregated encoder context tokens, and outputs
    per-modality TokensAndMasks ready for the Reconstructor.

    Args:
        supported_modalities: Modalities this decoder supports.
        encoder_embedding_size: Embedding size of the encoder output.
        decoder_embedding_size: Internal embedding size for decoder transformer.
        depth: Number of cross-attention transformer blocks.
        mlp_ratio: MLP ratio in transformer blocks.
        num_heads: Number of attention heads.
        max_sequence_length: Maximum temporal sequence length.
        drop_path: Drop path rate.
        output_embedding_size: Output embedding dimension. Defaults to encoder_embedding_size.
        use_flash_attn: Whether to use flash attention.
        qk_norm: Whether to normalize Q and K.
        tokenization_config: Band grouping config.
    """

    def __init__(
        self,
        supported_modalities: list[ModalitySpec],
        encoder_embedding_size: int = 128,
        decoder_embedding_size: int = 128,
        depth: int = 2,
        mlp_ratio: float = 2.0,
        num_heads: int = 8,
        max_sequence_length: int = 24,
        drop_path: float = 0.0,
        output_embedding_size: int | None = None,
        use_flash_attn: bool = False,
        qk_norm: bool = False,
        tokenization_config: TokenizationConfig | None = None,
    ):
        """Initialize AggregatedPredictor."""
        super().__init__()
        # Lazy import to avoid circular dependency
        from olmoearth_pretrain.nn.attention import Block
        from olmoearth_pretrain.nn.flexi_vit import CompositeEncodings

        self.supported_modalities = supported_modalities
        self.supported_modality_names = [m.name for m in supported_modalities]
        self.encoder_embedding_size = encoder_embedding_size
        self.decoder_embedding_size = decoder_embedding_size
        self.tokenization_config = tokenization_config or TokenizationConfig()
        self.use_flash_attn = use_flash_attn

        if output_embedding_size is None:
            output_embedding_size = encoder_embedding_size
        self.output_embedding_size = output_embedding_size

        # Projection from encoder dim to decoder dim
        self.input_norm = nn.LayerNorm(encoder_embedding_size)
        self.encoder_to_decoder_embed = nn.Linear(
            encoder_embedding_size, decoder_embedding_size, bias=True
        )

        # Learnable mask token
        self.mask_token = nn.Parameter(torch.zeros(decoder_embedding_size))

        # Composite encodings for the mask tokens (channel + spatial + temporal)
        self.composite_encodings = CompositeEncodings(
            embedding_size=decoder_embedding_size,
            supported_modalities=supported_modalities,
            max_sequence_length=max_sequence_length,
            learnable_channel_embeddings=True,
            tokenization_config=self.tokenization_config,
        )

        # Cross-attention transformer blocks
        self.blocks = nn.ModuleList(
            [
                Block(
                    decoder_embedding_size,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    qk_norm=qk_norm,
                    norm_layer=nn.LayerNorm,
                    cross_attn=True,
                    drop_path=drop_path,
                    use_flash_attn=use_flash_attn,
                )
                for _ in range(depth)
            ]
        )

        # Output projection
        self.norm = nn.LayerNorm(decoder_embedding_size)
        self.to_output_embed = nn.Linear(
            decoder_embedding_size, output_embedding_size, bias=True
        )

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _build_mask_tokens(
        self,
        original_masks: dict[str, Tensor],
        timestamps: Tensor,
        patch_size: int,
        input_res: int,
    ) -> tuple[Tensor, Tensor, dict[str, tuple], dict[str, Tensor]]:
        """Build per-modality/bandset mask tokens for DECODER positions.

        Returns:
            mask_tokens: (B, N_decode, D_decoder) flattened mask tokens
            decode_mask: (B, N_decode) binary mask (1 = real token)
            modalities_to_dims: dict mapping modality -> original shape
            output_masks: dict of per-modality masks for the output
        """
        from olmoearth_pretrain.nn.flexi_vit import (
            get_modalities_to_process,
        )

        all_tokens_list = []
        all_masks_list = []
        modalities_to_dims = {}
        output_masks = {}

        available = [
            k.replace("_mask", "") for k in original_masks if k.endswith("_mask")
        ]
        modalities_to_process = get_modalities_to_process(
            available, self.supported_modality_names
        )

        for modality in modalities_to_process:
            mask_key = f"{modality}_mask"
            modality_mask = original_masks[mask_key]  # (B, P_H, P_W, T, bandsets)

            # Create tokens: broadcast mask_token to full shape
            token_shape = list(modality_mask.shape) + [self.decoder_embedding_size]
            mask_tokens = self.mask_token.expand(token_shape)
            # Clone so we have a real tensor (not just an expanded view)
            mask_tokens = mask_tokens.clone()

            modalities_to_dims[modality] = mask_tokens.shape
            all_tokens_list.append(rearrange(mask_tokens, "b ... d -> b (...) d"))
            all_masks_list.append(rearrange(modality_mask, "b ... -> b (...)"))
            output_masks[mask_key] = modality_mask

        if len(all_tokens_list) == 0:
            raise ValueError("No modalities to decode")

        # Concatenate all modalities
        all_tokens = torch.cat(all_tokens_list, dim=1)  # (B, N_total, D)
        all_mask_vals = torch.cat(all_masks_list, dim=1)  # (B, N_total)

        # Apply composite encodings per-modality, then re-flatten.
        tokens_only_dict: dict[str, Tensor] = {}
        for modality in modalities_to_process:
            mask_key = f"{modality}_mask"
            modality_mask = original_masks[mask_key]
            token_shape = list(modality_mask.shape) + [self.decoder_embedding_size]
            mask_tokens = self.mask_token.expand(token_shape).clone()
            tokens_only_dict[modality] = mask_tokens

        # Apply composite encodings (channel + spatial + temporal)
        tokens_with_encodings = self.composite_encodings.forward(
            tokens_only_dict,
            timestamps=timestamps,
            patch_size=patch_size,
            input_res=input_res,
        )

        encoded_tokens_list: list[Tensor] = []
        for modality in modalities_to_process:
            encoded_tokens_list.append(
                rearrange(tokens_with_encodings[modality], "b ... d -> b (...) d")
            )

        all_tokens = torch.cat(encoded_tokens_list, dim=1)
        return all_tokens, all_mask_vals, modalities_to_dims, output_masks

    def forward(
        self,
        encoder_output: TokensAndMasks,
        original_masks: dict[str, Tensor],
        timestamps: Tensor,
        patch_size: int,
        input_res: int = BASE_GSD,
        **kwargs: Any,
    ) -> TokensAndMasks:
        """Decode aggregated encoder tokens into per-modality predictions.

        Args:
            encoder_output: TokensAndMasks with aggregated field from encoder.
            original_masks: Dict of per-modality masks from encoder
                (e.g. {"sentinel2_l2a_mask": Tensor, ...}).
            timestamps: Timestamps tensor.
            patch_size: Patch size.
            input_res: Input resolution.
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            TokensAndMasks with per-modality/bandset predictions.
        """
        from olmoearth_pretrain.nn.flexi_vit import TokensAndMasks
        from olmoearth_pretrain.nn.utils import get_cumulative_sequence_lengths

        # 1. Prepare context from aggregated encoder tokens
        agg_tokens = encoder_output.aggregated  # (B, H, W, T, N_agg, D_enc)
        agg_mask = encoder_output.aggregated_mask  # (B, H, W, T, N_agg)
        assert agg_tokens is not None and agg_mask is not None

        # Flatten to (B, N_context, D_enc)
        context = rearrange(agg_tokens, "b ... d -> b (...) d")
        context_mask = rearrange(agg_mask, "b ... -> b (...)")

        # Project to decoder dim (cast for mixed-precision compatibility)
        context = self.input_norm(context.to(self.input_norm.weight.dtype))
        context = self.encoder_to_decoder_embed(context)

        # Context mask: True where token is ONLINE_ENCODER (should attend)
        context_attn_mask = context_mask == MaskValue.ONLINE_ENCODER.value

        # 2. Build mask tokens for all DECODER positions
        mask_tokens, mask_vals, modalities_to_dims, output_masks = (
            self._build_mask_tokens(original_masks, timestamps, patch_size, input_res)
        )

        # Separate tokens to decode (DECODER) from others
        is_decoder = mask_vals == MaskValue.DECODER.value
        # Count decoder tokens per sample
        seq_lengths = is_decoder.sum(dim=-1)
        max_decode_len = seq_lengths.max().item()

        if max_decode_len == 0:
            # Nothing to decode - return empty tokens
            empty_dict: dict[str, Tensor] = {}
            for modality, dims in modalities_to_dims.items():
                empty_dict[modality] = torch.zeros(
                    *dims[:-1],
                    self.output_embedding_size,
                    device=mask_tokens.device,
                    dtype=mask_tokens.dtype,
                )
                empty_dict[f"{modality}_mask"] = output_masks[f"{modality}_mask"]
            return TokensAndMasks(**empty_dict)

        # Sort so decoder tokens are first
        sorted_mask, indices = torch.sort(
            is_decoder.int(), dim=1, descending=True, stable=True
        )
        mask_tokens = mask_tokens.gather(1, indices[:, :, None].expand_as(mask_tokens))

        # Extract decoder tokens
        tokens_to_decode = mask_tokens[:, :max_decode_len]
        decode_attn_mask = sorted_mask[:, :max_decode_len]

        # 3. Cross-attention: Q=tokens_to_decode, K/V=context
        if self.use_flash_attn:
            cu_seqlens_q = get_cumulative_sequence_lengths(seq_lengths)
            cu_seqlens_k = get_cumulative_sequence_lengths(
                context_attn_mask.sum(dim=-1)
            )
            og_shape_q = tokens_to_decode.shape
            # Pack
            tokens_to_decode_packed = tokens_to_decode.flatten(0, 1)[
                decode_attn_mask.flatten().bool()
            ]
            context_packed = context.flatten(0, 1)[context_attn_mask.flatten().bool()]
        else:
            cu_seqlens_q = None
            cu_seqlens_k = None

        for blk in self.blocks:
            if self.use_flash_attn:
                tokens_to_decode_packed = blk(
                    x=tokens_to_decode_packed,
                    y=context_packed,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_decode_len,
                    max_seqlen_k=context_attn_mask.sum(dim=-1).max().item(),
                )
            else:
                tokens_to_decode = blk(
                    x=tokens_to_decode,
                    y=context,
                    attn_mask=context_attn_mask,
                )

        if self.use_flash_attn:
            # Unpack
            tokens_to_decode = torch.zeros(
                og_shape_q,
                dtype=tokens_to_decode_packed.dtype,
                device=tokens_to_decode_packed.device,
            )
            tokens_to_decode.flatten(0, 1)[decode_attn_mask.flatten().bool()] = (
                tokens_to_decode_packed
            )

        # 4. Put decoded tokens back into full sequence
        B, N_total = mask_vals.shape
        D = tokens_to_decode.shape[-1]
        full_tokens = torch.zeros(
            B, N_total, D, device=mask_tokens.device, dtype=mask_tokens.dtype
        )
        full_tokens[:, :max_decode_len] = tokens_to_decode * decode_attn_mask.unsqueeze(
            -1
        )
        # Scatter back to original positions
        full_tokens = full_tokens.scatter(
            1, indices[:, :, None].expand_as(full_tokens), full_tokens
        )

        # 5. Split back per modality and apply output projection
        output_dict: dict[str, Tensor] = {}
        token_offset = 0
        from olmoearth_pretrain.nn.flexi_vit import get_modalities_to_process

        available = [
            k.replace("_mask", "") for k in original_masks if k.endswith("_mask")
        ]
        modalities_to_process = get_modalities_to_process(
            available, self.supported_modality_names
        )

        for modality in modalities_to_process:
            dims = modalities_to_dims[modality]
            middle_dims = dims[1:-1]  # everything except B and D
            num_tokens = 1
            for d in middle_dims:
                num_tokens *= d

            modality_tokens = full_tokens[:, token_offset : token_offset + num_tokens]
            modality_tokens = modality_tokens.view(dims[0], *middle_dims, D)
            token_offset += num_tokens

            # Per-bandset output projection
            num_bandsets = self.tokenization_config.get_num_bandsets(modality)
            per_bandset_outputs = []
            for idx in range(num_bandsets):
                bandset_data = modality_tokens[..., idx, :].to(self.norm.weight.dtype)
                out = self.to_output_embed(self.norm(bandset_data))
                per_bandset_outputs.append(out)
            output_dict[modality] = torch.stack(per_bandset_outputs, dim=-2)
            output_dict[f"{modality}_mask"] = output_masks[f"{modality}_mask"]

        return TokensAndMasks(**output_dict)

    def apply_fsdp(self, **fsdp_kwargs: Any) -> None:
        """Apply FSDP to the model."""
        from torch.distributed.fsdp import fully_shard

        for block in self.blocks:
            if hasattr(block, "apply_fsdp"):
                block.apply_fsdp(**fsdp_kwargs)
        fully_shard(self, **fsdp_kwargs)

    def apply_compile(self) -> None:
        """Apply torch.compile to the model."""
        for block in self.blocks:
            if hasattr(block, "apply_compile"):
                block.apply_compile()


@dataclass
class AggregatedPredictorConfig(Config):
    """Configuration for the AggregatedPredictor."""

    supported_modality_names: list[str]
    encoder_embedding_size: int = 16
    decoder_embedding_size: int = 16
    depth: int = 2
    mlp_ratio: float = 1.0
    num_heads: int = 2
    max_sequence_length: int = 12
    drop_path: float = 0.0
    output_embedding_size: int | None = None
    use_flash_attn: bool = False
    qk_norm: bool = False
    tokenization_config: TokenizationConfig | None = None

    @property
    def supported_modalities(self) -> list[ModalitySpec]:
        """Get the supported modalities."""
        return get_modality_specs_from_names(self.supported_modality_names)

    def validate(self) -> None:
        """Validate the configuration."""
        if len(self.supported_modalities) == 0:
            raise ValueError("At least one modality must be added!")

    def build(self) -> AggregatedPredictor:
        """Build the AggregatedPredictor."""
        self.validate()
        kwargs = self.as_dict(exclude_none=True, recurse=False)
        kwargs.pop("supported_modality_names")
        kwargs["supported_modalities"] = self.supported_modalities
        return AggregatedPredictor(**kwargs)
