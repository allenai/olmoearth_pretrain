""" Alternate Predictor that pools the tokens across modalities. And then predicts all the other modalities from that spatiall pooled representation"""

from typing import Any
from helios.nn.flexihelios import Predictor, TokensAndMasks, return_modalities_from_dict, get_modalities_to_process, Encoder, EncoderConfig, PredictorConfig
from helios.data.constants import Modality, ModalitySpec, BASE_GSD
from helios.train.masking import MaskedHeliosSample, MaskValue
import torch
from enum import StrEnum
from torch import Tensor
from olmo_core.config import Config
from dataclasses import dataclass
from helios.dataset.utils import get_modality_specs_from_names
from helios.nn.attention import Mlp
import logging
from einops import rearrange
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.fsdp import fully_shard
logger = logging.getLogger(__name__)

# should this go after the composite encodings or before?
# It is only happening on the encoding tokens
# so after seems easier to implement because you otherwise need to repack everything to do this

# I should try both and see if there is a difference

# First I will do it after the composite encodings

class AttnPool(nn.Module):
    """
    Multi-query attention pooling with gated averaging.

    Args:
        in_dim (int): token dim (must be divisible by 64; head_dim=64).
        hidden_dim (int): MLP hidden/out dim (defaults to in_dim unless mlp_ratio provided).
        mlp_ratio (float|None): if set, hidden_dim := int(in_dim * mlp_ratio)
        num_queries (int): number of learned queries per (t,s) group.
        gate_temperature (float): temperature for softmax gating (>0).
    """
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int | None = None,
        mlp_ratio: float | None = None,
        num_queries: int = 3,
        gate_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        assert in_dim % 64 == 0, "in_dim must be divisible by 64"
        self.num_heads: int = in_dim // 64
        self.num_queries: int = num_queries
        self.gate_temperature: float = gate_temperature

        # k learned queries (k, D)
        self.query_tokens: nn.Parameter = nn.Parameter(torch.empty(num_queries, in_dim))

        # shared KV projection
        self.kv: nn.Linear = nn.Linear(in_dim, in_dim * 2)

        # output MLP (+ optional expansion via mlp_ratio)
        if mlp_ratio is not None:
            hidden_dim = int(in_dim * mlp_ratio)
        hidden_dim = hidden_dim or in_dim
        self.out_layer: Mlp = Mlp(in_dim, hidden_dim)
        self.out_norm = nn.LayerNorm(in_dim)

        # gating over k query outputs (maps D -> 1 per query)
        self.gate: nn.Linear | None = nn.Linear(in_dim, 1, bias=False) if num_queries > 1 else None

        self.init_weights()

    def init_weights(self) -> None:
        nn.init.zeros_(self.query_tokens)  # start at 0 so QK = 0 → uniform attn
        # K = 0, V = identity at init
        nn.init.zeros_(self.kv.weight)
        with torch.no_grad():
            D = self.kv.in_features
            self.kv.weight[D:, :] = torch.eye(D)  # second half is V-proj

        nn.init.zeros_(self.kv.bias)
        if self.gate is not None:
            nn.init.zeros_(self.gate.weight)  # start near uniform mix

    def masked_mean(self, x, mask):  # x:[B*,N,D], mask:[B*,N] (True=masked)
        """
        Compute the mean of x, weighted by the inverse of the mask.
        """
        if mask is None:
            return x.mean(dim=1)
        w = (~mask).float()
        w = w / (w.sum(dim=1, keepdim=True).clamp_min(1))
        return (x * w.unsqueeze(-1)).sum(dim=1)


    def forward(self, feat_tokens: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        """
        feat_tokens: [B*, N, D]  (B* = collapsed batch like B·T·S)
        mask:       [B*, N] or None  (True/1 = mask out)
        returns:    [B*, D]
        """
        Bc, N, D = feat_tokens.shape
        H = self.num_heads
        Dh = D // H

        # queries: [B*, k, D] -> [B*, H, k, Dh]
        q = self.query_tokens[None, :, :].expand(Bc, -1, -1).reshape(Bc, self.num_queries, H, Dh)
        q = rearrange(q, "b k h d -> b h k d")

        # K/V: [B*, N, D] -> [2, B*, H, N, Dh]
        feat_tokens = feat_tokens.to(self.kv.weight.dtype)
        masked_mean_feat_tokens = self.masked_mean(feat_tokens, mask)
        kv = self.kv(feat_tokens).reshape(Bc, N, 2, H, Dh)
        kv = rearrange(kv, "b n two h d -> two b h n d")
        k, v = torch.unbind(kv, dim=0)  # [B*, H, N, Dh] each

        # mask -> [B*, H, k, N] (broadcastable is fine, but expand for clarity)
        attn_mask = None
        if mask is not None:
            m = mask[:, None, None, :]  # [B*,1,1,N]
            attn_mask = m.expand(Bc, H, self.num_queries, N)

        # H100 chunking on batch axis
        max_size = 63488
        x_chunks = []
        for i in range(0, Bc, max_size):
            q_chunk = q[i : i + max_size, ...]
            k_chunk = k[i : i + max_size, ...]
            v_chunk = v[i : i + max_size, ...]
            m_chunk = attn_mask[i : i + max_size, ...] if attn_mask is not None else None
            # SDPA expects [B,H,Q,D] x [B,H,K,D] -> [B,H,Q,D]
            x_chunk = F.scaled_dot_product_attention(q_chunk, k_chunk, v_chunk, attn_mask=m_chunk)
            x_chunks.append(x_chunk)

        # [B*, H, k, Dh] -> [B*, k, D]
        x = torch.cat(x_chunks, dim=0)
        o = rearrange(x, "b h k d -> b k (h d)")

        # gated average across k, or pass-through if k=1
        if self.num_queries > 1:
            o_for_gate = F.layer_norm(o, (D,))   # normalize only for gating
            logits = self.gate(o_for_gate).squeeze(-1)  # [B*, k]
            w = torch.softmax(logits, dim=1)
            z = (w.unsqueeze(-1) * o).sum(dim=1)        # mix the *unnormalized* values
        else:
            z = o.squeeze(1)

        # MLP + LN head
        z = self.out_norm(self.out_layer(z))
        return z + masked_mean_feat_tokens

class PooledModalityPredictor(Predictor):
    """Predictor that pools the tokens across modalities. And then predicts all the other modalities from that spatiall pooled representation"""
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.attn_pool = AttnPool(self.embedding_size, self.embedding_size)

    def apply_attn(
        self,
        x: dict[str, Tensor],
        timestamps: Tensor,
        patch_size: int,
        input_res: int,
    ) -> dict[str, Tensor]:
        """Apply attention to the tokens."""
        logger.warning(
            "Calling apply_attn for PooledModalityPredictor"
        )
        tokens_only_dict, original_masks_dict, modalities_to_dims_dict = (
            self.split_tokens_masks_and_dims(x)
        )
        tokens_dict = self.composite_encodings(
            tokens_only_dict, timestamps, patch_size, input_res
        )
        tokens_dict.update(original_masks_dict)

        spatial_tokens, spatial_masks = self.stack_spatial_modalities_and_masks(tokens_dict)
        # I want to get to a shape of (B, H, W T) x M x D and then attentive pool across modalities
        B, H, W, T, M, D = spatial_tokens.shape
        spatial_tokens = rearrange(spatial_tokens, "b h w t m d -> (b h w t) m d")

        spatial_masks = rearrange(spatial_masks, "b h w t m -> (b h w t) m")
        # print the unique values of the masks
        logger.info(f"unique values of the masks: {torch.unique(spatial_masks)}")
        pooled_attn_mask = spatial_masks == MaskValue.ONLINE_ENCODER.value
        # Do I potentially need to filter out tokens that have no online marked modalities? Maybe not because we will just disgard those


        pooled_tokens = self.attn_pool(spatial_tokens, pooled_attn_mask)
        logger.info(f"shape of pooled tokens: {pooled_tokens.shape}")
        pooled_tokens = rearrange(pooled_tokens, "(b h w t) d -> b (h w t) d", b=B, h=H, w=W, t=T, d=D)
        # for spatial_masks if any in the modality dimension is online encode, set the token to online encoder only
        # otherwise set to Missing Value
        online_encoder_only_mask = (spatial_masks == MaskValue.ONLINE_ENCODER.value).any(dim=-1)
        pooled_attn_mask = torch.where(online_encoder_only_mask, MaskValue.ONLINE_ENCODER.value, MaskValue.MISSING.value)

        pooled_attn_mask = rearrange(pooled_attn_mask, "(b h w t) -> b (h w t)", b=B, h=H, w=W, t=T)
        logger.info(f"shape of pooled tokens: {pooled_tokens.shape}")

        (
            _,
            pooled_tokens,
            _,
            pooled_attn_mask,
            _,
            _,
            _,
            _,
            _,
        ) = self.split_x_y(pooled_tokens, pooled_attn_mask)


        # I need to do a step where I basically split the pooled tokens up so that I have an instance wide
        # collapsed mask of these

        all_tokens, mask = self.collapse_and_combine_hwtc(tokens_dict)
        # X contains the tokens to decode, Y contains the tokens to attend to for context
        (
            tokens_to_decode,
            unmasked_tokens,
            tokens_to_decode_mask,
            unmasked_tokens_mask,
            indices,
            seqlens_tokens_to_decode,
            seqlens_unmasked_tokens,
            max_length_of_tokens_to_decode,
            max_length_of_unmasked_tokens,
        ) = self.split_x_y(all_tokens, mask)
        # Pack x tokens
        if self.use_flash_attn:
            og_shape_tokens_to_decode = tokens_to_decode.shape
            tokens_to_decode = self.pack_tokens(
                tokens_to_decode, tokens_to_decode_mask.bool()
            )
            og_shape_unmasked_tokens = unmasked_tokens.shape
            unmasked_tokens = self.pack_tokens(
                unmasked_tokens, unmasked_tokens_mask.bool()
            )
            cu_seqlens_tokens_to_decode = get_cumulative_sequence_lengths(
                seqlens_tokens_to_decode
            )
            cu_seqlens_unmasked_tokens = get_cumulative_sequence_lengths(
                seqlens_unmasked_tokens
            )
        else:
            cu_seqlens_tokens_to_decode = None
            cu_seqlens_unmasked_tokens = None

        for blk in self.blocks:
            # note that we are not taking the inverse of the mask, since split_x_y gives us
            # true values for values we want to take part in attention
            tokens_to_decode = blk(
                x=tokens_to_decode,
                y=pooled_tokens,
                attn_mask=(
                    pooled_attn_mask.bool() if not self.use_flash_attn else None
                ),  # only for flash attn though this should not be left in
                # Assume not compatible with flash attn for now
                # cu_seqlens_q=cu_seqlens_tokens_to_decode,
                # cu_seqlens_k=cu_seqlens_unmasked_tokens,
                # max_seqlen_q=max_length_of_tokens_to_decode,
                # max_seqlen_k=max_length_of_unmasked_tokens,
            )

        if self.use_flash_attn:
            tokens_to_decode = self.unpack_tokens(
                tokens_to_decode,
                tokens_to_decode_mask.bool(),
                og_shape_tokens_to_decode,
            )
            unmasked_tokens = self.unpack_tokens(
                unmasked_tokens, unmasked_tokens_mask.bool(), og_shape_unmasked_tokens
            )

        x = self.combine_x_y(
            tokens_to_decode=tokens_to_decode,
            unmasked_tokens=unmasked_tokens,
            tokens_to_decode_mask=tokens_to_decode_mask,
            unmasked_tokens_mask=unmasked_tokens_mask,
            indices=indices,
        )
        tokens_per_modality_dict = self.split_and_expand_per_modality(
            x, modalities_to_dims_dict
        )
        tokens_per_modality_dict.update(original_masks_dict)
        return tokens_per_modality_dict


@dataclass
class PooledModalityPredictorConfig(Config):


    """Configuration for the Predictor."""

    supported_modality_names: list[str]
    encoder_embedding_size: int = 16
    decoder_embedding_size: int = 16
    depth: int = 2
    mlp_ratio: float = 1.0
    num_heads: int = 2
    max_sequence_length: int = 12
    drop_path: float = 0.0
    learnable_channel_embeddings: bool = True
    random_channel_embeddings: bool = False
    output_embedding_size: int | None = None
    use_flash_attn: bool = False
    qk_norm: bool = False

    def validate(self) -> None:
        """Validate the configuration."""
        if len(self.supported_modalities) == 0:
            raise ValueError("At least one modality must be added!")
        else:
            for modality in self.supported_modalities:
                if modality not in Modality.values():
                    raise ValueError(f"Modality {modality} is not supported")

    @property
    def supported_modalities(self) -> list[ModalitySpec]:
        """Get the supported modalities."""
        return get_modality_specs_from_names(self.supported_modality_names)

    def build(self) -> "Predictor":
        """Build the predictor."""
        self.validate()
        kwargs = self.as_dict(exclude_none=True, recurse=False)
        # supported_modality_names is replaced by supported_modalities
        kwargs.pop("supported_modality_names")
        kwargs["supported_modalities"] = self.supported_modalities
        logger.info(f"Predictor kwargs: {kwargs}")
        return PooledModalityPredictor(**kwargs)

# Pooled modality predictor V2
class PooledModalityPredictorV2(Predictor):
    """Predictor that pools the tokens across modalities."""
    def __init__(self, include_encoder_encodings: bool = True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.include_encoder_encodings = include_encoder_encodings

    def apply_attn(
        self,
        x: dict[str, Tensor],
        pooled_dict: dict[str, Tensor],
        timestamps: Tensor,
        patch_size: int,
        input_res: int,
    ) -> dict[str, Tensor]:
        """Apply attention to the tokens."""
        logger.warning(
            "Calling apply_attn for PooledModalityPredictor"
        )
        tokens_only_dict, original_masks_dict, modalities_to_dims_dict = (
            self.split_tokens_masks_and_dims(x)
        )
        tokens_dict = self.composite_encodings(
            tokens_only_dict, timestamps, patch_size, input_res
        )
        tokens_dict.update(original_masks_dict)

        pooled_tokens = pooled_dict["modality_pooled_tokens"]
        if self.include_encoder_encodings:
            pooled_tokens = self.composite_encodings._apply_encodings_per_modality(Modality.SENTINEL2_L2A.name, pooled_tokens, timestamps, patch_size, input_res, use_modality_encodings=False)
        pooled_tokens = rearrange(pooled_tokens, "b ... d -> b (...) d")
        pooled_attn_mask = rearrange(pooled_dict["modality_pooled_masks"], "b ... -> b (...)")

        (
            _,
            pooled_tokens,
            _,
            pooled_attn_mask,
            _,
            _,
            _,
            _,
            _,
        ) = self.split_x_y(pooled_tokens, pooled_attn_mask)


        # I need to do a step where I basically split the pooled tokens up so that I have an instance wide
        # collapsed mask of these

        all_tokens, mask = self.collapse_and_combine_hwtc(tokens_dict)
        # X contains the tokens to decode, Y contains the tokens to attend to for context
        (
            tokens_to_decode,
            unmasked_tokens,
            tokens_to_decode_mask,
            unmasked_tokens_mask,
            indices,
            seqlens_tokens_to_decode,
            seqlens_unmasked_tokens,
            max_length_of_tokens_to_decode,
            max_length_of_unmasked_tokens,
        ) = self.split_x_y(all_tokens, mask)
        # Pack x tokens
        if self.use_flash_attn:
            og_shape_tokens_to_decode = tokens_to_decode.shape
            tokens_to_decode = self.pack_tokens(
                tokens_to_decode, tokens_to_decode_mask.bool()
            )
            og_shape_unmasked_tokens = unmasked_tokens.shape
            unmasked_tokens = self.pack_tokens(
                unmasked_tokens, unmasked_tokens_mask.bool()
            )
            cu_seqlens_tokens_to_decode = get_cumulative_sequence_lengths(
                seqlens_tokens_to_decode
            )
            cu_seqlens_unmasked_tokens = get_cumulative_sequence_lengths(
                seqlens_unmasked_tokens
            )
        else:
            cu_seqlens_tokens_to_decode = None
            cu_seqlens_unmasked_tokens = None

        for blk in self.blocks:
            # note that we are not taking the inverse of the mask, since split_x_y gives us
            # true values for values we want to take part in attention
            tokens_to_decode = blk(
                x=tokens_to_decode,
                y=pooled_tokens,
                attn_mask=(
                    pooled_attn_mask.bool() if not self.use_flash_attn else None
                ),  # only for flash attn though this should not be left in
                # Assume not compatible with flash attn for now
                # cu_seqlens_q=cu_seqlens_tokens_to_decode,
                # cu_seqlens_k=cu_seqlens_unmasked_tokens,
                # max_seqlen_q=max_length_of_tokens_to_decode,
                # max_seqlen_k=max_length_of_unmasked_tokens,
            )

        if self.use_flash_attn:
            tokens_to_decode = self.unpack_tokens(
                tokens_to_decode,
                tokens_to_decode_mask.bool(),
                og_shape_tokens_to_decode,
            )
            unmasked_tokens = self.unpack_tokens(
                unmasked_tokens, unmasked_tokens_mask.bool(), og_shape_unmasked_tokens
            )

        x = self.combine_x_y(
            tokens_to_decode=tokens_to_decode,
            unmasked_tokens=unmasked_tokens,
            tokens_to_decode_mask=tokens_to_decode_mask,
            unmasked_tokens_mask=unmasked_tokens_mask,
            indices=indices,
        )
        tokens_per_modality_dict = self.split_and_expand_per_modality(
            x, modalities_to_dims_dict
        )
        tokens_per_modality_dict.update(original_masks_dict)
        return tokens_per_modality_dict

    def forward(
        self,
        x: TokensAndMasks,
        pooled_dict: dict[str, Tensor],
        timestamps: Tensor,
        patch_size: int,
        input_res: int = BASE_GSD,
    ) -> TokensAndMasks:
        """Generate predictions from encoded token representations.

        Args:
            x: TokensAndMasks containing the encoded tokens to make predictions from
            timestamps: Timestamps of the tokens
            patch_size: Patch size of the tokens
            input_res: Input resolution of the tokens

        Returns:
            TokensAndMasks containing the predicted tokens and their masks
        """
        decoder_emedded_dict = x.as_dict(return_none=False)
        # Apply Input Norms and encoder to decoder embeds to each modality
        available_modalities = x.modalities
        modalities_to_process = get_modalities_to_process(
            available_modalities, self.supported_modality_names
        )
        for modality in modalities_to_process:
            x_modality = getattr(x, modality)
            # Are these normalizations masked correctly?
            # Does not account for missing tokens
            x_modality = self.input_norm(x_modality)
            x_modality = self.encoder_to_decoder_embed(x_modality)
            masked_modality_name = x.get_masked_modality_name(modality)
            decoder_emedded_dict[modality] = x_modality
            decoder_emedded_dict[masked_modality_name] = getattr(
                x, masked_modality_name
            )

        # Apply input norma nd projection on pooled tokens
        pooled_tokens = pooled_dict["modality_pooled_tokens"]
        pooled_tokens = self.input_norm(pooled_tokens)
        pooled_tokens = self.encoder_to_decoder_embed(pooled_tokens)
        pooled_dict["modality_pooled_tokens"] = pooled_tokens

        tokens_only_dict = self.add_masks(decoder_emedded_dict)
        decoder_emedded_dict.update(tokens_only_dict)
        tokens_and_masks = self.apply_attn(
            decoder_emedded_dict, pooled_dict, timestamps, patch_size, input_res
        )
        # TODO: Factor this out into a more readable function
        output_dict = {}
        available_modalities = return_modalities_from_dict(tokens_and_masks)
        modalities_to_process = get_modalities_to_process(
            available_modalities, self.supported_modality_names
        )
        for modality in modalities_to_process:
            masked_modality_name = MaskedHeliosSample.get_masked_modality_name(modality)
            modality_mask = tokens_and_masks[masked_modality_name]
            # patchify masked data
            per_modality_output_tokens = []
            modality_data = tokens_and_masks[modality]

            band_sets = Modality.get(modality).band_sets
            for idx in range(len(band_sets)):
                per_channel_modality_data = modality_data[..., idx, :]
                output_data = self.to_output_embed(self.norm(per_channel_modality_data))
                per_modality_output_tokens.append(output_data)
            output_dict[modality] = torch.stack(per_modality_output_tokens, dim=-2)
            output_dict[masked_modality_name] = modality_mask
        return TokensAndMasks(**output_dict)




@dataclass
class PooledModalityPredictorV2Config(PredictorConfig):
    """Configuration for the PooledModalityPredictorV2."""
    include_encoder_encodings: bool = True
    def build(self) -> "Predictor":
        """Build the predictor."""
        self.validate()
        kwargs = self.as_dict(exclude_none=True, recurse=False)
        # supported_modality_names is replaced by supported_modalities
        kwargs.pop("supported_modality_names")
        kwargs["supported_modalities"] = self.supported_modalities
        logger.info(f"Predictor kwargs: {kwargs}")
        return PooledModalityPredictorV2(**kwargs)




# I can do Modality Pooling then Temporal pooling then spaital pooling
# Or I can just pool altogether in whatever dimensions I want (Try this first)
# Design Options
# Pooling can be deeper
# Pooling can be normed after pooling? (start with no but this may be needed)

class DimsToPool(StrEnum):
    MODALITY = "modality" # 1
    TEMPORAL = "temporal" # 2
    SPATIAL = "spatial"
    MODALITY_TEMPORAL = "modality_temporal" # 3
    # MODALITY_SPATIAL = "modality_spatial"
    # TEMPORAL_SPATIAL = "temporal_spatial"
    ALL = "all" # 4

# Try doing each seperately first then 1 predictor for each

# Encoder Pooling predictor
# in the end the pooled tokens dict should just be a more granular option depending on the task so we don't have to worry about mean max pooling average pooling or anyhting like that
class EncoderAttnPool(Encoder):
    """Encoder that pools the tokens across modalities."""
    def __init__(self, dims_to_pool: str, attn_pool_mlp_ratio: float | None = None, num_queries: int = 1, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.attn_pool = AttnPool(self.embedding_size, self.embedding_size, mlp_ratio=attn_pool_mlp_ratio, num_queries=num_queries)

        self.dims_to_pool = dims_to_pool

    def _get_reduce_and_expand_args(self, shape: tuple[int, ...]) -> tuple[str, str, dict[str, int], dict[str, int]]:
        """Get the reduction and expansion arguments for the dimensions to pool."""
        B, H, W, T, M, D = shape
        # Make a reduction args and expand args for each dim pooling type
        if self.dims_to_pool == DimsToPool.MODALITY:
            reduction_args = f"(b h w t) m d"
            reduction_mask_args = "(b h w t) m"
            pre_expand_args = "(b h w t) d"
            expand_args = "b h w t d"
            expand_mask_kwargs = {"b": B, "h": H, "w": W, "t": T}
            expand_kwargs = {"b": B, "h": H, "w": W, "t": T, "d": D}
        elif self.dims_to_pool == DimsToPool.TEMPORAL:
            reduction_args = f"(b h w m) t d"
            reduction_mask_args = "(b h w m) t"
            pre_expand_args = "(b h w m) d"
            expand_args = "b h w m d"
            expand_mask_kwargs = {"b": B, "h": H, "w": W, "m": M}
            expand_kwargs = {"b": B, "h": H, "w": W, "m": M, "d": D}
        elif self.dims_to_pool == DimsToPool.SPATIAL:
            reduction_args = f"(b t m) (h w) d"
            reduction_mask_args = "(b t m) (h w)"
            pre_expand_args = "(b t m) d"
            expand_args = "b t m d"
            expand_mask_kwargs = {"b": B, "t": T, "m": M}
            expand_kwargs = {"b": B, "t": T, "m": M, "d": D}
            # Next do Modality and Temporal
            # Then do All
        elif self.dims_to_pool == DimsToPool.MODALITY_TEMPORAL:
            reduction_args = f"(b h w ) (t m) d"
            reduction_mask_args = "(b h w ) (t m)"
            pre_expand_args = "(b h w) d"
            expand_args = "b h w d"
            expand_mask_kwargs = {"b": B, "h": H, "w": W}
            expand_kwargs = {"b": B, "h": H, "w": W, "d": D}
        elif self.dims_to_pool == DimsToPool.ALL:
            reduction_args = f"b (h w t m)  d"
            reduction_mask_args = "b (h w t m)"
            pre_expand_args = "(b n) d"
            expand_args = "b n d"
            expand_mask_kwargs = {"b": B, "n": 1}
            expand_kwargs = {"b": B, "n": 1, "d": D}
        else:
            raise ValueError(f"Invalid dimensions to pool options: {self.dims_to_pool}")
        pre_expand_mask_args = pre_expand_args.replace(" d", "")
        expand_mask_args = expand_args.replace(" d", "")
        return reduction_args, reduction_mask_args, pre_expand_args, pre_expand_mask_args, expand_args, expand_mask_args, expand_mask_kwargs, expand_kwargs

    def apply_attn_pooling(self, spatial_tokens: torch.Tensor, spatial_masks: torch.Tensor) -> dict[str, torch.Tensor]:
        """Attentive pool the tokens across the dimensions specified in self.dims_to_pool."""
        reduction_args, reduction_mask_args, pre_expand_args, pre_expand_mask_args, expand_args, expand_mask_args, expand_mask_kwargs, expand_kwargs = self._get_reduce_and_expand_args(spatial_tokens.shape)
        # Here is where I pick which dimensions to collapse out of modality, time, and space
        spatial_tokens = rearrange(spatial_tokens, f"b h w t m d -> {reduction_args}")

        spatial_masks = rearrange(spatial_masks, f"b h w t m -> {reduction_mask_args}")
        # print the unique values of the masks
        logger.info(f"unique values of the masks: {torch.unique(spatial_masks)}")
        pooled_attn_mask = spatial_masks == MaskValue.ONLINE_ENCODER.value
        # Do I potentially need to filter out tokens that have no online marked modalities? Maybe not because we will just disgard those
        logger.info(f"shape of spatial tokens before pooling: {spatial_tokens.shape}")
        pooled_tokens = self.attn_pool(spatial_tokens, pooled_attn_mask)
        logger.info(f"shape of pooled tokens: {pooled_tokens.shape}")
        pooled_tokens = rearrange(pooled_tokens, f"{pre_expand_args} -> {expand_args}", **expand_kwargs)
        # for spatial_masks if any in the modality dimension is online encode, set the token to online encoder only
        # otherwise set to Missing Value
        online_encoder_only_mask = (spatial_masks == MaskValue.ONLINE_ENCODER.value).any(dim=-1)
        pooled_attn_mask = torch.where(online_encoder_only_mask, MaskValue.ONLINE_ENCODER.value, MaskValue.MISSING.value)

        pooled_attn_mask = rearrange(pooled_attn_mask, f"{pre_expand_mask_args} -> {expand_mask_args}", **expand_mask_kwargs)
        # TODO: Update names so they make sense for all the different options
        pooled_dict = {
            "modality_pooled_tokens": pooled_tokens,
            "modality_pooled_masks": pooled_attn_mask,
        }
        return pooled_dict


    def forward(
        self,
        x: MaskedHeliosSample,
        patch_size: int,
        input_res: int = BASE_GSD,
        token_exit_cfg: dict | None = None,
        always_pass_none_mask_to_transformer: bool = False,
    ) -> tuple[TokensAndMasks, torch.Tensor]:
        """Process masked input samples into token representations.

        Args:
            x: Masked input sample containing the data to be encoded
            patch_size: Size of patches to divide the input into
            input_res: Resolution of the input data
            token_exit_cfg: Configuration for token exit
            always_pass_none_mask_to_transformer: Whether to always pass None as the mask to the transformer, this enables torch based flash attention

        Returns:
            TokensAndMasks containing the encoded representations and their masks
        """
        # TODO: Add step to validate the exit config is valid
        patchified_tokens_and_masks = self.patch_embeddings.forward(x, patch_size)
        if token_exit_cfg is None or any(
            [exit_depth > 0 for exit_depth in token_exit_cfg.values()]
        ):
            patchified_tokens_and_masks = self.apply_attn(
                x=patchified_tokens_and_masks,
                timestamps=x.timestamps,
                patch_size=patch_size,
                input_res=input_res,
                token_exit_cfg=token_exit_cfg,
                always_pass_none_mask_to_transformer=always_pass_none_mask_to_transformer,
            )

        ## Extra code for modality pooling

        spatial_tokens, spatial_masks = self.stack_spatial_modalities_and_masks(patchified_tokens_and_masks)
        pooled_dict = self.apply_attn_pooling(spatial_tokens, spatial_masks)
        output = TokensAndMasks(**patchified_tokens_and_masks)
        return output, self.project_and_aggregate(output), pooled_dict

@dataclass
class EncoderAttnPoolConfig(EncoderConfig):
    """Configuration for the EncoderAttnPool."""
    dims_to_pool: DimsToPool = DimsToPool.MODALITY
    num_queries: int = 1
    attn_pool_mlp_ratio: float | None = None
    def build(self) -> "EncoderAttnPool":
        """Build the encoder."""
        self.validate()
        kwargs = self.as_dict(exclude_none=True, recurse=False)
        # supported_modality_names is replaced by supported_modalities
        kwargs.pop("supported_modality_names")
        kwargs["supported_modalities"] = self.supported_modalities
        logger.info(f"Encoder kwargs: {kwargs}")
        return EncoderAttnPool(**kwargs)


# Need to make evals optionally use the pooled tokens or not
# V1 Pool modality tokens and use those for evals as wel
# V2 Pool temporally and across modalities and predict
# V3 pool spatially and temporally and acrosss modality and predict