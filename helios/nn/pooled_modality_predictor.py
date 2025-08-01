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
    """Attention Pooling Probe.

    Args:
        in_dim (int): Input feature dimension. Must be divisible by 64.
        out_dim (int): Output dimension (typically num_classes * patch_size * patch_size).

    Attributes:
        query_token (nn.Parameter): Learnable query token for attention pooling.
        num_heads (int): Number of attention heads.
        kv (nn.Linear): Linear layer to produce keys and values.
        linear (nn.Linear): Final linear layer for output logits.
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        """Initialize the attention pooling linear probe."""
        super().__init__()
        assert in_dim % 64 == 0, "in_dim must be divisible by 64"
        self.query_token: nn.Parameter = nn.Parameter(torch.empty(in_dim))
        self.num_heads: int = in_dim // 64
        self.kv: nn.Linear = nn.Linear(in_dim, in_dim * 2)
        self.linear: nn.Linear = nn.Linear(in_dim, out_dim)
        self.init_weights()

    def init_weights(self) -> None:
        """Initialize weights for the probe."""
        nn.init.trunc_normal_(self.query_token, std=0.02)
        nn.init.trunc_normal_(self.kv.weight, std=0.02)
        nn.init.zeros_(self.kv.bias)
        nn.init.trunc_normal_(self.linear.weight, std=0.02)
        nn.init.zeros_(self.linear.bias)

    def forward(self, feat_tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Forward pass for attention pooling linear probe.
        """

        # B, H, W, N, D = feat_tokens.shape
        # feat_tokens = rearrange(feat_tokens, "b h w n d -> (b h w) n d")
        logger.info(f"shape of feat_tokens: {feat_tokens.shape}")
        collapsed_dim , N, D = feat_tokens.shape
        q = self.query_token.expand(collapsed_dim, 1, -1)
        q = q.reshape(
            collapsed_dim, 1, self.num_heads, D // self.num_heads
        )  # [B, 1, head, D_head]
        q = rearrange(q, "b h n d -> b n h d")
        # log the dtype of kv weights and the feat_tokens
        logger.info(f"dtype of kv weights: {self.kv.weight.dtype}")
        logger.info(f"dtype of feat_tokens: {feat_tokens.dtype}")
        # convert feat_tokens to dto
        # why is this hack needed?
        feat_tokens = feat_tokens.to(self.kv.weight.dtype)
        kv = self.kv(feat_tokens).reshape(
            collapsed_dim, N, 2, self.num_heads, D // self.num_heads
        )  # [B, N, 2, head, D_head]
        kv = rearrange(kv, "b n two h d -> two b h n d")
        k, v = torch.unbind(kv, dim=0)  # 2 * [B, head, N, D_head]
        logger.info(f"shape of k: {k.shape}")
        logger.info(f"shape of v: {v.shape}")
        logger.info(f"shape of q: {q.shape}")
        # mask shape
        logger.info(f"shape of mask: {mask.shape}")
        if mask is not None:
            mask = mask[:, None, None].repeat((1, self.num_heads, 1, 1))
        logger.info(f"shape of mask: {mask.shape}")

        # True indicates that the token should take part in attention
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)  # [B, head, 1, D_head]
        x = rearrange(x, "b h 1 d -> b (h d)")
        return x

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
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.attn_pool = AttnPool(self.embedding_size, self.embedding_size)

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

        pooled_tokens = rearrange(pooled_dict["modality_pooled_tokens"], "b ... d -> b (...) d")
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
    def __init__(self, dims_to_pool: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.attn_pool = AttnPool(self.embedding_size, self.embedding_size)

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