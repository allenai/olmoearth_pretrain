"""Alternate Encoder Predictor pairs that allow for tokens to be pooled at different stages of the model and then the decoding to happen from the pooled tokens."""

import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import torch
from einops import rearrange
from torch import Tensor

from helios.data.constants import BASE_GSD, Modality
from helios.nn.attention import AttnPool
from helios.nn.flexihelios import (
    Encoder,
    EncoderConfig,
    PredictorBase,
    PredictorConfig,
    TokensAndMasks,
    get_modalities_to_process,
    return_modalities_from_dict,
)
from helios.nn.utils import get_cumulative_sequence_lengths
from helios.train.masking import MaskedHeliosSample, MaskValue

logger = logging.getLogger(__name__)


class DimsToPool(StrEnum):
    """Dimensions to pool over."""

    MODALITY = "modality"  # 1
    TEMPORAL = "temporal"  # 2
    SPATIAL = "spatial"
    MODALITY_TEMPORAL = "modality_temporal"  # 3
    ALL = "all"  # 4


class EncodeEarlyAttnPool(Encoder):
    """Encoder that pools the tokens across modalities."""

    def __init__(
        self,
        dims_to_pool: str,
        attn_pool_mlp_ratio: float | None = None,
        num_queries: int = 1,
        num_pre_modality_pooling_layers: int = 0,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize the EncodeEarlyAttnPool."""
        super().__init__(*args, **kwargs)
        self.attn_pool = AttnPool(
            self.embedding_size,
            self.embedding_size,
            mlp_ratio=attn_pool_mlp_ratio,
            num_queries=num_queries,
        )
        self.num_pre_modality_pooling_layers = num_pre_modality_pooling_layers

        self.dims_to_pool = dims_to_pool
        if self.use_flash_attn:
            raise NotImplementedError("Flash attn not implemented")

    def _get_reduce_and_expand_args(
        self, shape: tuple[int, ...]
    ) -> tuple[str, str, str, str, str, str, dict[str, int], dict[str, int]]:
        """Get the reduction and expansion arguments for the dimensions to pool."""
        B, H, W, T, M, D = shape
        # Make a reduction args and expand args for each dim pooling type
        if self.dims_to_pool == DimsToPool.MODALITY:
            reduction_args = "(b h w t) m d"
            reduction_mask_args = "(b h w t) m"
            pre_expand_args = "(b h w t) d"
            expand_args = "b h w t d"
            expand_mask_kwargs = {"b": B, "h": H, "w": W, "t": T}
            expand_kwargs = {"b": B, "h": H, "w": W, "t": T, "d": D}
        elif self.dims_to_pool == DimsToPool.TEMPORAL:
            reduction_args = "(b h w m) t d"
            reduction_mask_args = "(b h w m) t"
            pre_expand_args = "(b h w m) d"
            expand_args = "b h w m d"
            expand_mask_kwargs = {"b": B, "h": H, "w": W, "m": M}
            expand_kwargs = {"b": B, "h": H, "w": W, "m": M, "d": D}
        elif self.dims_to_pool == DimsToPool.SPATIAL:
            reduction_args = "(b t m) (h w) d"
            reduction_mask_args = "(b t m) (h w)"
            pre_expand_args = "(b t m) d"
            expand_args = "b t m d"
            expand_mask_kwargs = {"b": B, "t": T, "m": M}
            expand_kwargs = {"b": B, "t": T, "m": M, "d": D}
            # Next do Modality and Temporal
            # Then do All
        elif self.dims_to_pool == DimsToPool.MODALITY_TEMPORAL:
            reduction_args = "(b h w ) (t m) d"
            reduction_mask_args = "(b h w ) (t m)"
            pre_expand_args = "(b h w) d"
            expand_args = "b h w d"
            expand_mask_kwargs = {"b": B, "h": H, "w": W}
            expand_kwargs = {"b": B, "h": H, "w": W, "d": D}
        elif self.dims_to_pool == DimsToPool.ALL:
            reduction_args = "b (h w t m)  d"
            reduction_mask_args = "b (h w t m)"
            pre_expand_args = "(b n) d"
            expand_args = "b n d"
            expand_mask_kwargs = {"b": B, "n": 1}
            expand_kwargs = {"b": B, "n": 1, "d": D}
        else:
            raise ValueError(f"Invalid dimensions to pool options: {self.dims_to_pool}")
        pre_expand_mask_args = pre_expand_args.replace(" d", "")
        expand_mask_args = expand_args.replace(" d", "")
        return (
            reduction_args,
            reduction_mask_args,
            pre_expand_args,
            pre_expand_mask_args,
            expand_args,
            expand_mask_args,
            expand_mask_kwargs,
            expand_kwargs,
        )

    def apply_attn_pooling(
        self, spatial_tokens: torch.Tensor, spatial_masks: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Attentive pool the tokens across the dimensions specified in self.dims_to_pool."""
        (
            reduction_args,
            reduction_mask_args,
            pre_expand_args,
            pre_expand_mask_args,
            expand_args,
            expand_mask_args,
            expand_mask_kwargs,
            expand_kwargs,
        ) = self._get_reduce_and_expand_args(spatial_tokens.shape)
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
        pooled_tokens = rearrange(
            pooled_tokens, f"{pre_expand_args} -> {expand_args}", **expand_kwargs
        )
        # for spatial_masks if any in the modality dimension is online encode, set the token to online encoder only
        # otherwise set to Missing Value
        online_encoder_only_mask = (
            spatial_masks == MaskValue.ONLINE_ENCODER.value
        ).any(dim=-1)
        pooled_attn_mask = torch.where(
            online_encoder_only_mask,
            MaskValue.ONLINE_ENCODER.value,
            MaskValue.MISSING.value,
        )

        pooled_attn_mask = rearrange(
            pooled_attn_mask,
            f"{pre_expand_mask_args} -> {expand_mask_args}",
            **expand_mask_kwargs,
        )
        # TODO: Update names so they make sense for all the different options
        pooled_tokens_and_masks = {
            "modality_pooled_tokens": pooled_tokens,
            "modality_pooled_masks": pooled_attn_mask,
        }
        return pooled_tokens_and_masks

    def collapse_and_combine_hwtc_pooled_tokens(
        self, x: dict[str, Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Collapse and combine the pooled tokens and masks."""
        pooled_tokens = x["modality_pooled_tokens"]
        pooled_masks = x["modality_pooled_masks"]
        pooled_tokens = rearrange(pooled_tokens, "b ... d -> b (...) d")
        pooled_masks = rearrange(pooled_masks, "b ...  -> b (...) ")
        return pooled_tokens, pooled_masks

    def reshape_pooled_tokens(
        self, pooled_tokens: torch.Tensor, pooled_dims: tuple[int, ...]
    ) -> torch.Tensor:
        """Reshape the pooled tokens to the dimensions specified in pooled_dims."""
        b = pooled_tokens.shape[0]
        middle_dims = pooled_dims[1:-1]
        d = pooled_tokens.shape[-1]
        tokens_reshaped = pooled_tokens.view(b, *middle_dims, d)
        return tokens_reshaped

    @staticmethod
    def remove_masked_tokens(
        x: Tensor, mask: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Remove masked tokens from the tokens and masks.

        Implementation from https://stackoverflow.com/a/68621610/2332296

        On Input:
        0 means this token should be removed
        1 means this token should be kept

        Args:
            x: Tokens to remove masked tokens from
            mask: Mask to remove masked tokens from

        Returns:
            tokens: [B, T, D]
            indices: [B, T]
            updated_mask: [B, T]
            seqlens: [B]
            max_length: [1]
            where T is the max number of unmasked tokens for an instance
        """
        # log the shape of x and mask
        logger.info(f"remove masked tokens shape of x: {x.shape}")
        logger.info(f"remove masked tokens shape of mask: {mask.shape}")
        sorted_mask, indices = torch.sort(mask, dim=1, descending=True, stable=True)
        # Now all the places where we want to keep the token are at the front of the tensor
        x = x.gather(1, indices[:, :, None].expand_as(x))
        # Now all tokens that should be kept are first in the tensor

        # set masked values to 0 (not really necessary since we'll ignore them anyway)
        x = x * sorted_mask.unsqueeze(-1)

        # cut off to the length of the longest sequence
        seq_lengths = sorted_mask.sum(-1)
        max_length = seq_lengths.max()
        x = x[:, :max_length]
        # New mask chopped to the longest sequence
        updated_mask = sorted_mask[:, :max_length]

        return x, indices, updated_mask, seq_lengths, max_length

    def apply_unpooled_attn(
        self,
        tokens_and_masks_dict: dict[str, Tensor],
        modalities_to_dims_dict: dict,
        exit_ids_seq: Tensor | None = None,
        exited_tokens: Tensor | None = None,
        always_pass_none_mask_to_transformer: bool = False,
    ) -> dict[str, Tensor]:
        """Apply the attention to the tokens and masks."""
        tokens, mask = self.collapse_and_combine_hwtc(tokens_and_masks_dict)

        bool_mask = mask == MaskValue.ONLINE_ENCODER.value

        tokens, indices, new_mask, seq_lengths, max_seqlen = self.remove_masked_tokens(
            tokens, bool_mask
        )
        if exit_ids_seq is not None:
            exit_ids_seq, _, _, _, _ = self.remove_masked_tokens(
                exit_ids_seq, bool_mask
            )
            # still linear projections
            exited_tokens, _, _, _, _ = self.remove_masked_tokens(
                exited_tokens, bool_mask
            )
        cu_seqlens = get_cumulative_sequence_lengths(seq_lengths)
        # Pack x tokens
        if self.use_flash_attn:
            og_shape = tokens.shape
            tokens = self.pack_tokens(tokens, new_mask)

        attn_mask = self.get_attn_or_none_mask(
            new_mask, always_pass_none_mask_to_transformer
        )
        # Apply attn with varying encoder depths
        for i_blk, blk in enumerate(self.blocks):
            if i_blk == self.num_pre_modality_pooling_layers:
                break
            logger.debug(f"i_blk pre-modality pooling: {i_blk}")
            # Skip the zeroth block because we want to use the exited tokens that don't have encodings as this allows trivial solution of predicting the shared encodings
            if (exit_ids_seq is not None) and (i_blk > 0):
                # this should only ever be called by the target encoder,
                # in a torch.no_grad context
                assert exited_tokens is not None
                # If a token should exit, then we update the exit token with the current token at the same position
                exited_tokens = torch.where(
                    condition=(exit_ids_seq == i_blk),
                    input=tokens,
                    other=exited_tokens,
                )
            # we take the inverse of the mask because a value
            # of True indicates the value *should* take part in
            # attention
            # WARNING: THIS MAY CHANGE DEPENDING ON THE ATTENTION IMPLEMENTATION
            tokens = blk(
                x=tokens,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                # we will have to specify k and q lens for cross attention
                attn_mask=attn_mask,
            )

        if self.use_flash_attn:
            tokens = self.unpack_tokens(tokens, new_mask, og_shape)

        if exit_ids_seq is not None:
            # this should only ever be called by the target encoder,
            # in a torch.no_grad context
            assert exited_tokens is not None
            # full depth
            # IMPORTANT: write this to x
            tokens = torch.where(
                condition=(exit_ids_seq == (i_blk + 1)),  # 2 for full depth
                input=tokens,
                other=exited_tokens,
            )
        # we don't care about the mask returned by add_removed_tokens, since we will
        # just use the original, unclipped mask here
        tokens, _ = self.add_removed_tokens(tokens, indices, new_mask)
        tokens_per_modality_dict = self.split_and_expand_per_modality(
            tokens, modalities_to_dims_dict
        )
        return tokens_per_modality_dict

    def apply_attn(
        self,
        x: dict[str, Tensor],
        timestamps: Tensor,
        patch_size: int,
        input_res: int,
        token_exit_cfg: dict[str, int] | None = None,
        always_pass_none_mask_to_transformer: bool = False,
    ) -> dict[str, Tensor]:
        """Apply the attention to the tokens and masks."""
        tokens_only_dict, original_masks_dict, pre_pooled_modality_to_dims_dict = (
            self.split_tokens_masks_and_dims(x)
        )
        exit_ids_seq = self.create_exit_seqs(
            tokens_only_dict, original_masks_dict, token_exit_cfg
        )
        # exited tokens are just the linear projection
        exited_tokens, _ = self.collapse_and_combine_hwtc(x)

        tokens_dict = self.composite_encodings.forward(
            tokens_only_dict,
            timestamps,
            patch_size,
            input_res,
        )
        tokens_dict.update(original_masks_dict)

        # TODO: token exit config isn't really meant to be used here but is no-op so leaving it in
        tokens_dict = self.apply_unpooled_attn(
            tokens_dict,
            pre_pooled_modality_to_dims_dict,
            exit_ids_seq,
            exited_tokens,
            always_pass_none_mask_to_transformer,
        )
        # update the tokens_dict with the original masks
        tokens_dict.update(original_masks_dict)
        logger.info(f"tokens_dict keys: {tokens_dict.keys()}")
        spatial_tokens, spatial_masks = self.stack_spatial_modalities_and_masks(
            tokens_dict
        )

        tokens_dict = self.apply_attn_pooling(spatial_tokens, spatial_masks)
        pooled_dims = tokens_dict["modality_pooled_tokens"].shape
        original_pooled_masks = tokens_dict["modality_pooled_masks"]
        tokens, mask = self.collapse_and_combine_hwtc_pooled_tokens(tokens_dict)
        bool_mask = mask == MaskValue.ONLINE_ENCODER.value

        tokens, indices, new_mask, seq_lengths, max_seqlen = self.remove_masked_tokens(
            tokens, bool_mask
        )
        if exit_ids_seq is not None:
            exit_ids_seq, _, _, _, _ = self.remove_masked_tokens(
                exit_ids_seq, bool_mask
            )
            # still linear projections
            exited_tokens, _, _, _, _ = self.remove_masked_tokens(
                exited_tokens, bool_mask
            )
        cu_seqlens = get_cumulative_sequence_lengths(seq_lengths)

        attn_mask = self.get_attn_or_none_mask(
            new_mask, always_pass_none_mask_to_transformer
        )
        # Apply attn with varying encoder depths
        for i_blk, blk in enumerate(self.blocks):
            if i_blk < self.num_pre_modality_pooling_layers:
                # skip the pre-modality pooling layer attention blocks
                continue
            logger.debug(f"i_blk post-modality pooling: {i_blk}")
            # Skip the zeroth block because we want to use the exited tokens that don't have encodings as this allows trivial solution of predicting the shared encodings
            if (exit_ids_seq is not None) and (i_blk > 0):
                # this should only ever be called by the target encoder,
                # in a torch.no_grad context
                assert exited_tokens is not None
                # If a token should exit, then we update the exit token with the current token at the same position
                exited_tokens = torch.where(
                    condition=(exit_ids_seq == i_blk),
                    input=tokens,
                    other=exited_tokens,
                )
            # we take the inverse of the mask because a value
            # of True indicates the value *should* take part in
            # attention
            # WARNING: THIS MAY CHANGE DEPENDING ON THE ATTENTION IMPLEMENTATION
            tokens = blk(
                x=tokens,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                # we will have to specify k and q lens for cross attention
                attn_mask=attn_mask,
            )

        if exit_ids_seq is not None:
            # this should only ever be called by the target encoder,
            # in a torch.no_grad context
            assert exited_tokens is not None
            # full depth
            # IMPORTANT: write this to x
            tokens = torch.where(
                condition=(exit_ids_seq == (i_blk + 1)),  # 2 for full depth
                input=tokens,
                other=exited_tokens,
            )
        # we apply the norm before we add the removed tokens,
        # so that the norm is only computed against "real" tokens
        tokens = self.norm(tokens)
        # we don't care about the mask returned by add_removed_tokens, since we will
        # just use the original, unclipped mask here
        tokens, _ = self.add_removed_tokens(tokens, indices, new_mask)
        out_dict = {}
        out_dict["modality_pooled_tokens"] = self.reshape_pooled_tokens(
            tokens, pooled_dims
        )
        out_dict["modality_pooled_masks"] = original_pooled_masks
        return out_dict

    def forward(
        self,
        x: MaskedHeliosSample,
        patch_size: int,
        input_res: int = BASE_GSD,
        token_exit_cfg: dict | None = None,
        always_pass_none_mask_to_transformer: bool = False,
    ) -> dict[str, Any]:
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
        tokenized_output = TokensAndMasks(**patchified_tokens_and_masks)
        if token_exit_cfg is None or any(
            [exit_depth > 0 for exit_depth in token_exit_cfg.values()]
        ):
            pooled_tokens_and_masks = self.apply_attn(
                x=patchified_tokens_and_masks,
                timestamps=x.timestamps,
                patch_size=patch_size,
                input_res=input_res,
                token_exit_cfg=token_exit_cfg,
                always_pass_none_mask_to_transformer=always_pass_none_mask_to_transformer,
            )
        else:
            pooled_tokens_and_masks = {}

        output_dict: dict[str, Any] = {
            "tokens_and_masks": tokenized_output,
            "project_aggregated": self.project_and_aggregate(tokenized_output),
        }
        if pooled_tokens_and_masks:
            output_dict["pooled_tokens_and_masks"] = pooled_tokens_and_masks

        return output_dict


@dataclass
class EncoderEarlyAttnPoolConfig(EncoderConfig):
    """Configuration for the EncoderAttnPool."""

    dims_to_pool: DimsToPool = DimsToPool.MODALITY
    num_queries: int = 1
    attn_pool_mlp_ratio: float | None = None
    num_pre_modality_pooling_layers: int = 0

    def build(self) -> "EncodeEarlyAttnPool":
        """Build the encoder."""
        self.validate()
        kwargs = self.as_dict(exclude_none=True, recurse=False)
        # supported_modality_names is replaced by supported_modalities
        kwargs.pop("supported_modality_names")
        kwargs["supported_modalities"] = self.supported_modalities
        logger.info(f"Encoder kwargs: {kwargs}")
        return EncodeEarlyAttnPool(**kwargs)


class PooledModalityPredictor(PredictorBase):
    """Predictor that pools the tokens across modalities."""

    def __init__(
        self,
        include_encoder_encodings: bool = True,
        dims_to_pool: str = "modality",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize the PooledModalityPredictor."""
        super().__init__(*args, **kwargs)
        self.include_encoder_encodings = include_encoder_encodings
        self.dims_to_pool = dims_to_pool
        if self.use_flash_attn:
            raise NotImplementedError("Flash attn not implemented")

    def _which_encodings_to_use(self) -> dict[str, bool]:
        # TODO: Not great probably should jsut compute bools that we pass instead of this
        if self.dims_to_pool == DimsToPool.MODALITY:
            return {"use_modality_encodings": False, "use_temporal_encodings": True}
        elif self.dims_to_pool == DimsToPool.TEMPORAL:
            return {"use_modality_encodings": True, "use_temporal_encodings": False}
        elif self.dims_to_pool == DimsToPool.MODALITY_TEMPORAL:
            return {"use_modality_encodings": False, "use_temporal_encodings": False}
        else:
            raise NotImplementedError(
                f"Dims to pool {self.dims_to_pool} not implemented"
            )

    def apply_attn(
        self,
        x: dict[str, Tensor],
        timestamps: Tensor,
        patch_size: int,
        input_res: int,
        pooled_tokens_and_masks: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        """Apply attention to the tokens."""
        logger.warning("Calling apply_attn for PooledModalityPredictor")
        tokens_only_dict, original_masks_dict, modalities_to_dims_dict = (
            self.split_tokens_masks_and_dims(x)
        )
        tokens_dict = self.composite_encodings(
            tokens_only_dict, timestamps, patch_size, input_res
        )
        tokens_dict.update(original_masks_dict)

        pooled_tokens = pooled_tokens_and_masks["modality_pooled_tokens"]
        if self.include_encoder_encodings:
            encoding_kwargs = self._which_encodings_to_use()
            logger.info(f"encoding_kwargs: {encoding_kwargs}")
            pooled_tokens = self.composite_encodings._apply_encodings_per_modality(
                Modality.SENTINEL2_L2A.name,
                pooled_tokens,
                timestamps,
                patch_size,
                input_res,
                **encoding_kwargs,
            )
        pooled_tokens = rearrange(pooled_tokens, "b ... d -> b (...) d")
        pooled_attn_mask = rearrange(
            pooled_tokens_and_masks["modality_pooled_masks"], "b ... -> b (...)"
        )

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
            raise NotImplementedError("Flash attn not implemented")

        for blk in self.blocks:
            # note that we are not taking the inverse of the mask, since split_x_y gives us
            # true values for values we want to take part in attention
            tokens_to_decode = blk(
                x=tokens_to_decode,
                y=pooled_tokens,
                attn_mask=(
                    pooled_attn_mask.bool() if not self.use_flash_attn else None
                ),  # only for flash attn though this should not be left in
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
        pooled_tokens_and_masks: dict[str, Tensor],
        timestamps: Tensor,
        patch_size: int,
        input_res: int = BASE_GSD,
    ) -> TokensAndMasks:
        """Generate predictions from encoded token representations.

        Args:
            x: TokensAndMasks containing the masks to use to make decodings.
                These tokens are discarded in this function - only the masks are used
            pooled_tokens_and_masks: Dictionary containing the pooled tokens and their masks
            timestamps: Timestamps of the tokens
            patch_size: Patch size of the tokens
            input_res: Input resolution of the tokens
        Returns:
            TokensAndMasks containing the predicted tokens and their masks
        """
        # Apply Input Norms and encoder to decoder embeds to each modality
        available_modalities = x.modalities
        modalities_to_process = get_modalities_to_process(
            available_modalities, self.supported_modality_names
        )

        # Apply input norma nd projection on pooled tokens
        pooled_tokens = pooled_tokens_and_masks["modality_pooled_tokens"]
        pooled_tokens = self.input_norm(pooled_tokens)
        pooled_tokens = self.encoder_to_decoder_embed(pooled_tokens)
        pooled_tokens_and_masks["modality_pooled_tokens"] = pooled_tokens

        # Prepare the Learnable Masked Outputs on the original Unpooled Tokens
        decoder_emedded_dict = x.as_dict(return_none=False)
        tokens_only_dict = self.add_masks(decoder_emedded_dict)
        decoder_emedded_dict.update(tokens_only_dict)
        tokens_and_masks = self.apply_attn(
            decoder_emedded_dict,
            timestamps,
            patch_size,
            input_res,
            pooled_tokens_and_masks=pooled_tokens_and_masks,
        )

        # Project and Normalize Output Tokens
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
class PooledModalityPredictorConfig(PredictorConfig):
    """Configuration for the PooledModalityPredictor."""

    include_encoder_encodings: bool = True
    dims_to_pool: DimsToPool = DimsToPool.MODALITY

    def build(self) -> "PooledModalityPredictor":
        """Build the predictor."""
        self.validate()
        kwargs = self.as_dict(exclude_none=True, recurse=False)
        # supported_modality_names is replaced by supported_modalities
        kwargs.pop("supported_modality_names")
        kwargs["supported_modalities"] = self.supported_modalities
        logger.info(f"Predictor kwargs: {kwargs}")
        return PooledModalityPredictor(**kwargs)
