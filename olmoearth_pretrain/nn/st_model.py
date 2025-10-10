"""Model that performs spatial attention and temporal attention separately."""

import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch
from einops import rearrange, repeat
from olmo_core.config import Config
from torch import Tensor, nn
from torch.distributed.fsdp import fully_shard, register_fsdp_forward_method

from olmoearth_pretrain.data.constants import Modality, ModalitySpec
from olmoearth_pretrain.dataset.utils import get_modality_specs_from_names
from olmoearth_pretrain.nn.attention import Block
from olmoearth_pretrain.nn.flexi_vit import (
    BASE_GSD,
    CompositeEncodings,
    MultiModalPatchEmbeddings,
    ProjectAndAggregate,
    TokensAndMasks,
    get_modalities_to_process,
    return_modalities_from_dict,
)
from olmoearth_pretrain.train.masking import MaskedOlmoEarthSample, MaskValue

logger = logging.getLogger(__name__)


class AttentionMode(Enum):
    """Mode to perform attention."""

    FULL = 0
    SPATIAL = 1
    TEMPORAL = 2
    WINDOWED = 3


class STBase(nn.Module):
    """STBase is a base class for ST models."""

    cross_attn: bool = False

    def __init__(
        self,
        embedding_size: int,
        max_sequence_length: int,
        num_heads: int,
        mlp_ratio: float,
        depth: int,
        drop_path: float,
        supported_modalities: list[ModalitySpec],
        windowed_attention_size: int | None = None,
        learnable_channel_embeddings: bool = True,
        random_channel_embeddings: bool = False,
        last_layer_cross_attn: bool = False,
    ) -> None:
        """Initialize the STBase class."""
        super().__init__()

        self.embedding_size = embedding_size
        self.supported_modalities = supported_modalities
        self.supported_modality_names = [x.name for x in supported_modalities]
        logger.info(f"modalities being used by model: {self.supported_modality_names}")

        self.max_sequence_length = max_sequence_length
        self.windowed_attention_size = windowed_attention_size
        self.learnable_channel_embeddings = learnable_channel_embeddings
        self.random_channel_embeddings = random_channel_embeddings

        self.blocks = nn.ModuleList(
            [
                Block(
                    embedding_size,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=nn.LayerNorm,  # TODO: This should be configurable
                    cross_attn=self.cross_attn
                    or (last_layer_cross_attn and idx == depth - 1),
                    drop_path=drop_path,
                )
                for idx in range(depth)
            ]
        )

        self.composite_encodings = CompositeEncodings(
            embedding_size,
            self.supported_modalities,
            max_sequence_length,
            learnable_channel_embeddings,
            random_channel_embeddings,
        )
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def grab_modality_specific_dims(modality_data: Tensor) -> tuple[int, ...]:
        """Grab the modality specific dimensions from the modality data.

        Assumes [B, ..., C, D]

        Every modality will have a batch dimension, a channel dimension and embedding dimension.

        Args:
            modality_data: Modality data

        Returns:
            Modality specific dimensions
        """
        return modality_data.shape[1:-2] if modality_data.ndim > 3 else ()

    def collapse_and_combine(
        self, x: dict[str, Tensor], mode: AttentionMode, block_idx: int
    ) -> tuple[Tensor, Tensor]:
        """Collapse the tokens and masks into two tensors.

        Args:
            x: the dictionary with tokens and masks.
            mode: what kind of attention to apply.
            block_idx: the block index, used by some attention modes.
        """
        if mode == AttentionMode.FULL:
            return self.collapse_and_combine_full(x)
        elif mode == AttentionMode.SPATIAL:
            return self.collapse_and_combine_spatial(x)
        elif mode == AttentionMode.TEMPORAL:
            return self.collapse_and_combine_temporal(x)
        elif mode == AttentionMode.WINDOWED:
            return self.collapse_and_combine_windowed(x, block_idx)
        # Should not be possible.
        assert False

    def collapse_and_combine_full(self, x: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Collapse the tokens and masks, respectively, into two tensors.

        This is for attention across all tokens in each example.
        """
        tokens, masks = [], []
        available_modalities = return_modalities_from_dict(x)
        modalities_to_process = get_modalities_to_process(
            available_modalities, self.supported_modality_names
        )
        for modality in modalities_to_process:
            masked_modality_name = MaskedOlmoEarthSample.get_masked_modality_name(
                modality
            )
            x_modality = x[modality]
            x_modality_mask = x[masked_modality_name]
            tokens.append(rearrange(x_modality, "b ... d -> b (...) d"))
            masks.append(rearrange(x_modality_mask, "b ... -> b (...)"))
        tokens_tensor = torch.cat(tokens, dim=1)
        masks_tensor = torch.cat(masks, dim=1)

        return tokens_tensor, masks_tensor

    def collapse_and_combine_temporal(
        self, x: dict[str, Tensor]
    ) -> tuple[Tensor, Tensor]:
        """Collapse the tokens and masks, respectively, into two tensors.

        This combines the batch/height/width dimensions so that attention is applied
        temporally but not spatially.
        """
        available_modalities = return_modalities_from_dict(x)
        modalities_to_process = get_modalities_to_process(
            available_modalities, self.supported_modality_names
        )

        # First determine the height and width.
        # We require all modalities that are not static in space to have the same
        # spatial dimensions.
        # The height and width will then be used to pad modalities that are static in
        # space.
        h: int | None = None
        w: int | None = None
        for modality in modalities_to_process:
            x_modality = x[modality]
            if len(x_modality.shape) != 6:
                continue
            cur_h = x_modality.shape[1]
            cur_w = x_modality.shape[2]
            if h is None:
                h = cur_h
                w = cur_w
            elif h != cur_h or w != cur_w:
                raise ValueError(
                    "expected all modalities to have the same spatial dimensions"
                )

        if h is None or w is None:
            raise ValueError("expected at least one spatial modality")

        tokens, masks = [], []
        for modality in modalities_to_process:
            masked_modality_name = MaskedOlmoEarthSample.get_masked_modality_name(
                modality
            )
            x_modality = x[modality]
            x_modality_mask = x[masked_modality_name]

            num_tokens: int

            if len(x_modality.shape) == 3:
                # This is static in space/time, so we duplicate the tokens.
                # Later we will do mean pooling to split it back into one.
                flattened_tokens = repeat(
                    x_modality, "b b_s d -> (b h w) b_s d", h=h, w=w
                )
                flattened_masks = repeat(
                    x_modality_mask, "b b_s -> (b h w) b_s", h=h, w=w
                )

            elif len(x_modality.shape) == 6:
                flattened_tokens = rearrange(
                    x_modality, "b h w ... d -> (b h w) (...) d"
                )
                flattened_masks = rearrange(
                    x_modality_mask, "b h w ... -> (b h w) (...)"
                )

            else:
                raise NotImplementedError(
                    f"not implemented for {len(x_modality.shape)} dimensions"
                )

            num_tokens = flattened_tokens.shape[1]
            logger.debug(f"Modality {modality} has {num_tokens} tokens")
            tokens.append(flattened_tokens)
            masks.append(flattened_masks)

        # Concatenate along temporal (token) dimension.
        tokens_tensor = torch.cat(tokens, dim=1)
        masks_tensor = torch.cat(masks, dim=1)
        return tokens_tensor, masks_tensor

    def collapse_and_combine_spatial(
        self, x: dict[str, Tensor]
    ) -> tuple[Tensor, Tensor]:
        """Collapse the tokens and masks, respectively, into two tensors.

        This combines the batch/time dimensions so that attention is applied spatially
        but not temporally.
        """
        available_modalities = return_modalities_from_dict(x)
        modalities_to_process = get_modalities_to_process(
            available_modalities, self.supported_modality_names
        )

        # First determine the height and width.
        # We require all modalities that are not static in space to have the same
        # spatial dimensions.
        # The height and width will then be used to pad modalities that are static in
        # space.
        h: int | None = None
        w: int | None = None
        for modality in modalities_to_process:
            x_modality = x[modality]
            if len(x_modality.shape) != 6:
                continue
            cur_h = x_modality.shape[1]
            cur_w = x_modality.shape[2]
            if h is None:
                h = cur_h
                w = cur_w
            elif h != cur_h or w != cur_w:
                raise ValueError(
                    "expected all modalities to have the same spatial dimensions"
                )

        if h is None or w is None:
            raise ValueError("expected at least one spatial modality")

        tokens, masks = [], []
        for modality in modalities_to_process:
            masked_modality_name = MaskedOlmoEarthSample.get_masked_modality_name(
                modality
            )
            x_modality = x[modality]
            x_modality_mask = x[masked_modality_name]

            if len(x_modality.shape) == 3:
                # This is static in space/time, so we pad the tokens.
                # We collapse any band sets so attention is across all the tokens for
                # this modality.
                b_s = x_modality.shape[1]
                amount_to_pad = h * w - b_s
                flattened_tokens = torch.nn.functional.pad(
                    x_modality, (0, 0, 0, amount_to_pad)
                )
                flattened_masks = torch.nn.functional.pad(
                    x_modality_mask, (0, amount_to_pad), value=MaskValue.MISSING.value
                )

            elif len(x_modality.shape) == 6:
                flattened_tokens = rearrange(
                    x_modality, "b h w ... d -> (b ...) (h w) d"
                )
                flattened_masks = rearrange(
                    x_modality_mask, "b h w ... -> (b ...) (h w)"
                )

            else:
                raise NotImplementedError(
                    f"not implemented for {len(x_modality.shape)} dimensions"
                )

            num_tokens = flattened_tokens.shape[0]
            logger.debug(f"Modality {modality} has {num_tokens} tokens")
            tokens.append(flattened_tokens)
            masks.append(flattened_masks)

        # Concatenate along temporal (batch) dimension.
        tokens_tensor = torch.cat(tokens, dim=0)
        masks_tensor = torch.cat(masks, dim=0)
        return tokens_tensor, masks_tensor

    def collapse_and_combine_windowed(
        self, x: dict[str, Tensor], block_idx: int
    ) -> tuple[Tensor, Tensor]:
        """Collapse the tokens and masks, respectively, into two tensors.

        This applies attention that is full along the temporal dimension but windowed
        along the spatial dimension. The size of the windows is controlled by
        windowed_attention_size. The windows used for even blocks will be offset by
        half of the window size.

        Args:
            x: the tokens and masks dictionary.
            block_idx: the index of this block in the transformer.

        Returns:
            the (tokens, masks) tuple.
        """
        size = self.windowed_attention_size
        assert size is not None
        if block_idx % 2 == 0:
            offset_padding = size // 2
        else:
            offset_padding = 0

        available_modalities = return_modalities_from_dict(x)
        modalities_to_process = get_modalities_to_process(
            available_modalities, self.supported_modality_names
        )

        # For each modality, we will compute a tensor like this:
        # (Batch x Window ID) x Number of Tokens x Embedding Size
        # The number of tokens will be the product of the window height and width along
        # with the number of timesteps and the number of band sets.
        tokens, masks = [], []
        for modality in modalities_to_process:
            masked_modality_name = MaskedOlmoEarthSample.get_masked_modality_name(
                modality
            )
            x_modality = x[modality]
            x_modality_mask = x[masked_modality_name]

            if len(x_modality.shape) == 6:
                # First collapse the temporal and band set dimensions.
                cur_tokens = rearrange(x_modality, "b h w ... d -> b (...) d h w")
                cur_masks = rearrange(x_modality_mask, "b h w ... -> b (...) h w")
                # Add the offset padding that shifts the windows to the beginning.
                cur_tokens = torch.nn.functional.pad(
                    cur_tokens, (offset_padding, 0, offset_padding, 0)
                )
                cur_masks = torch.nn.functional.pad(
                    cur_masks,
                    (offset_padding, 0, offset_padding, 0),
                    value=MaskValue.MISSING.value,
                )
                # Add padding to the end to make it multiple of window size.
                w_padding = (-cur_tokens.shape[-1]) % size
                h_padding = (-cur_tokens.shape[-2]) % size
                cur_tokens = torch.nn.functional.pad(
                    cur_tokens, (0, w_padding, 0, h_padding)
                )
                cur_masks = torch.nn.functional.pad(
                    cur_masks,
                    (0, w_padding, 0, h_padding),
                    value=MaskValue.MISSING.value,
                )
                # Now we can split it up into the windows.
                flattened_tokens = rearrange(
                    cur_tokens,
                    "b tbs d (hn hs) (wn ws) -> (b hn wn) (tbs hs ws) d",
                    hs=size,
                    ws=size,
                )
                flattened_masks = rearrange(
                    cur_masks,
                    "b tbs (hn hs) (wn ws) -> (b hn wn) (tbs hs ws)",
                    hs=size,
                    ws=size,
                )

            else:
                raise NotImplementedError(
                    f"not implemented for {len(x_modality.shape)} dimensions"
                )

            num_tokens = flattened_tokens.shape[0]
            logger.debug(f"Modality {modality} has {num_tokens} tokens")
            tokens.append(flattened_tokens)
            masks.append(flattened_masks)

        # Concatenate along the token dimension.
        tokens_tensor = torch.cat(tokens, dim=1)
        masks_tensor = torch.cat(masks, dim=1)
        logger.info(
            f"collapse_and_combine_windowed: end up with {tokens_tensor.shape[0]} batches of {tokens_tensor.shape[1]} tokens"
        )
        return tokens_tensor, masks_tensor

    @staticmethod
    def _construct_einops_pattern(
        spatial_dims: tuple[int, ...],
    ) -> tuple[str, dict[str, int]]:
        """Given a tuple of spatial dimensions (e.g. [B, H, W, T, ...]).

        build (1) an einops rearrange pattern of the form:
            "d -> (dim0) (dim1) (dim2)... d"
        and (2) a dictionary mapping dim0..dimN to the actual sizes.

        This allows reshaping a single-dimensional tensor [D] into
        [B, H, W, T, ..., D] using einops.
        """
        dim_dict = {f"dim{i}": size for i, size in enumerate(spatial_dims)}
        # e.g., "d -> (dim0) (dim1) (dim2) (dim3) d"
        pattern_input = (
            "d -> " + " ".join(f"(dim{i})" for i in range(len(spatial_dims))) + " d"
        )
        return pattern_input, dim_dict

    def split_tokens_masks_and_dims(
        self, x: dict[str, Tensor]
    ) -> tuple[dict[str, Tensor], dict[str, Tensor], dict[str, tuple]]:
        """Split the tokens, masks, and dimensions out into separate dicts."""
        tokens_only_dict = {}
        original_masks_dict = {}
        modalities_to_dims_dict = {}
        # TODO: Should I have a dict like object that has methods that can return a mask or atoken here?
        available_modalities = return_modalities_from_dict(x)
        modalities_to_process = get_modalities_to_process(
            available_modalities, self.supported_modality_names
        )
        for modality in modalities_to_process:
            x_modality = x[modality]
            tokens_only_dict[modality] = x_modality
            modalities_to_dims_dict[modality] = x_modality.shape
            masked_modality_name = MaskedOlmoEarthSample.get_masked_modality_name(
                modality
            )
            original_masks_dict[masked_modality_name] = x[masked_modality_name]
        return tokens_only_dict, original_masks_dict, modalities_to_dims_dict

    def split_and_expand_per_modality(
        self,
        x: dict[str, Tensor],
        modalities_to_dims_dict: dict[str, tuple],
        mode: AttentionMode,
        block_idx: int,
    ) -> dict[str, Tensor]:
        """Split and expand the tokens per modality."""
        if mode == AttentionMode.FULL:
            return self.split_and_expand_per_modality_full(x, modalities_to_dims_dict)
        elif mode == AttentionMode.SPATIAL:
            return self.split_and_expand_per_modality_spatial(
                x, modalities_to_dims_dict
            )
        elif mode == AttentionMode.TEMPORAL:
            return self.split_and_expand_per_modality_temporal(
                x, modalities_to_dims_dict
            )
        elif mode == AttentionMode.WINDOWED:
            assert self.windowed_attention_size is not None
            return self.split_and_expand_per_modality_windowed(
                x, modalities_to_dims_dict, self.windowed_attention_size, block_idx
            )
        # Should not be possible.
        assert False

    @staticmethod
    def split_and_expand_per_modality_full(
        x: Tensor, modalities_to_dims_dict: dict[str, tuple]
    ) -> dict[str, Tensor]:
        """Split and expand the tokens per modality.

        This is for full attention corresponding to collapse_and_combine_full.

        Args:
            x: Tokens to split and expand (b n d)
            modalities_to_dims_dict: Dictionary mapping modalities to their dimensions
        Returns:
            tokens_only_dict: mapping modalities to their tokens
        """
        tokens_only_dict = {}
        tokens_reshaped = 0
        for modality, dims in modalities_to_dims_dict.items():
            # Skip batch (first) and embedding (last) dimensions
            middle_dims = dims[1:-1]
            num_tokens_for_modality = math.prod(middle_dims)

            # Extract tokens for this modality (b n d)
            modality_tokens = x[
                :, tokens_reshaped : tokens_reshaped + num_tokens_for_modality
            ]

            # TODO: see if there  is a general and clean einops way to do this
            # Reshape to original dimensions (e.g., for 4D spatial dims: b d1 d2 d3 d4 e)
            x_modality = modality_tokens.view(x.shape[0], *middle_dims, x.shape[-1])

            tokens_reshaped += num_tokens_for_modality
            tokens_only_dict[modality] = x_modality

        return tokens_only_dict

    @staticmethod
    def split_and_expand_per_modality_temporal(
        x: Tensor, modalities_to_dims_dict: dict[str, tuple]
    ) -> dict[str, Tensor]:
        """Split and expand the tokens per modality.

        This is for tokens that were collapsed using collapse_and_combine_temporal (for
        doing temporal attention only).

        Args:
            x: Tokens to split and expand (b*h*w t*b_s d)
            modalities_to_dims_dict: Dictionary mapping modalities to their dimensions
        Returns:
            tokens_only_dict: mapping modalities to their tokens
        """
        tokens_only_dict = {}
        tokens_reshaped = 0
        for modality, dims in modalities_to_dims_dict.items():
            if len(dims) == 3:
                batch, b_s, _ = dims
                num_tokens_for_modality = b_s
                modality_tokens = x[
                    :, tokens_reshaped : tokens_reshaped + num_tokens_for_modality, :
                ]
                # We pool the tokens across space back down into a single one.
                modality_tokens = rearrange(
                    modality_tokens, "(b hw) b_s d -> b hw b_s d", b=batch
                )
                x_modality = torch.mean(modality_tokens, dim=1)

            elif len(dims) == 6:
                batch, h, w, t, b_s, _ = dims

                # Extract tokens for this modality (b*h*w t*b_s d).
                # Modalities are stacked on the temporal (token) axis.
                num_tokens_for_modality = t * b_s
                modality_tokens = x[
                    :, tokens_reshaped : tokens_reshaped + num_tokens_for_modality, :
                ]

                # Reshape to original dimensions.
                x_modality = rearrange(
                    modality_tokens,
                    "(b h w) (t b_s) d -> b h w t b_s d",
                    b=batch,
                    h=h,
                    w=w,
                    t=t,
                    b_s=b_s,
                )

            else:
                raise NotImplementedError(f"not implemented for {len(dims)} dimensions")

            tokens_reshaped += num_tokens_for_modality
            tokens_only_dict[modality] = x_modality

        assert tokens_reshaped == x.shape[1]

        return tokens_only_dict

    @staticmethod
    def split_and_expand_per_modality_spatial(
        x: Tensor, modalities_to_dims_dict: dict[str, tuple]
    ) -> dict[str, Tensor]:
        """Split and expand the tokens per modality.

        This is for tokens that were collapsed using collapse_and_combine_spatial (for
        doing spatial attention only).

        Args:
            x: Tokens to split and expand (b*t*b_s h*w d)
            modalities_to_dims_dict: Dictionary mapping modalities to their dimensions
        Returns:
            tokens_only_dict: mapping modalities to their tokens
        """
        tokens_only_dict = {}
        tokens_reshaped = 0
        for modality, dims in modalities_to_dims_dict.items():
            if len(dims) == 3:
                # Tokens were stacked along batch dimension, and padded to be h*w.
                # So we just get the first few tokens.
                batch, b_s, _ = dims
                num_tokens_for_modality = batch
                modality_tokens = x[
                    tokens_reshaped : tokens_reshaped + num_tokens_for_modality, :, :
                ]
                x_modality = modality_tokens[:, 0:b_s, :]

            elif len(dims) == 6:
                # Extract tokens for this modality (b*t*b_s h*w d).
                # Modalities are stacked on the temporal axis, which is part of the batch
                # dimension.
                batch, h, w, t, b_s, _ = dims
                num_tokens_for_modality = batch * t * b_s
                modality_tokens = x[
                    tokens_reshaped : tokens_reshaped + num_tokens_for_modality, :, :
                ]

                # Reshape to original dimensions.
                x_modality = rearrange(
                    modality_tokens,
                    "(b t b_s) (h w) d -> b h w t b_s d",
                    b=batch,
                    h=h,
                    w=w,
                    t=t,
                    b_s=b_s,
                )

            tokens_reshaped += num_tokens_for_modality
            tokens_only_dict[modality] = x_modality

        assert tokens_reshaped == x.shape[0]

        return tokens_only_dict

    @staticmethod
    def split_and_expand_per_modality_windowed(
        x: Tensor,
        modalities_to_dims_dict: dict[str, tuple],
        window_size: int,
        block_idx: int,
    ) -> dict[str, Tensor]:
        """Split and expand the tokens per modality.

        This is for tokens that were collapsed using collapse_and_combine_windowed (for
        doing windowed attention).

        Args:
            x: Tokens to split and expand (b*hn*wn t*bs*hs*ws d)
            modalities_to_dims_dict: Dictionary mapping modalities to their dimensions
            window_size: the window size to use.
            block_idx: the block index. Even blocks are shifted so we need to account
                for that when expanding.

        Returns:
            tokens_only_dict: mapping modalities to their tokens
        """
        if block_idx % 2 == 0:
            offset_padding = window_size // 2
        else:
            offset_padding = 0
        tokens_only_dict = {}
        tokens_reshaped = 0
        for modality, dims in modalities_to_dims_dict.items():
            if len(dims) != 6:
                raise NotImplementedError(f"not implemented for {len(dims)} dimensions")

            batch, h, w, t, b_s, _ = dims
            hn = (h + offset_padding + window_size - 1) // window_size
            wn = (w + offset_padding + window_size - 1) // window_size
            # Extract tokens for this modality (b*hn*wn t*bs*hs*ws d).
            # Modalities are stacked on the token axis.
            num_tokens_for_modality = t * b_s * window_size * window_size
            modality_tokens = x[
                :, tokens_reshaped : tokens_reshaped + num_tokens_for_modality, :
            ]
            # Rearrange to padded form.
            modality_tokens = rearrange(
                modality_tokens,
                "(b hn wn) (t bs hs ws) d -> b (hn hs) (wn ws) t bs d",
                b=batch,
                hn=hn,
                wn=wn,
                hs=window_size,
                ws=window_size,
                t=t,
                bs=b_s,
            )
            # Remove beginning padding.
            modality_tokens = modality_tokens[:, offset_padding:, offset_padding:]
            # Remove end padding.
            x_modality = modality_tokens[:, 0:h, 0:w]

            tokens_reshaped += num_tokens_for_modality
            tokens_only_dict[modality] = x_modality

        assert tokens_reshaped == x.shape[1]
        return tokens_only_dict

    def apply_fsdp(self, **fsdp_kwargs: Any) -> None:
        """Apply FSDP to the model."""
        for block in self.blocks:
            block.apply_fsdp(**fsdp_kwargs)

    def apply_compile(self) -> None:
        """Apply torch.compile to the model."""
        for block in self.blocks:
            block.apply_compile()


class STEncoder(STBase):
    """Encoder module that processes masked input samples into token representations."""

    cross_attn: bool = False

    def __init__(
        self,
        embedding_size: int,
        max_patch_size: int,
        min_patch_size: int,
        num_heads: int,
        mlp_ratio: float,
        depth: int,
        drop_path: float,
        supported_modalities: list[ModalitySpec],
        max_sequence_length: int,
        windowed_attention_size: int | None = None,
        learnable_channel_embeddings: bool = True,
        random_channel_embeddings: bool = False,
        num_projection_layers: int = 1,
        aggregate_then_project: bool = True,
        fuse_layers: int | None = None,
        layer_attention_modes: list[AttentionMode] | None = None,
        fuse_using_cross_attn: bool = True,
    ):
        """Initialize the encoder.

        Args:
            embedding_size: Size of token embeddings
            max_patch_size: Maximum patch size for patchification
            min_patch_size: Minimum patch size for patchification
            num_heads: Number of attention heads
            mlp_ratio: Ratio for MLP hidden dimension
            depth: Number of transformer layers
            drop_path: Drop path rate
            supported_modalities: list documenting modalities used in a given model instantiation
            max_sequence_length: Maximum sequence length
            windowed_attention_size: Window size to do windowed attention instead of spatial/temporal attention.
            learnable_channel_embeddings: Whether to use learnable channel embeddings
            random_channel_embeddings: Initialize channel embeddings randomly (zeros if False)
            num_projection_layers: Number of projection layers
            aggregate_then_project: Whether to aggregate then project
            fuse_layers: do spatial attention for the first portion of the model, then do full
                attention for this many layers, and then on the last layer we do cross attention
                to compute a fused representation for each spatial patch.
            layer_attention_modes: directly specify the attention mode to use at each layer.
            fuse_using_cross_attn: fuse using cross attention. If disabled, we perform self-attention and then
                arbitrarily pick one unmasked token at each spatial patch to copy to all the other tokens at
                that patch.
        """
        super().__init__(
            embedding_size=embedding_size,
            depth=depth,
            mlp_ratio=mlp_ratio,
            num_heads=num_heads,
            max_sequence_length=max_sequence_length,
            learnable_channel_embeddings=learnable_channel_embeddings,
            drop_path=drop_path,
            supported_modalities=supported_modalities,
            windowed_attention_size=windowed_attention_size,
            random_channel_embeddings=random_channel_embeddings,
            last_layer_cross_attn=fuse_layers is not None and fuse_using_cross_attn,
        )
        self.min_patch_size = min_patch_size
        self.max_patch_size = max_patch_size
        self.embedding_size = embedding_size
        self.fuse_layers = fuse_layers
        self.layer_attention_modes = layer_attention_modes
        self.fuse_using_cross_attn = fuse_using_cross_attn
        self.patch_embeddings = MultiModalPatchEmbeddings(
            self.supported_modality_names,
            self.max_patch_size,
            self.embedding_size,
        )
        # TODO: add backwards compatibility without the project and aggregate module
        self.project_and_aggregate = ProjectAndAggregate(
            embedding_size=self.embedding_size,
            num_layers=num_projection_layers,
            aggregate_then_project=aggregate_then_project,
        )
        self.norm = nn.LayerNorm(self.embedding_size)

        if self.fuse_layers is not None:
            self.fusing_token = nn.Parameter(torch.zeros(embedding_size))

        self.apply(self._init_weights)

    def create_token_exit_ids(
        self, x: dict[str, Tensor], token_exit_cfg: dict[str, int]
    ) -> dict[str, Tensor]:
        """Create the token exit ids for # of layers of attention for each band group.

        Assumes modality channel groups are in the second to last dimension of the tokens.
        """
        exit_ids_per_modality_dict = {}
        available_modalities = return_modalities_from_dict(x)
        modalities_to_process = get_modalities_to_process(
            available_modalities, self.supported_modality_names
        )
        for modality in modalities_to_process:
            num_exit_layers = token_exit_cfg[modality]
            exit_seq_modality = torch.full_like(x[modality], fill_value=num_exit_layers)
            exit_ids_per_modality_dict[modality] = exit_seq_modality
        return exit_ids_per_modality_dict

    @staticmethod
    def remove_masked_tokens(x: Tensor, mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
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
            where T is the max number of unmasked tokens for an instance
        """
        sorted_mask, indices = torch.sort(mask, dim=1, descending=True, stable=True)
        # Now all the places where we want to keep the token are at the front of the tensor
        x = x.gather(1, indices[:, :, None].expand_as(x))
        # Now all tokens that should be kept are first in the tensor

        # set masked values to 0 (not really necessary since we'll ignore them anyway)
        x = x * sorted_mask.unsqueeze(-1)

        # cut off to the length of the longest sequence
        max_length = sorted_mask.sum(-1).max()
        x = x[:, :max_length]
        # New mask chopped to the longest sequence
        updated_mask = sorted_mask[:, :max_length]

        return x, indices, updated_mask

    @staticmethod
    def add_removed_tokens(
        x: Tensor, indices: Tensor, mask: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Add removed tokens to the tokens and masks.

        Args:
            x: Tokens to add removed tokens to
            indices: Original indices of the masked tokens
            mask: Mask to add removed tokens to

        Returns:
            tokens: Tokens with removed tokens added
            mask: Mask with removed tokens added
        """
        assert x.shape[1] > 0, (
            "x must have at least one token we should not mask all tokens"
        )
        masked_tokens = repeat(
            torch.zeros_like(x[0, 0, :]), "d -> b t d", b=x.shape[0], t=indices.shape[1]
        )
        full_mask = torch.cat(
            (
                mask,
                torch.zeros(
                    (x.shape[0], indices.shape[1] - x.shape[1]),
                    device=x.device,
                    dtype=mask.dtype,
                ),
            ),
            dim=-1,
        )
        # can't set value on leaf variable
        out = masked_tokens.clone()
        # put tokens in full masked tensor (at the first N positions in every row)
        out[full_mask] = x[mask]
        # then move them to their original positions
        out = out.scatter(1, indices[:, :, None].expand_as(out), out)
        full_mask = full_mask.scatter(1, indices.expand_as(full_mask), full_mask)
        # Values that were masked out are not returned but the values that are still there are returned to the original positions
        return out, full_mask

    def create_exit_seqs(
        self,
        tokens_only_dict: dict[str, Tensor],
        mask_only_dict: dict[str, Tensor],
        token_exit_cfg: dict[str, int] | None,
    ) -> tuple[Tensor | None]:
        """Create the exit sequences and tokens."""
        # Check that tokens_only_dict doesn't contain any mask keys
        assert all(not key.endswith("_mask") for key in tokens_only_dict), (
            "tokens_only_dict should not contain mask keys"
        )
        if token_exit_cfg:
            exit_ids_per_modality = self.create_token_exit_ids(
                tokens_only_dict, token_exit_cfg
            )
            exit_ids_per_modality.update(mask_only_dict)
            # Exit ids seqs tells us which layer to exit each token
            exit_ids_seq, _ = self.collapse_and_combine_full(exit_ids_per_modality)
        else:
            exit_ids_seq = None
        return exit_ids_seq

    def copy_first_unmasked_token(
        self, tokens: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """For each batch, find the first unmasked token and copy it to all unmasked positions.

        Args:
            tokens (torch.Tensor): Tensor of shape [B, T, D] with token embeddings.
            mask (torch.Tensor): Tensor of shape [B, T] with 1 for unmasked and 0 for masked tokens.

        Returns:
            torch.Tensor: Updated tokens of shape [B, T, D].
        """
        B, T, D = tokens.shape

        # Get indices of the first unmasked token for each batch
        first_unmasked_idx = (mask == 1).float().cumsum(dim=1)
        first_unmasked_idx[first_unmasked_idx != 1] = (
            0  # only keep the first occurrence
        )
        first_unmasked_idx[first_unmasked_idx == 1] = 1
        idx = first_unmasked_idx.argmax(dim=1)  # shape: [B]

        # Gather the first unmasked tokens
        idx_expanded = idx.view(B, 1, 1).expand(-1, 1, D)  # shape: [B, 1, D]
        first_tokens = torch.gather(tokens, dim=1, index=idx_expanded).squeeze(
            1
        )  # shape: [B, D]

        # Expand to [B, T, D] and mask
        output = tokens.clone()
        output[mask == 1] = first_tokens.unsqueeze(1).expand(-1, T, -1)[mask == 1]

        return output

    def apply_attn(
        self,
        x: dict[str, Tensor],
        timestamps: Tensor,
        patch_size: int,
        input_res: int,
        token_exit_cfg: dict[str, int] | None = None,
    ) -> dict[str, Tensor]:
        """Apply the attention to the tokens and masks."""
        tokens_only_dict, original_masks_dict, modalities_to_dims_dict = (
            self.split_tokens_masks_and_dims(x)
        )
        exit_ids_seq = self.create_exit_seqs(
            tokens_only_dict, original_masks_dict, token_exit_cfg
        )
        exited_tokens = None

        if exit_ids_seq is not None:
            # exited tokens are just the linear projection
            exited_tokens, mask = self.collapse_and_combine_full(x)
            bool_mask = mask == MaskValue.ONLINE_ENCODER.value
            exit_ids_seq, _, _ = self.remove_masked_tokens(exit_ids_seq, bool_mask)
            exited_tokens, _, _ = self.remove_masked_tokens(exited_tokens, bool_mask)

        x = self.composite_encodings.forward(
            tokens_only_dict,
            timestamps,
            patch_size,
            input_res,
        )
        x.update(original_masks_dict)

        # Apply attn with varying encoder depths
        for i_blk, blk in enumerate(self.blocks):
            # Skip the zeroth block because we want to use the exited tokens that don't have encodings as this allows trivial solution of predicting the shared encodings
            if (exit_ids_seq is not None) and (i_blk > 0):
                tokens, _ = self.collapse_and_combine_full(x)
                # this should only ever be called by the target encoder,
                # in a torch.no_grad context
                assert exited_tokens is not None
                # If a token should exit, then we update the exit token with the current token at the same position
                exited_tokens = torch.where(
                    condition=(exit_ids_seq == i_blk),
                    input=tokens,
                    other=exited_tokens,
                )

            # On even blocks, do temporal attention.
            # On odd blocks, do spatial attention.
            # Unless windowed attention is configured.
            do_token_fusing = False
            if self.layer_attention_modes:
                attention_mode = self.layer_attention_modes[i_blk]
                # With fusing, the last layer must be temporal attention.
                if self.fuse_layers is not None and i_blk == len(self.blocks) - 1:
                    if attention_mode != AttentionMode.TEMPORAL:
                        raise ValueError(
                            f"with fusing enabled, the last layer must be temporal attention but got {attention_mode}"
                        )
                    do_token_fusing = True
            elif self.windowed_attention_size is not None:
                attention_mode = AttentionMode.WINDOWED
            elif self.fuse_layers is not None:
                # With fuse_layers:
                # First portion: do spatial attention.
                # For fuse_layers: do full attention.
                # Last layer: do temporal cross attention.
                if i_blk < len(self.blocks) - self.fuse_layers - 1:
                    attention_mode = AttentionMode.SPATIAL
                elif i_blk < len(self.blocks) - 1:
                    attention_mode = AttentionMode.FULL
                else:
                    attention_mode = AttentionMode.TEMPORAL
                    do_token_fusing = True
            elif i_blk % 2 == 0:
                attention_mode = AttentionMode.TEMPORAL
            else:
                attention_mode = AttentionMode.SPATIAL

            logger.debug(f"Layer {i_blk} applying attention mode {attention_mode}")
            x, mask = self.collapse_and_combine(x, attention_mode, i_blk)
            bool_mask = mask == MaskValue.ONLINE_ENCODER.value
            tokens, indices, new_mask = self.remove_masked_tokens(x, bool_mask)

            if do_token_fusing and self.fuse_using_cross_attn:
                # Last layer with fusing enabled, that means we do cross attention to compute
                # per-spatial-patch tokens.
                logger.debug(f"Layer {i_blk} fusing tokens using cross attention")
                attention_batch_size = tokens.shape[0]
                attention_seq_len = tokens.shape[1]
                fuse_x = (
                    self.fusing_token.unsqueeze(0)
                    .unsqueeze(1)
                    .repeat(attention_batch_size, 1, 1)
                )
                # Computed tokens will also be [B, 1, D].
                tokens = blk(x=fuse_x, y=tokens, attn_mask=new_mask)
                # Now expand the tokens to [B, T, D].
                # This is to keep consistent with the expected output format.
                tokens = tokens.expand(-1, attention_seq_len, -1)
            else:
                tokens = blk(x=tokens, y=None, attn_mask=new_mask)

            # Apply normalization on last block.
            if i_blk == len(self.blocks) - 1:
                tokens = self.norm(tokens)

            if do_token_fusing and not self.fuse_using_cross_attn:
                # In this case we arbitrarily pick one of the tokens in each temporal attention batch
                # to replicate to all the other unmasked tokens.
                logger.debug(
                    f"Layer {i_blk} fusing tokens by replicating the first unmasked token"
                )
                tokens = self.copy_first_unmasked_token(tokens, new_mask)

            tokens, _ = self.add_removed_tokens(tokens, indices, new_mask)
            x = self.split_and_expand_per_modality(
                tokens, modalities_to_dims_dict, attention_mode, i_blk
            )
            x.update(original_masks_dict)

        if exit_ids_seq is not None:
            tokens, _ = self.collapse_and_combine_full(x)
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
            x = self.split_and_expand_per_modality_full(tokens, modalities_to_dims_dict)
            x.update(original_masks_dict)

        return x

    def forward(
        self,
        x: MaskedOlmoEarthSample,
        patch_size: int,
        input_res: int = BASE_GSD,
        token_exit_cfg: dict | None = None,
    ) -> tuple[TokensAndMasks, Tensor]:
        """Process masked input samples into token representations.

        Args:
            x: Masked input sample containing the data to be encoded
            patch_size: Size of patches to divide the input into
            input_res: Resolution of the input data
            token_exit_cfg: Configuration for token exit

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
            )
        output = TokensAndMasks(**patchified_tokens_and_masks)
        return output, self.project_and_aggregate(output)

    def apply_fsdp(self, **fsdp_kwargs: Any) -> None:
        """Apply FSDP to the model."""
        super().apply_fsdp(**fsdp_kwargs)
        fully_shard(self.patch_embeddings, **fsdp_kwargs)
        register_fsdp_forward_method(self.patch_embeddings, "forward")
        fully_shard(self, **fsdp_kwargs)


class STPredictor(STBase):
    """Predictor module that generates predictions from encoded tokens."""

    cross_attn = True

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
        learnable_channel_embeddings: bool = True,
        random_channel_embeddings: bool = False,
        output_embedding_size: int | None = None,
        windowed_attention_size: int | None = None,
        layer_attention_modes: list[AttentionMode] | None = None,
    ):
        """Initialize the predictor.

        Args:
            supported_modalities: modalities this model instantiation supports
            encoder_embedding_size: Size of encoder embeddings
            decoder_embedding_size: Size of decoder embeddings
            depth: Number of transformer layers
            mlp_ratio: Ratio for MLP hidden dimension
            num_heads: Number of attention heads
            max_sequence_length: Maximum sequence length
            drop_path: Drop path rate
            learnable_channel_embeddings: Whether to use learnable channel embeddings
            random_channel_embeddings: Whether to randomly initialize channel embeddings
            output_embedding_size: Size of output embeddings
            windowed_attention_size: the size for windowed attention. If set, we do
                windowed attention instead of spatial/temporal attention.
            layer_attention_modes: directly specify the attention mode to use at each layer.
        """
        super().__init__(
            embedding_size=decoder_embedding_size,
            depth=depth,
            mlp_ratio=mlp_ratio,
            num_heads=num_heads,
            max_sequence_length=max_sequence_length,
            drop_path=drop_path,
            learnable_channel_embeddings=learnable_channel_embeddings,
            random_channel_embeddings=random_channel_embeddings,
            supported_modalities=supported_modalities,
            windowed_attention_size=windowed_attention_size,
        )
        # TODO: Rename this weird misname
        self.learnable_channel_embeddings = learnable_channel_embeddings
        self.random_channel_embeddings = random_channel_embeddings
        self.encoder_embedding_size = encoder_embedding_size
        self.layer_attention_modes = layer_attention_modes
        self.encoder_to_decoder_embed = nn.Linear(
            encoder_embedding_size, decoder_embedding_size, bias=True
        )
        if output_embedding_size is None:
            output_embedding_size = encoder_embedding_size
        self.output_embedding_size = output_embedding_size
        self.to_output_embed = nn.Linear(
            decoder_embedding_size, output_embedding_size, bias=True
        )
        # THIS is the learnable mask token
        self.mask_token = nn.Parameter(torch.zeros(decoder_embedding_size))

        self.input_norm = nn.LayerNorm(encoder_embedding_size)
        self.norm = nn.LayerNorm(decoder_embedding_size)
        self.apply(self._init_weights)

    def add_masks(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        """Replace tokens that should be decoded (MaskValue.DECODER_ONLY) with the learnable mask token.

        in a dimension-agnostic way using einops. We assume the final dimension of each token tensor
        is the embedding dimension matching self.mask_token's size.
        """
        output_dict = {}
        available_modalities = return_modalities_from_dict(x)
        modalities_to_process = get_modalities_to_process(
            available_modalities, self.supported_modality_names
        )
        for modality in modalities_to_process:
            x_modality = x[modality]
            mask_name = MaskedOlmoEarthSample.get_masked_modality_name(modality)
            mask_modality = x[mask_name]
            # A boolean mask: True where tokens must be replaced by the mask token
            kept_mask = mask_modality == MaskValue.DECODER.value

            # Build the einops pattern and dimension dict
            spatial_dims = x_modality.shape[
                :-1
            ]  # all dimensions except the last (embedding)
            pattern_input, dim_dict = self._construct_einops_pattern(spatial_dims)

            mask_token_broadcasted = repeat(self.mask_token, pattern_input, **dim_dict)

            # Where kept_mask is True, use the broadcasted mask token
            x_modality = torch.where(
                kept_mask.unsqueeze(-1).bool(), mask_token_broadcasted, x_modality
            )

            output_dict[modality] = x_modality

        return output_dict

    # TODO: These are duplicated static methods maybe they should just be utils functions if they are shared or in some base class
    # TODO: GIVE more explicit function names
    @staticmethod
    def split_x_y(
        tokens: Tensor, mask: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Splits tokens into three groups based on mask values.

        This function:
        1. Sorts tokens according to the mask and gathers them in order.
        2. Chooses tokens to be decoded (x) based on the mask value DECODER.
        3. Chooses tokens to be used as context (y) based on the mask value ONLINE_ENCODER.
        4. Identifies missing tokens (z) based on the mask value MISSING.
        5. Returns boolean masks for x, y, and z along with indices to revert to the original ordering.

        Args:
            tokens: Tokens to split of shape [B, T, D].
            mask: Mask of shape [B, T].

        Returns:
            tokens_to_decode: Tokens to be decoded of shape [B, X_len, D].
            unmasked_tokens: Tokens to be used as context of shape [B, Y_len, D].
            tokens_to_decode_mask: Binary mask for x tokens of shape [B, X_len].
            unmasked_tokens_mask: Binary mask for y tokens of shape [B, Y_len].
            indices: Indices for restoring the original token ordering of shape [B, T].
        """
        # Set Missing Masks to Target Encoder ONLY so that we can have all unused tokens in the middle
        org_mask_dtype = mask.dtype
        missing_mask = mask == MaskValue.MISSING.value
        mask[missing_mask] = MaskValue.TARGET_ENCODER_ONLY.value

        # Sort tokens by mask value (descending order)
        sorted_mask, indices = torch.sort(
            mask.int(), dim=1, descending=True, stable=True
        )
        tokens = tokens.gather(1, indices[:, :, None].expand_as(tokens))

        # Create binary masks for Encoder and Decoder
        binarized_decoder_mask = sorted_mask == MaskValue.DECODER.value
        binarized_online_encoder_mask = sorted_mask == MaskValue.ONLINE_ENCODER.value

        max_length_of_unmasked_tokens = binarized_online_encoder_mask.sum(dim=-1).max()
        max_length_of_decoded_tokens = binarized_decoder_mask.sum(dim=-1).max()

        # the y mask is going to be used to determine which of the y values take. True values
        # take part in the attention (we don't take the inverse here, unlike in the decoder)
        tokens_to_decode = tokens[:, :max_length_of_decoded_tokens]
        tokens_to_decode_mask = binarized_decoder_mask[
            :, :max_length_of_decoded_tokens
        ].to(org_mask_dtype)

        unmasked_tokens = tokens[:, -max_length_of_unmasked_tokens:]
        # the x_mask is just going to be used in the reconstruction, to know which
        # x tokens to add back into the token list. TODO is this even necessary? it could
        # get padded with noise tokens since we don't care about reconstruction at all
        # for a whole bunch of tokens
        unmasked_tokens_mask = binarized_online_encoder_mask[
            :, -max_length_of_unmasked_tokens:
        ].to(org_mask_dtype)

        return (
            tokens_to_decode,
            unmasked_tokens,
            tokens_to_decode_mask,
            unmasked_tokens_mask,
            indices,
        )

    @staticmethod
    def combine_x_y(
        tokens_to_decode: Tensor,
        unmasked_tokens: Tensor,
        tokens_to_decode_mask: Tensor,
        unmasked_tokens_mask: Tensor,
        indices: Tensor,
    ) -> Tensor:
        """Reintegrate the separated token sequences into their original order.

        The token masks zero out positions which are not used/needed,
        and the final scatter step re-applies the original ordering tracked in 'indices'.

        Args:
            tokens_to_decode: Key/value tokens of shape [B, X_len, D].
            unmasked_tokens: Query tokens of shape [B, Y_len, D].
            tokens_to_decode_mask: Binary mask for tokens to decode of shape [B, X_len].
            unmasked_tokens_mask: Binary mask for unmasked tokens of shape [B, Y_len].
            indices: Indices for restoring the original token ordering of shape [B, T].

        Returns:
            A merged tokens tensor of shape [B, T, D] with all tokens in their
            original positions.
        """
        # Get dimensions
        B, T = indices.shape[0], indices.shape[1]
        D = tokens_to_decode.shape[-1]
        tokens = torch.zeros(
            (B, T, D), dtype=tokens_to_decode.dtype, device=tokens_to_decode.device
        )
        tokens[:, -unmasked_tokens.shape[1] :] = (
            unmasked_tokens * unmasked_tokens_mask.unsqueeze(-1)
        )
        tokens[:, : tokens_to_decode.shape[1]] += (
            tokens_to_decode * tokens_to_decode_mask.unsqueeze(-1)
        )
        tokens = tokens.scatter(1, indices[:, :, None].expand_as(tokens), tokens)
        return tokens

    def apply_attn(
        self,
        x: dict[str, Tensor],
        timestamps: Tensor,
        patch_size: int,
        input_res: int,
    ) -> dict[str, Tensor]:
        """Apply attention to the tokens."""
        tokens_only_dict, original_masks_dict, modalities_to_dims_dict = (
            self.split_tokens_masks_and_dims(x)
        )
        tokens_dict = self.composite_encodings(
            tokens_only_dict, timestamps, patch_size, input_res
        )
        tokens_dict.update(original_masks_dict)
        x = tokens_dict

        for i_blk, blk in enumerate(self.blocks):
            # On even blocks, do temporal attention.
            # On odd blocks, do spatial attention.
            # Unless windowed attention is configured.
            if self.layer_attention_modes is not None:
                attention_mode = self.layer_attention_modes[i_blk]
            elif self.windowed_attention_size is not None:
                attention_mode = AttentionMode.WINDOWED
            elif i_blk % 2 == 0:
                attention_mode = AttentionMode.TEMPORAL
            else:
                attention_mode = AttentionMode.SPATIAL

            x, mask = self.collapse_and_combine(x, attention_mode, i_blk)
            x, y, x_mask, y_mask, indices = self.split_x_y(x, mask)

            # note that we are not taking the inverse of the mask, since split_x_y gives us
            # true values for values we want to take part in attention
            x = blk(x=x, y=y, attn_mask=y_mask.bool())

            x = self.combine_x_y(
                tokens_to_decode=x,
                unmasked_tokens=y,
                tokens_to_decode_mask=x_mask,
                unmasked_tokens_mask=y_mask,
                indices=indices,
            )
            x = self.split_and_expand_per_modality(
                x, modalities_to_dims_dict, attention_mode, i_blk
            )
            x.update(original_masks_dict)

        return x

    def is_any_data_to_be_decoded(self, modality_mask: Tensor) -> bool:
        """Check if any data is to be decoded for a given modality."""
        return (MaskValue.DECODER.value == modality_mask).any()

    def forward(
        self,
        x: TokensAndMasks,
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
        decoder_emedded_dict = x._asdict()
        # Apply Input Norms and encoder to decoder embeds to each modality
        available_modalities = x.modalities
        modalities_to_process = get_modalities_to_process(
            available_modalities, self.supported_modality_names
        )
        for modality in modalities_to_process:
            x_modality = getattr(x, modality)
            # Are these normalizations masked correctly?
            x_modality = self.input_norm(x_modality)
            x_modality = self.encoder_to_decoder_embed(x_modality)
            masked_modality_name = x.get_masked_modality_name(modality)
            decoder_emedded_dict[modality] = x_modality
            decoder_emedded_dict[masked_modality_name] = getattr(
                x, masked_modality_name
            )

        tokens_only_dict = self.add_masks(decoder_emedded_dict)
        decoder_emedded_dict.update(tokens_only_dict)
        tokens_and_masks = self.apply_attn(
            decoder_emedded_dict, timestamps, patch_size, input_res
        )
        # TODO: Factor this out into a more readable function
        output_dict = {}
        available_modalities = return_modalities_from_dict(tokens_and_masks)
        modalities_to_process = get_modalities_to_process(
            available_modalities, self.supported_modality_names
        )

        for modality in modalities_to_process:
            masked_modality_name = MaskedOlmoEarthSample.get_masked_modality_name(
                modality
            )
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

    def apply_fsdp(self, **fsdp_kwargs: Any) -> None:
        """Apply FSDP to the model."""
        super().apply_fsdp(**fsdp_kwargs)
        fully_shard(self, **fsdp_kwargs)


@dataclass
class STEncoderConfig(Config):
    """Configuration for the STEncoder."""

    supported_modality_names: list[str]
    embedding_size: int = 16
    # This is the base patch size for the patch embedder
    max_patch_size: int = 8
    min_patch_size: int = 1
    num_heads: int = 2
    mlp_ratio: float = 1.0
    depth: int = 2
    drop_path: float = 0.1
    max_sequence_length: int = 12
    windowed_attention_size: int | None = None
    fuse_layers: int | None = None
    learnable_channel_embeddings: bool = True
    random_channel_embeddings: bool = False
    layer_attention_modes: list[str] | None = None
    fuse_using_cross_attn: bool = True

    def validate(self) -> None:
        """Validate the configuration."""
        if len(self.supported_modalities) == 0:
            raise ValueError("At least one modality must be added!")
        else:
            for modality in self.supported_modalities:
                if modality not in Modality.values():
                    raise ValueError(f"Modality {modality} is not supported")

        if self.layer_attention_modes is not None:
            if len(self.layer_attention_modes) != self.depth:
                raise ValueError(
                    f"got {len(self.layer_attention_modes)} layer attention modes but depth is {self.depth}"
                )
            for mode in self.layer_attention_modes:
                if mode not in AttentionMode.__members__:
                    raise ValueError(f"Invalid attention mode {mode}")

    @property
    def supported_modalities(self) -> list[ModalitySpec]:
        """Get the supported modalities."""
        return get_modality_specs_from_names(self.supported_modality_names)

    def build(self) -> "STEncoder":
        """Build the encoder."""
        self.validate()
        kwargs = self.as_dict(exclude_none=True, recurse=False)
        # supported_modality_names is replaced by supported_modalities
        kwargs.pop("supported_modality_names")
        kwargs["supported_modalities"] = self.supported_modalities
        kwargs["layer_attention_modes"] = (
            [AttentionMode[mode] for mode in self.layer_attention_modes]
            if self.layer_attention_modes
            else None
        )
        logger.info(f"Encoder kwargs: {kwargs}")
        return STEncoder(**kwargs)


@dataclass
class STPredictorConfig(Config):
    """Configuration for the STPredictor."""

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
    windowed_attention_size: int | None = None
    layer_attention_modes: list[str] | None = None

    def validate(self) -> None:
        """Validate the configuration."""
        if len(self.supported_modalities) == 0:
            raise ValueError("At least one modality must be added!")
        else:
            for modality in self.supported_modalities:
                if modality not in Modality.values():
                    raise ValueError(f"Modality {modality} is not supported")

        if self.layer_attention_modes is not None:
            if len(self.layer_attention_modes) != self.depth:
                raise ValueError(
                    f"got {len(self.layer_attention_modes)} layer attention modes but depth is {self.depth}"
                )
            for mode in self.layer_attention_modes:
                if mode not in AttentionMode.__members__:
                    raise ValueError(f"Invalid attention mode {mode}")

    @property
    def supported_modalities(self) -> list[ModalitySpec]:
        """Get the supported modalities."""
        return get_modality_specs_from_names(self.supported_modality_names)

    def build(self) -> "STPredictor":
        """Build the predictor."""
        self.validate()
        kwargs = self.as_dict(exclude_none=True, recurse=False)
        # supported_modality_names is replaced by supported_modalities
        kwargs.pop("supported_modality_names")
        kwargs["supported_modalities"] = self.supported_modalities
        kwargs["layer_attention_modes"] = (
            [AttentionMode[mode] for mode in self.layer_attention_modes]
            if self.layer_attention_modes
            else None
        )
        logger.info(f"Predictor kwargs: {kwargs}")
        return STPredictor(**kwargs)
