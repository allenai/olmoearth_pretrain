"""Model code for the Helios model."""

import logging
import math
from typing import NamedTuple

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor, nn

from helios.constants import BASE_GSD
from helios.nn.attention import Block
from helios.nn.encodings import (
    get_1d_sincos_pos_encoding,
    get_2d_sincos_pos_encoding_with_resolution,
    get_month_encoding_table,
)
from helios.nn.flexi_patch_embed import FlexiPatchEmbed
from helios.train.masking import MaskedHeliosSample, MaskValue

logger = logging.getLogger(__name__)


class TokensAndMasks(NamedTuple):
    """Output to compute the loss on.

    Args:
        s2: sentinel 2 data of shape (B, C_G, T, P_H, P_W)
        s2_mask: sentinel 2 mask indicating which tokens are masked/unmasked
        latlon: lat lon data containing geographical coordinates
        latlon_mask: lat lon mask indicating which coordinates are masked/unmasked
        timestamps: timestamps of the data
    """

    s2: Tensor  # (B, C_G, T, P_H, P_W)
    s2_mask: Tensor
    latlon: Tensor
    latlon_mask: Tensor

    @property
    def device(self) -> torch.device:
        """Get the device of the tokens and masks."""
        return self.s2.device

    # TODO: It seems like we want a lot of our named tuples to have this functionality so we should probably create a utility base class for the named tuples and double subclass
    @classmethod
    def get_masked_modality_name(cls, modality: str) -> str:
        """Get the masked modality name."""
        return f"{modality}_mask"

    @property
    def data_fields(self) -> list[str]:
        """Return all data fields."""
        return [x for x in self._fields if not x.endswith("mask")]

    def get_shape_dict(self) -> dict[str, tuple]:
        """Return a dictionary of the shapes of the fields."""
        return {x: getattr(self, x).shape for x in self._fields}


class FlexiHeliosPatchEmbeddings(nn.Module):
    """Module that patchifies and encodes the input data."""

    def __init__(
        self,
        modalities_to_channel_groups_dict: dict[str, dict[str, list[int]]],
        max_patch_size: int,
        embedding_size: int,
    ):
        """Initialize the patch embeddings.

        Args:
            modalities_to_channel_groups_dict: Dictionary mapping modalities to channel groups
            max_patch_size: Maximum size of patches
            embedding_size: Size of embeddings
        """
        super().__init__()
        self.modalities_to_channel_groups_dict = modalities_to_channel_groups_dict
        self.max_patch_size = max_patch_size
        self.embedding_size = embedding_size
        # TODO: want to be able to remove certain bands and modalities
        # dict will be modality -> channel_group -> bands

        # I want to be able to easily assign an embeddign to a modality and I want it to be explicit for now
        self.per_modality_embeddings = nn.ModuleDict({})
        for modality in self.modalities_to_channel_groups_dict.keys():
            self.per_modality_embeddings[modality] = (
                self._get_patch_embedding_module_for_modality(modality)
            )

    def _get_patch_embedding_module_for_modality(self, modality: str) -> nn.Module:
        """Get the patch embedding module for a modality."""
        channel_groups_dict = self.modalities_to_channel_groups_dict[modality]
        # Based on the modality name we choose the way to embed the data

        # I likely will need to know about what the embedding strategy is in the forward as well
        # Static modality
        if modality == "latlon":
            return nn.ModuleDict(
                {
                    channel_group: nn.Linear(
                        len(channel_band_idxs), self.embedding_size
                    )
                    for channel_group, channel_band_idxs in channel_groups_dict.items()
                }
            )
        else:
            return nn.ModuleDict(
                {
                    channel_group: FlexiPatchEmbed(
                        in_chans=len(channel_band_idxs),
                        embedding_size=self.embedding_size,
                        patch_size=self.max_patch_size,
                    )
                    for channel_group, channel_band_idxs in channel_groups_dict.items()
                }
            )

    # TODO: Likely we want a single object that stores all the data related configuration etc per modality including channel grous bands patch size etc
    def apply_embedding_to_modality(
        self, modality: str, input_data: MaskedHeliosSample, patch_size: int
    ) -> tuple[Tensor, Tensor]:
        """Apply embedding to a modality."""
        channel_groups_dict = self.modalities_to_channel_groups_dict[modality]
        masked_modality_name = input_data.get_masked_modality_name(modality)
        modality_mask = getattr(input_data, masked_modality_name)
        modality_data = getattr(input_data, modality)
        modality_tokens, modality_masks = [], []
        for idx, (channel_group, channel_band_idxs) in enumerate(
            channel_groups_dict.items()
        ):
            modality_specific_kwargs = {}
            if modality == "latlon":
                token_mask = modality_mask[:, idx]
            else:
                token_mask = modality_mask[:, 0::patch_size, 0::patch_size, :, idx]
                modality_specific_kwargs = {"patch_size": patch_size}
            patchified_dims = token_mask.shape[1:]
            # Now apply the embedding to
            if self.is_any_data_seen_by_encoder(token_mask):
                patchified_data = modality_data[..., channel_band_idxs]
                patchified_data = self.per_modality_embeddings[modality][channel_group](
                    patchified_data, **modality_specific_kwargs
                )
            else:
                patchified_data = torch.empty(
                    modality_data.shape[0],
                    *patchified_dims,
                    self.embedding_size,
                    dtype=modality_data.dtype,
                    device=modality_data.device,
                )
            modality_tokens.append(patchified_data)
            modality_masks.append(token_mask)
        return torch.stack(modality_tokens, dim=-2), torch.stack(modality_masks, dim=-1)

    @staticmethod
    def is_any_data_seen_by_encoder(modality_mask: Tensor) -> bool:
        """Check if any data is seen by the encoder."""
        return modality_mask.min() == MaskValue.ONLINE_ENCODER.value

    def forward(
        self,
        input_data: MaskedHeliosSample,
        patch_size: int,
    ) -> TokensAndMasks:
        """Return flexibly patchified embeddings for each modality of the input data.

        Given a [B, H, W, (T), C] inputs, returns a [B, H, W, (T), C_G, D] output.
        We assume that the spatial masks are consistent for the given patch size,
        so that if patch_size == 2 then one possible mask would be
        [0, 0, 1, 1]
        [0, 0, 1, 1]
        [1, 1, 0, 0]
        [1, 1, 0, 0]
        for the H, W dimensions
        """
        output_dict = {}
        for modality in self.modalities_to_channel_groups_dict.keys():
            modality_tokens, modality_masks = self.apply_embedding_to_modality(
                modality, input_data, patch_size
            )
            output_dict[modality] = modality_tokens
            modality_mask_name = input_data.get_masked_modality_name(modality)
            output_dict[modality_mask_name] = modality_masks
        return TokensAndMasks(**output_dict)


class FlexiHeliosCompositeEncodings(nn.Module):
    """Composite encodings for the FlexiHelios model."""

    def __init__(
        self,
        embedding_size: int,
        modalities_to_channel_groups_dict: dict[str, dict[str, list[int]]],
        max_sequence_length: int,
        base_patch_size: int,
        use_channel_embs: bool = True,
    ):
        """Initialize the composite encodings.

        Args:
            embedding_size: Size of token embeddings
            modalities_to_channel_groups_dict: Dictionary mapping modalities to channel groups
            max_sequence_length: Maximum sequence length
            base_patch_size: Base patch size
            use_channel_embs: Whether to use learnable channel embeddings
        """
        super().__init__()
        self.embedding_size = embedding_size
        self.modalities_to_channel_groups_dict = modalities_to_channel_groups_dict
        self.embedding_size = embedding_size
        self.base_patch_size = base_patch_size
        self.max_sequence_length = (
            max_sequence_length  # This max sequence length is a time dim thing
        )
        # TODO: we need to be able to calculate the size of the param based on what types of embeddings it will get

        # we have 4 embeddings types (pos_in_time, pos_in_space, month, channel) so each get
        # 0.25 of the dimension
        self.embedding_dim_per_embedding_type = int(embedding_size * 0.25)
        # Position encodings for time dimension initialized to 1D sinusoidal encodings
        self.pos_embed = nn.Parameter(
            get_1d_sincos_pos_encoding(
                torch.arange(max_sequence_length),
                self.embedding_dim_per_embedding_type,
            ),
            requires_grad=False,
        )
        # M
        month_tab = get_month_encoding_table(self.embedding_dim_per_embedding_type)
        self.month_embed = nn.Embedding.from_pretrained(month_tab, freeze=True)
        if use_channel_embs:
            args = {"requires_grad": True}
        else:
            args = {"requires_grad": False}

        self.per_modality_channel_embeddings = nn.ParameterDict(
            {
                modality: nn.Parameter(
                    torch.zeros(
                        len(channel_groups_dict.keys()),
                        self.embedding_dim_per_embedding_type,
                    ),
                    **args,
                )
                for modality, channel_groups_dict in self.modalities_to_channel_groups_dict.items()
            }
        )

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def calculate_gsd_ratio(input_res: float, patch_size: int) -> float:
        """Calculate the Ground Sample Distance ratio."""
        return input_res * patch_size / BASE_GSD

    def _apply_encodings_per_modality(
        self,
        modality: str,
        modality_tokens: Tensor,
        timestamps: Tensor | None = None,
        patch_size: int | None = None,
        input_res: int | None = None,
    ) -> Tensor:
        """Apply the encodings to the patchified data based on modality type.

        Args:
            modality: Name of the modality being processed
            modality_tokens: Token embeddings for the modality
            timestamps: Optional timestamps for temporal encodings
            patch_size: Optional patch size for spatial encodings
            input_res: Optional input resolution for spatial encodings

        Returns:
            Tensor with encodings applied based on modality type
        """
        # TODO: Improve this implementation
        if modality == "latlon":
            b, c_g, _ = modality_tokens.shape
            # Static modality only needs channel embeddings
            modality_channel_embed = self.per_modality_channel_embeddings[modality]
            modality_channel_embed = repeat(
                modality_channel_embed, "c_g d -> b c_g d", b=b
            )
            modality_channel_embed = F.pad(
                modality_channel_embed,
                (0, self.embedding_size - modality_channel_embed.shape[-1]),
            )
            return modality_tokens + modality_channel_embed

        # For temporal modalities like s1/s2
        if timestamps is None or patch_size is None or input_res is None:
            raise ValueError(
                f"timestamps, patch_size and input_res required for modality {modality}"
            )

        b, h, w, t, c_g, _ = modality_tokens.shape

        # Channel embeddings
        modality_channel_embed = self.per_modality_channel_embeddings[modality]
        modality_channel_embed = repeat(
            modality_channel_embed, "c_g d -> b h w t c_g d", b=b, h=h, w=w, t=t
        )

        # Time position encodings
        modality_pos_embed = repeat(
            self.pos_embed[:t], "t d -> b h w t c_g d", b=b, h=h, w=w, c_g=c_g
        )

        # Month encodings
        months = timestamps[:, :, 1]
        month_embed = self.month_embed(months)
        modality_month_embed = repeat(
            month_embed, "b t d -> b h w t c_g d", h=h, w=w, c_g=c_g
        )

        # Spatial encodings
        gsd_ratio = self.calculate_gsd_ratio(input_res, patch_size)
        current_device = modality_tokens.device
        spatial_embed = get_2d_sincos_pos_encoding_with_resolution(
            grid_size=h,
            res=torch.ones(b, device=current_device) * gsd_ratio,
            encoding_dim=self.embedding_dim_per_embedding_type,
            device=current_device,
        )
        spatial_embed = rearrange(spatial_embed, "b (h w) d -> b h w d", h=h, w=w)
        spatial_embed = repeat(spatial_embed, "b h w d -> b h w t c_g d", c_g=c_g, t=t)

        # Combine all encodings
        modality_embed = torch.cat(
            [
                modality_channel_embed,
                modality_pos_embed,
                modality_month_embed,
                spatial_embed,
            ],
            dim=-1,
        )

        return modality_tokens + modality_embed

    def forward(
        self,
        per_modality_input_tokens: dict[str, Tensor],
        timestamps: Tensor,
        patch_size: int,
        input_res: int = BASE_GSD,
    ) -> dict[str, Tensor]:
        """Apply the encodings to the patchified data.

        Args:
            per_modality_input_tokens: Tokens only for each modality
            timestamps: Timestamps of the data
            patch_size: Size of patches
            input_res: Resolution of the input data

        Returns:
            Tokens only for each modality
        """
        # TODO: Improve this implementation
        output_dict = {}
        for modality in self.modalities_to_channel_groups_dict.keys():
            if modality == "latlon":
                output_dict[modality] = self._apply_encodings_per_modality(
                    modality, per_modality_input_tokens[modality]
                )
            else:
                output_dict[modality] = self._apply_encodings_per_modality(
                    modality,
                    per_modality_input_tokens[modality],
                    timestamps=timestamps,
                    patch_size=patch_size,
                    input_res=input_res,
                )
        return output_dict


class FlexiHeliosBase(nn.Module):
    """FlexiHeliosBase is a base class for FlexiHelios models."""

    cross_attn: bool = False

    def __init__(
        self,
        embedding_size: int,
        max_sequence_length: int,
        base_patch_size: int,
        use_channel_embs: bool,
        num_heads: int,
        mlp_ratio: float,
        depth: int,
        drop_path: float,
        modalities_to_channel_groups_dict: dict[str, dict[str, list[int]]],
    ):
        """Initialize the FlexiHeliosBase class."""
        super().__init__()

        self.embedding_size = embedding_size
        self.modalities_to_channel_groups_dict = modalities_to_channel_groups_dict
        logger.info(
            f"modalities being used by model: {modalities_to_channel_groups_dict.keys()}"
        )

        self.max_sequence_length = max_sequence_length
        self.base_patch_size = base_patch_size
        self.use_channel_embs = use_channel_embs

        self.blocks = nn.ModuleList(
            [
                Block(
                    embedding_size,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=nn.LayerNorm,  # TODO: This should be configurable
                    cross_attn=self.cross_attn,
                    drop_path=drop_path,
                )
                for _ in range(depth)
            ]
        )

        self.composite_encodings = FlexiHeliosCompositeEncodings(
            embedding_size,
            modalities_to_channel_groups_dict,
            max_sequence_length,
            base_patch_size,
            use_channel_embs,
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

    # is naming here confusing if one of these channels can be missing?
    def collapse_and_combine_hwtc(self, x: TokensAndMasks) -> tuple[Tensor, Tensor]:
        """Collapse the tokens and masks, respectively, into two tensors."""
        tokens, masks = [], []
        for modality in self.modalities_to_channel_groups_dict.keys():
            masked_modality_name = x.get_masked_modality_name(modality)
            x_modality = getattr(x, modality)
            x_modality_mask = getattr(x, masked_modality_name)
            tokens.append(rearrange(x_modality, "b ... d -> b (...) d"))
            masks.append(rearrange(x_modality_mask, "b ... -> b (...)"))
        tokens = torch.cat(tokens, dim=1)
        masks = torch.cat(masks, dim=1)
        return tokens, masks

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

    @staticmethod
    def split_and_expand_per_modality(
        x: Tensor, modalities_to_dims_dict: dict[str, tuple]
    ) -> dict[str, Tensor]:
        """Split and expand the tokens per modality.

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


class Encoder(FlexiHeliosBase):
    """Encoder module that processes masked input samples into token representations."""

    cross_attn: bool = False

    def __init__(
        self,
        embedding_size: int,
        max_patch_size: int,
        num_heads: int,
        mlp_ratio: float,
        depth: int,
        drop_path: float,
        modalities_to_channel_groups_dict: dict[str, dict[str, list[int]]],
        max_sequence_length: int,
        base_patch_size: int,
        use_channel_embs: bool = True,
    ):
        """Initialize the encoder.

        Args:
            embedding_size: Size of token embeddings
            max_patch_size: Maximum patch size for patchification
            num_heads: Number of attention heads
            mlp_ratio: Ratio for MLP hidden dimension
            depth: Number of transformer layers
            drop_path: Drop path rate
            modalities_to_channel_groups_dict: Dictionary mapping modalities to channel groups
            max_sequence_length: Maximum sequence length
            base_patch_size: Base patch size
            use_channel_embs: Whether to use learnable channel embeddings
        """
        super().__init__(
            embedding_size=embedding_size,
            depth=depth,
            mlp_ratio=mlp_ratio,
            num_heads=num_heads,
            max_sequence_length=max_sequence_length,
            base_patch_size=base_patch_size,
            use_channel_embs=use_channel_embs,
            drop_path=drop_path,
            modalities_to_channel_groups_dict=modalities_to_channel_groups_dict,
        )
        self.patch_embeddings = FlexiHeliosPatchEmbeddings(
            modalities_to_channel_groups_dict,
            max_patch_size,
            embedding_size,
        )
        self.norm = nn.LayerNorm(embedding_size)
        self.apply(self._init_weights)

    def create_token_exit_ids(
        self, x: dict[str, Tensor], token_exit_cfg: dict[str, int]
    ) -> dict[str, Tensor]:
        """Create the token exit ids for # of layers of attention for each band group.

        Assumes modality channel groups are in the second to last dimension of the tokens.
        """
        exit_ids_per_modality_dict = {}
        for (
            modality,
            band_groups_dict,
        ) in self.modalities_to_channel_groups_dict.items():
            exit_seq_modality = torch.zeros_like(x[modality])
            for idx, (band_group, _) in enumerate(band_groups_dict.items()):
                num_exit_layers = token_exit_cfg[band_group]
                exit_seq_modality[..., idx, :] = num_exit_layers
            exit_ids_per_modality_dict[modality] = exit_seq_modality
        return exit_ids_per_modality_dict

    @staticmethod
    def remove_masked_tokens(x: Tensor, mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Remove masked tokens from the tokens and masks.

        Implementation from https://stackoverflow.com/a/68621610/2332296

        On Input:
        1 means this token should be removed
        0 means this token should be kept

        Args:
            x: Tokens to remove masked tokens from
            mask: Mask to remove masked tokens from

        Returns:
            tokens: [B, T, D]
            indices: [B, T]
            updated_mask: [B, T]
            where T is the max number of unmasked tokens for an instance
        """
        org_mask_dtype = mask.dtype
        mask = mask.bool()
        # At this point when we flip the mask 1 means keep 0 means remove
        sorted_mask, indices = torch.sort(
            (~mask).int(), dim=1, descending=True, stable=True
        )
        # Now all the places where we want to keep the token are at the front of the tensor
        x = x.gather(1, indices[:, :, None].expand_as(x))
        # Now all tokens that should be kept are first in the tensor

        # set masked values to 0 (not really necessary since we'll ignore them anyway)
        x = x * sorted_mask.unsqueeze(-1)

        # cut off to the length of the longest sequence
        max_length = sorted_mask.sum(-1).max()
        x = x[:, :max_length]
        # New mask chopped to the longest sequence
        updated_mask = 1 - sorted_mask[:, :max_length]

        return x, indices, updated_mask.to(dtype=org_mask_dtype)

    @staticmethod
    def should_exit(i_blk: int, exit_after_n_layers: int | None) -> bool:
        """Determine if the current block should exit the attention layers."""
        if exit_after_n_layers is None:
            return False
        return i_blk >= exit_after_n_layers

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
        assert (
            x.shape[1] > 0
        ), "x must have at least one token we should not mask all tokens"
        masked_tokens = repeat(
            torch.zeros_like(x[0, 0, :]), "d -> b t d", b=x.shape[0], t=indices.shape[1]
        )
        full_mask = torch.cat(
            (
                mask,
                torch.ones(
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
        out[~full_mask.bool()] = x[~mask.bool()]
        # then move them to their original positions
        out = out.scatter(1, indices[:, :, None].expand_as(out), out)
        full_mask = full_mask.scatter(1, indices.expand_as(full_mask), full_mask)
        # Values that were masked out are not returned but the values that are still there are returned to the original positions
        return out, full_mask

    def apply_attn(
        self,
        x: TokensAndMasks,
        timestamps: Tensor,
        patch_size: int,
        input_res: int,
        token_exit_cfg: dict[str, int] | None = None,
        exit_after_n_layers: int | None = None,
    ) -> TokensAndMasks:
        """Apply the attention to the tokens and masks."""
        # TODO: this part should be cleaner many unneded packaging and unpackaging of data
        tokens_only_dict = {}
        original_masks_dict = {}
        modalities_to_dims_dict = {}
        # TODO: add a class method for the named tuple here
        for modality in self.modalities_to_channel_groups_dict.keys():
            x_modality = getattr(x, modality)
            tokens_only_dict[modality] = x_modality
            modalities_to_dims_dict[modality] = x_modality.shape
            masked_modality_name = x.get_masked_modality_name(modality)
            original_masks_dict[masked_modality_name] = getattr(x, masked_modality_name)

        # TODO: wrap all this complicated exit token stuff into a seperate method
        if token_exit_cfg:
            exit_ids_per_modality = self.create_token_exit_ids(
                tokens_only_dict, token_exit_cfg
            )
            x_dict = x._asdict()
            x_dict.update(exit_ids_per_modality)
            tokens_and_masks_exit_ids_per_modality = TokensAndMasks(**x_dict)
            # Exit ids seqs tells us which layer to exit each token
            exit_ids_seq, _ = self.collapse_and_combine_hwtc(
                tokens_and_masks_exit_ids_per_modality
            )
            exited_tokens_and_masks = TokensAndMasks(**x_dict)
            # The exit tokens are the tensor that store tokens that exit early from the encoder
            exited_tokens, _ = self.collapse_and_combine_hwtc(exited_tokens_and_masks)
        else:
            exit_ids_seq = None
            exited_tokens = None

        tokens_dict = self.composite_encodings.forward(
            tokens_only_dict,
            timestamps,
            patch_size,
            input_res,
        )
        # Prepare data for collapsing and combining
        tokens_and_masks_dict = x._asdict()
        tokens_and_masks_dict.update(tokens_dict)
        tokens_and_masks = TokensAndMasks(**tokens_and_masks_dict)
        x, mask = self.collapse_and_combine_hwtc(tokens_and_masks)

        # we only care about the values >= 1 for this mask, since 2 just tells the decoder
        # to decode those tokens. From the perspective of the encoder, 1 and 2 are equivalent
        # since they both represent masked values
        new_mask = mask >= MaskValue.TARGET_ENCODER_ONLY.value

        tokens, indices, new_mask = self.remove_masked_tokens(x, new_mask)
        if exit_ids_seq is not None:
            exit_ids_seq, _, _ = self.remove_masked_tokens(exit_ids_seq, mask)
            # still linear projections
            exited_tokens, _, _ = self.remove_masked_tokens(exited_tokens, mask)

        # Apply attn with varying encoder depths
        for i_blk, blk in enumerate(self.blocks):
            if self.should_exit(i_blk, exit_after_n_layers):
                # if exit_after is N, then we exit after the Nth layer
                # if exit_after is 0, then all layers are skipped
                break

            # skip the 0th block since this is just the linear
            # projection
            if (exit_ids_seq is not None) and (i_blk > 0):
                assert exited_tokens is not None
                # If a token should exit, then we update the exit token with the current token at the same position
                exited_tokens = torch.where(
                    condition=(exit_ids_seq == i_blk),
                    input=tokens.detach(),
                    other=exited_tokens.detach(),
                )
            # we take the inverse of the mask because a value
            # of True indicates the value *should* take part in
            # attention
            # WARNING: THIS MAY CHANGE DEPENDING ON THE ATTENTION IMPLEMENTATION
            tokens = blk(x=tokens, y=None, attn_mask=~new_mask.bool())

        if exit_ids_seq is not None:
            assert exited_tokens is not None
            # full depth
            # IMPORTANT: write this to x
            tokens = torch.where(
                condition=(exit_ids_seq == (i_blk + 1)),  # 2 for full depth
                input=tokens.detach(),
                other=exited_tokens.detach(),
            )

        # we apply the norm before we add the removed tokens,
        # so that the norm is only computed against "real" tokens
        tokens = self.norm(tokens)
        # we don't care about the mask returned by add_removed_tokens, since we will
        # just use the original, unclipped mask here
        tokens, _ = self.add_removed_tokens(tokens, indices, new_mask)
        tokens_per_modality_dict = self.split_and_expand_per_modality(
            tokens, modalities_to_dims_dict
        )
        tokens_and_masks_dict = {}
        tokens_and_masks_dict.update(original_masks_dict)  # Add masks first
        tokens_and_masks_dict.update(tokens_per_modality_dict)  # Add tokens second
        return TokensAndMasks(**tokens_and_masks_dict)

    def forward(
        self,
        x: MaskedHeliosSample,
        patch_size: int,
        input_res: int = BASE_GSD,
        exit_after_n_layers: int | None = None,
        token_exit_cfg: dict | None = None,
    ) -> TokensAndMasks:
        """Process masked input samples into token representations.

        Args:
            x: Masked input sample containing the data to be encoded
            patch_size: Size of patches to divide the input into
            input_res: Resolution of the input data
            exit_after_n_layers: Layer to exit after
            token_exit_cfg: Configuration for token exit

        Returns:
            TokensAndMasks containing the encoded representations and their masks
        """
        # TODO: Add step to validate the exit config is valid
        patchified_tokens_and_masks = self.patch_embeddings.forward(x, patch_size)
        if (exit_after_n_layers is None) or (exit_after_n_layers > 0):
            patchified_tokens_and_masks = self.apply_attn(
                x=patchified_tokens_and_masks,
                timestamps=x.timestamps,
                patch_size=patch_size,
                input_res=input_res,
                exit_after_n_layers=exit_after_n_layers,
                token_exit_cfg=token_exit_cfg,
            )

        output_dict = {}
        for modality in self.modalities_to_channel_groups_dict.keys():
            x_modality = getattr(patchified_tokens_and_masks, modality)
            masked_modality_name = patchified_tokens_and_masks.get_masked_modality_name(
                modality
            )
            output_dict[modality] = x_modality
            output_dict[masked_modality_name] = getattr(
                patchified_tokens_and_masks, masked_modality_name
            )
        return TokensAndMasks(**output_dict)


class Predictor(FlexiHeliosBase):
    """Predictor module that generates predictions from encoded tokens."""

    cross_attn = True

    def __init__(
        self,
        modalities_to_channel_groups_dict: dict[str, dict[str, list[int]]],
        encoder_embedding_size: int = 128,
        decoder_embedding_size: int = 128,
        depth: int = 2,
        mlp_ratio: float = 2.0,
        num_heads: int = 8,
        max_sequence_length: int = 24,
        max_patch_size: int = 8,
        drop_path: float = 0.0,
        learnable_channel_embeddings: bool = False,
        output_embedding_size: int | None = None,
    ):
        """Initialize the predictor.

        Args:
            modalities_to_channel_groups_dict: Dictionary mapping modalities to channel groups
            encoder_embedding_size: Size of encoder embeddings
            decoder_embedding_size: Size of decoder embeddings
            depth: Number of transformer layers
            mlp_ratio: Ratio for MLP hidden dimension
            num_heads: Number of attention heads
            max_sequence_length: Maximum sequence length
            max_patch_size: Maximum patch size
            drop_path: Drop path rate
            learnable_channel_embeddings: Whether to use learnable channel embeddings
            output_embedding_size: Size of output embeddings
        """
        super().__init__(
            embedding_size=decoder_embedding_size,
            depth=depth,
            mlp_ratio=mlp_ratio,
            num_heads=num_heads,
            max_sequence_length=max_sequence_length,
            base_patch_size=max_patch_size,
            use_channel_embs=learnable_channel_embeddings,
            drop_path=drop_path,
            modalities_to_channel_groups_dict=modalities_to_channel_groups_dict,
        )
        self.learnable_channel_embeddings = learnable_channel_embeddings
        self.encoder_embedding_size = encoder_embedding_size
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

        self.max_patch_size = max_patch_size
        self.input_norm = nn.LayerNorm(encoder_embedding_size)
        self.norm = nn.LayerNorm(decoder_embedding_size)
        self.apply(self._init_weights)

    def add_masks(self, tokens_and_masks: TokensAndMasks) -> dict[str, Tensor]:
        """Replace tokens that should be decoded (MaskValue.DECODER_ONLY) with the learnable mask token.

        in a dimension-agnostic way using einops. We assume the final dimension of each token tensor
        is the embedding dimension matching self.mask_token's size.
        """
        output_dict = {}
        for modality in self.modalities_to_channel_groups_dict.keys():
            x_modality = getattr(tokens_and_masks, modality)
            mask_modality = getattr(
                tokens_and_masks, tokens_and_masks.get_masked_modality_name(modality)
            )

            # A boolean mask: True where tokens must be replaced by the mask token
            kept_mask = mask_modality == MaskValue.DECODER_ONLY.value

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

    @staticmethod
    def split_x_y(
        tokens: Tensor, mask: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Splits tokens into two groups—one for decoding (x) and one for context (y)—based on mask values.

        This function:
        1. Sorts tokens according to the mask and gathers them in order.
        2. Chooses a portion of tokens (x) to be decoded based on the mask.
        3. Chooses the remainder (y) as the context for attention based on the mask.
        4. Returns boolean masks for x and y along with indices to revert to the original ordering later if needed.

        Args:
            tokens: Tokens to split of shape [B, T, D].
            mask: Mask of shape [B, T].

        Returns:
            x: Tokens to be decoded of shape [B, X_len, D].
            y: Tokens to be used as context of shape [B, Y_len, D].
            x_mask: Binary mask for x tokens of shape [B, X_len].
            y_mask: Binary mask for y tokens of shape [B, Y_len]. 1 means the token is used in the attention.
            indices: Indices for restoring the original token ordering of shape [B, T].
        """
        org_mask_dtype = mask.dtype
        # https://stackoverflow.com/a/68621610/2332296
        # move all non-masked values to the front of their rows
        # and all masked values to be decoded to the end of their rows
        # since we multiply by -1, we now have that -2: to be decoded, -1: masked and ignored, 0: unmasked
        sorted_mask, indices = torch.sort(
            mask.int(), dim=1, descending=True, stable=True
        )
        tokens = tokens.gather(1, indices[:, :, None].expand_as(tokens))
        binarized_decoder_only_mask = sorted_mask == MaskValue.DECODER_ONLY.value
        binarized_online_encoder_mask = sorted_mask == MaskValue.ONLINE_ENCODER.value
        # cut off to the length of the longest sequence
        max_length_to_be_decoded = binarized_decoder_only_mask.sum(-1).max()
        max_length_of_unmasked_tokens = binarized_online_encoder_mask.sum(-1).max()
        # x will be the query tokens, and y will be the key / value tokens
        x = tokens[:, :max_length_to_be_decoded]
        y = tokens[:, -max_length_of_unmasked_tokens:]

        # the x_mask is just going to be used in the reconstruction, to know which
        # x tokens to add back into the token list. TODO is this even necessary? it could
        # get padded with noise tokens since we don't care about reconstruction at all
        # for a whole bunch of tokens
        x_mask = binarized_decoder_only_mask[:, :max_length_to_be_decoded].to(
            dtype=org_mask_dtype
        )
        # the y mask is going to be used to determine which of the y values take. True values
        # take part in the attention (we don't take the inverse here, unlike in the decoder)
        y_mask = binarized_online_encoder_mask[:, -max_length_of_unmasked_tokens:].to(
            dtype=org_mask_dtype
        )
        return x, y, x_mask, y_mask, indices

    @staticmethod
    def combine_x_y(
        x: Tensor, y: Tensor, x_mask: Tensor, y_mask: Tensor, indices: Tensor
    ) -> Tensor:
        """Reintegrate the separated x (query) and y (key-value) token sequences into their original order.

        The token masks (x_mask, y_mask) zero out positions which are not used/needed,
        and the final scatter step re-applies the original ordering tracked in 'indices'.

        Args:
            x: Query tokens of shape [B, X_len, D].
            y: Key/value tokens of shape [B, Y_len, D].
            x_mask: Binary mask for x tokens of shape [B, X_len].
            y_mask: Binary mask for y tokens of shape [B, Y_len].
            indices: Indices for restoring the original token ordering of shape [B, T].

        Returns:
            A merged tokens tensor of shape [B, T, D] with both x and y in their
            original positions.
        """
        # multiply by mask to zero out, then add
        B, T = indices.shape[0], indices.shape[1]
        D = x.shape[-1]
        tokens = torch.zeros((B, T, D), dtype=x.dtype, device=x.device)
        tokens[:, -y.shape[1] :] = y * y_mask.unsqueeze(-1)
        tokens[:, : x.shape[1]] += x * x_mask.unsqueeze(-1)
        tokens = tokens.scatter(1, indices[:, :, None].expand_as(tokens), tokens)
        return tokens

    def apply_attn(
        self,
        x: TokensAndMasks,
        timestamps: Tensor,
        patch_size: int,
        input_res: int,
    ) -> TokensAndMasks:
        """Apply the attention to the tokens and masks."""
        # TODO: This can likely be a method on the named tuple that returns these 3 dicts or a method in flexiheliosbase
        tokens_only_dict = {}
        original_masks_dict = {}
        modalities_to_dims_dict = {}
        for modality in self.modalities_to_channel_groups_dict.keys():
            x_modality = getattr(x, modality)
            tokens_only_dict[modality] = x_modality
            modalities_to_dims_dict[modality] = x_modality.shape
            masked_modality_name = x.get_masked_modality_name(modality)
            original_masks_dict[masked_modality_name] = getattr(x, masked_modality_name)
        tokens_dict = self.composite_encodings(
            tokens_only_dict, timestamps, patch_size, input_res
        )
        tokens_and_masks_dict = x._asdict()
        tokens_and_masks_dict.update(tokens_dict)
        x, mask = self.collapse_and_combine_hwtc(
            TokensAndMasks(**tokens_and_masks_dict)
        )
        x, y, x_mask, y_mask, indices = self.split_x_y(x, mask)
        for blk in self.blocks:
            # note that we are not taking the inverse of the mask, since split_x_y gives us
            # true values for values we want to take part in attention
            x = blk(x=x, y=y, attn_mask=y_mask.bool())
        x = self.combine_x_y(x, y, x_mask, y_mask, indices)
        tokens_per_modality_dict = self.split_and_expand_per_modality(
            x, modalities_to_dims_dict
        )
        tokens_and_masks_dict = {}
        tokens_and_masks_dict.update(original_masks_dict)
        tokens_and_masks_dict.update(tokens_per_modality_dict)
        return TokensAndMasks(**tokens_and_masks_dict)

    def is_any_data_to_be_decoded(self, modality_mask: Tensor) -> bool:
        """Check if any data is to be decoded for a given modality."""
        return modality_mask.max() == MaskValue.DECODER_ONLY.value

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
            timestamps: Timestamps of the input data
            patch_size: Size of patches to divide the input into
            input_res: Resolution of the input data (in meters)

        Returns:
            TokensAndMasks containing the predicted tokens and their masks
        """
        # Apply Input Norms and encoder to decoder embeds to each modality
        decoder_emedded_dict = {}
        for modality in self.modalities_to_channel_groups_dict.keys():
            x_modality = getattr(x, modality)
            x_modality = self.input_norm(x_modality)
            x_modality = self.encoder_to_decoder_embed(x_modality)
            masked_modality_name = x.get_masked_modality_name(modality)
            decoder_emedded_dict[modality] = x_modality
            decoder_emedded_dict[masked_modality_name] = getattr(
                x, masked_modality_name
            )
        tokens_only_dict = self.add_masks(TokensAndMasks(**decoder_emedded_dict))
        decoder_emedded_dict.update(tokens_only_dict)
        tokens_and_masks = TokensAndMasks(**decoder_emedded_dict)
        tokens_and_masks = self.apply_attn(
            tokens_and_masks, timestamps, patch_size, input_res
        )

        # TODO: Factor this out into a more readable function
        output_dict = {}
        for (
            modality,
            channel_groups_dict,
        ) in self.modalities_to_channel_groups_dict.items():
            masked_modality_name = tokens_and_masks.get_masked_modality_name(modality)
            modality_mask = getattr(tokens_and_masks, masked_modality_name)
            # patchify masked data
            per_modality_output_tokens = []
            modality_data = getattr(tokens_and_masks, modality)
            modality_specific_dims = self.grab_modality_specific_dims(modality_data)
            for idx in range(len(channel_groups_dict)):
                if self.is_any_data_to_be_decoded(modality_mask):
                    per_channel_modality_data = modality_data[..., idx, :]
                    output_data = self.to_output_embed(
                        self.norm(per_channel_modality_data)
                    )
                else:
                    # If all data should be ignored by encoder, we need to return an empty tensor
                    output_data = torch.empty(
                        modality_data.shape[0],
                        *modality_specific_dims,
                        self.output_embedding_size,
                        dtype=modality_data.dtype,
                        device=modality_data.device,
                    )
                per_modality_output_tokens.append(output_data)
            output_dict[modality] = torch.stack(per_modality_output_tokens, dim=-2)
            output_dict[masked_modality_name] = modality_mask
        return TokensAndMasks(**output_dict)


if __name__ == "__main__":
    import rasterio

    from helios.constants import S2_BANDS

    # Each band set is stored at different resolutions for monthly so that has to happen for us to load in
    path_to_example_s2_scene = "gs://ai2-helios/data/20250130-sample-dataset-helios/10_sentinel2_monthly/EPSG:32610_165_-1971_10.tif"
    other_bands_s2 = "gs://ai2-helios/data/20250130-sample-dataset-helios/10_sentinel2_monthly/EPSG:32610_165_-1971_20.tif"
    more_bands_s2 = "gs://ai2-helios/data/20250130-sample-dataset-helios/10_sentinel2_monthly/EPSG:32610_165_-1971_40.tif"

    # Read each file and print shapes
    with rasterio.open(path_to_example_s2_scene) as data:
        array_10m = data.read()

    with rasterio.open(other_bands_s2) as data:
        array_20m = data.read()
        # Convert to torch tensor and add batch dimension
        array_20m_tensor = torch.from_numpy(array_20m).float().unsqueeze(0)
        # Interpolate to 256x256
        array_20m_upsampled = F.interpolate(
            array_20m_tensor, size=(256, 256), mode="bilinear", align_corners=False
        ).squeeze(0)
        array_20m_upsampled = array_20m_upsampled

    with rasterio.open(more_bands_s2) as data:
        array_40m = data.read()
        # Convert to torch tensor and add batch dimension
        array_40m_tensor = torch.from_numpy(array_40m).float().unsqueeze(0)
        # Interpolate to 256x256
        array_40m_upsampled = F.interpolate(
            array_40m_tensor, size=(256, 256), mode="bilinear", align_corners=False
        ).squeeze(0)
        array_40m_upsampled = array_40m_upsampled
    array_10m = torch.from_numpy(array_10m).float()
    num_timesteps = 12
    num_bands = len(S2_BANDS)
    s2_array = torch.cat([array_10m, array_20m_upsampled, array_40m_upsampled], dim=0)
    s2_array = rearrange(
        s2_array, "(t c) h w -> h w t c", c=num_bands, t=num_timesteps
    ).unsqueeze(0)
    modalities_to_channel_groups_dict = {
        "s2": {
            "S2_RGB": [S2_BANDS.index(b) for b in ["B02", "B03", "B04"]],
            "S2_Red_Edge": [S2_BANDS.index(b) for b in ["B05", "B06", "B07"]],
            "S2_NIR_10m": [S2_BANDS.index(b) for b in ["B08"]],
            "S2_NIR_20m": [S2_BANDS.index(b) for b in ["B8A"]],
            "S2_SWIR": [S2_BANDS.index(b) for b in ["B11", "B12"]],
        }
    }

    s2_mask = torch.randint_like(s2_array, 0, 3).float()
    latlon = torch.randn(1, 2).float()
    latlon_mask = torch.ones_like(latlon).float()
    timestamps = (
        torch.tensor(
            [
                # 1
                [1, 2, 2018],
                # 2
                [5, 2, 2018],
                # 3
                [15, 5, 2018],
                # 4
                [25, 8, 2018],
                # 5
                [10, 9, 2018],
                # 6
                [20, 10, 2018],
                # 7
                [30, 10, 2018],
                # 8
                [15, 11, 2018],
                # 9
                [25, 1, 2019],
                # 10
                [10, 2, 2019],
                # 11
                [20, 3, 2019],
                # 12
                [30, 4, 2019],
            ]
        )
        .unsqueeze(0)
        .permute(0, 2, 1)
    )
    device = torch.device("cuda")
    x = MaskedHeliosSample(
        s2_array,
        s2_mask,
        latlon,
        latlon_mask,
        timestamps,
    )
    max_patch_size = 8
    embedding_size = 16
    max_sequence_length = 12  # For now we are not supporting variable time series
    base_patch_size = 4
    patch_size = 4
    use_channel_embs = True
    input_res = BASE_GSD
    patch_size = 4
    exit_after = None
    token_exit_cfg = None
    encoder = Encoder(
        embedding_size=embedding_size,
        max_patch_size=max_patch_size,
        num_heads=2,
        mlp_ratio=4.0,
        depth=2,
        drop_path=0.1,
        modalities_to_channel_groups_dict=modalities_to_channel_groups_dict,
        max_sequence_length=max_sequence_length,
        base_patch_size=base_patch_size,
        use_channel_embs=use_channel_embs,
    )
    predictor = Predictor(
        encoder_embedding_size=embedding_size,
        decoder_embedding_size=embedding_size,
        modalities_to_channel_groups_dict=modalities_to_channel_groups_dict,
        depth=2,
        mlp_ratio=4.0,
        num_heads=2,
        max_sequence_length=max_sequence_length,
        max_patch_size=max_patch_size,
        learnable_channel_embeddings=use_channel_embs,
        output_embedding_size=embedding_size,
    )
    encoded_tokens = encoder.forward(
        x, patch_size, input_res, exit_after, token_exit_cfg
    )

    print(f"encoded_tokens.s2.shape: {encoded_tokens.s2.shape}")

    decoded_tokens = predictor.forward(
        encoded_tokens, timestamps, patch_size, input_res
    )
    print(f"decoded_tokens.s2.shape: {decoded_tokens.s2.shape}")
    print(f"decoded_tokens.s2.shape: {decoded_tokens.s2.shape}")
    print(f"decoded_tokens.s2.shape: {decoded_tokens.s2.shape}")
    print(f"decoded_tokens.s2.shape: {decoded_tokens.s2.shape}")
    print(f"decoded_tokens.s2.shape: {decoded_tokens.s2.shape}")
    print(f"decoded_tokens.s2.shape: {decoded_tokens.s2.shape}")
    print(f"decoded_tokens.s2.shape: {decoded_tokens.s2.shape}")
    print(f"decoded_tokens.s2.shape: {decoded_tokens.s2.shape}")
    print(f"decoded_tokens.s2.shape: {decoded_tokens.s2.shape}")
    print(f"decoded_tokens.s2.shape: {decoded_tokens.s2.shape}")
    print(f"decoded_tokens.s2.shape: {decoded_tokens.s2.shape}")
    print(f"decoded_tokens.s2.shape: {decoded_tokens.s2.shape}")
