"""Model code for the Helios model."""

import logging
import math
from typing import NamedTuple

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor, nn

from helios.constants import BASE_GSD
from helios.data.constants import Modality, ModalitySpec
from helios.nn.attention import Block
from helios.nn.encodings import (
    get_1d_sincos_pos_encoding,
    get_2d_sincos_pos_encoding_with_resolution,
    get_month_encoding_table,
)
from helios.train.masking import MaskedHeliosSample, MaskValue

logger = logging.getLogger(__name__)


class TokensAndMasks(NamedTuple):
    """Output to compute the loss on.

    Args:
        sentinel2: sentinel 2 data of shape (B, P_H, P_W, T, Band_Sets, D)
        sentinel2_mask: sentinel 2 mask indicating which tokens are masked/unmasked (B, P_H, P_W, T, Band_Sets)
        latlon: lat lon data containing geographical coordinates
        latlon_mask: lat lon mask indicating which coordinates are masked/unmasked
    """

    sentinel2: Tensor | None = None
    sentinel2_mask: Tensor | None = None
    sentinel1: Tensor | None = None
    sentinel1_mask: Tensor | None = None
    worldcover: Tensor | None = None
    worldcover_mask: Tensor | None = None
    latlon: Tensor | None = None
    latlon_mask: Tensor | None = None

    @property
    def device(self) -> torch.device:
        """Get the device of the tokens and masks."""
        if self.sentinel2 is not None:
            return self.sentinel2.device
        else:
            # look for any other modality that is not None
            for modality in self._fields:
                if getattr(self, modality) is not None:
                    return getattr(self, modality).device
            raise ValueError("No data to get device from")

    # TODO: It seems like we want a lot of our named tuples to have this functionality so we should probably create a utility base class for the named tuples and double subclass
    @classmethod
    def get_masked_modality_name(cls, modality: str) -> str:
        """Get the masked modality name."""
        return f"{modality}_mask"

    @property
    def modalities(self) -> list[str]:
        """Return all data fields."""
        return [x for x in self._fields if not x.endswith("mask")]

    def get_shape_dict(self) -> dict[str, tuple]:
        """Return a dictionary of the shapes of the fields."""
        return {x: getattr(self, x).shape for x in self._fields}

    @property
    def data_fields(self) -> list[str]:
        """Return all data fields."""
        return [
            x
            for x in self._fields
            if not x.endswith("mask") and getattr(self, x) is not None
        ]


class FlattenEmbed(nn.Module):
    """Embedding that flattens the input pixels."""

    def __init__(
        self,
        in_chans: int,
        embedding_size: int,
    ):
        """Create a new FlattenEmbed.

        Args:
            in_chans: number of input channels.
            embedding_size: the depth of the embedding.
        """
        super().__init__()
        self.in_chans = in_chans
        self.embedding_size = embedding_size

    def forward(self, x: Tensor, patch_size: int) -> Tensor:
        """Compute patch embeddings by flattening the pixel values in the patch.

        Args:
            x: Input tensor with shape [b, h, w, t, c]
            patch_size: patch size to use.
        """
        h_patches = x.shape[1] // patch_size
        w_patches = x.shape[2] // patch_size
        # Put c first here so that if c*hps*wps > embedding_size, at least we keep an
        # entire channel rather than missing some parts of the patch fully.
        flat_x = rearrange(
            x,
            "b (hp hps) (wp wps) t c -> b hp wp t (c hps wps)",
            hp=h_patches,
            wp=w_patches,
            hps=patch_size,
            wps=patch_size,
        )
        # Adjust to match embedding size.
        logger.info(
            f"compare the flat_x={flat_x.shape} to the embedding_size={self.embedding_size}"
        )
        if flat_x.shape[-1] > self.embedding_size:
            flat_x = flat_x[:, :, :, :, 0 : self.embedding_size]
        elif flat_x.shape[-1] < self.embedding_size:
            flat_x = F.pad(flat_x, (0, self.embedding_size - flat_x.shape[-1]))
        return flat_x


class FlexiHeliosPatchEmbeddings(nn.Module):
    """Module that patchifies and encodes the input data."""

    def __init__(
        self,
        supported_modality_names: list[str],
        max_patch_size: int,
        embedding_size: int,
    ):
        """Initialize the patch embeddings.

        Args:
            supported_modality_names: Which modalities from Modality this model
                instantiation supports
            max_patch_size: Maximum size of patches
            embedding_size: Size of embeddings
        """
        super().__init__()
        self.max_patch_size = max_patch_size
        self.embedding_size = embedding_size
        self.supported_modality_names = supported_modality_names
        # TODO: want to be able to remove certain bands and modalities
        self.per_modality_embeddings = nn.ModuleDict({})
        for modality in self.supported_modality_names:
            self.per_modality_embeddings[modality] = (
                self._get_patch_embedding_module_for_modality(modality)
            )

    @staticmethod
    def _get_embedding_module_name(modality: str, idx: int) -> str:
        """Get the embedding module name.

        Module Dicts require string keys
        """
        return f"{modality}__{idx}"

    def _get_patch_embedding_module_for_modality(self, modality: str) -> nn.Module:
        """Get the patch embedding module for a modality."""
        modality_spec = Modality.get(modality)
        return nn.ModuleDict(
            {
                self._get_embedding_module_name(modality, idx): FlattenEmbed(
                    in_chans=len(channel_set_idxs),
                    embedding_size=self.embedding_size,
                )
                for idx, channel_set_idxs in enumerate(
                    modality_spec.bandsets_as_indices()
                )
            }
        )

    # TODO: Likely we want a single object that stores all the data related configuration etc per modality including channel grous bands patch size etc
    def apply_embedding_to_modality(
        self, modality: str, input_data: MaskedHeliosSample, patch_size: int
    ) -> tuple[Tensor, Tensor]:
        """Apply embedding to a modality."""
        masked_modality_name = input_data.get_masked_modality_name(modality)
        modality_mask = getattr(input_data, masked_modality_name)
        modality_data = getattr(input_data, modality)

        modality_spec = Modality.get(modality)

        modality_tokens, modality_masks = [], []
        for idx, channel_set_indices in enumerate(modality_spec.bandsets_as_indices()):
            modality_specific_kwargs = {}
            if modality_spec.get_tile_resolution() == 0:
                # static in time
                token_mask = modality_mask[..., idx]
            else:
                token_mask = modality_mask[:, 0::patch_size, 0::patch_size, ..., idx]
                modality_specific_kwargs = {"patch_size": patch_size}
            patchified_dims = token_mask.shape[1:]
            # Now apply the embedding to
            if self.is_any_data_seen_by_encoder(token_mask):
                patchified_data = modality_data[..., channel_set_indices]

                patchified_data = self.per_modality_embeddings[modality][
                    self._get_embedding_module_name(modality, idx)
                ](patchified_data, **modality_specific_kwargs)
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

    def _get_modalities_to_process(self, input_data: MaskedHeliosSample) -> list[str]:
        """Get the modalities to process."""
        available_modalities = input_data.modalities
        modalities_to_process = set(self.supported_modality_names).intersection(
            set(available_modalities)
        )
        return list(modalities_to_process)

    def forward(
        self,
        input_data: MaskedHeliosSample,
        patch_size: int,
    ) -> dict[str, Tensor]:
        """Return flexibly patchified embeddings for each modality of the input data.

        Given a [B, H, W, (T), C] inputs, returns a [B, H, W, (T), b_s, D] output.

        We assume that the spatial masks are consistent for the given patch size,
        so that if patch_size == 2 then one possible mask would be
        [0, 0, 1, 1]
        [0, 0, 1, 1]
        [1, 1, 0, 0]
        [1, 1, 0, 0]
        for the H, W dimensions
        """
        output_dict = {}
        modalities_to_process = self._get_modalities_to_process(input_data)
        for modality in modalities_to_process:
            logger.debug(f"Processing modality {modality}")
            modality_tokens, modality_masks = self.apply_embedding_to_modality(
                modality, input_data, patch_size
            )
            output_dict[modality] = modality_tokens
            modality_mask_name = input_data.get_masked_modality_name(modality)
            output_dict[modality_mask_name] = modality_masks
        return output_dict


class FlexiHeliosCompositeEncodings(nn.Module):
    """Composite encodings for the FlexiHelios model."""

    def __init__(
        self,
        embedding_size: int,
        supported_modalities: list[str],
        max_sequence_length: int,
        base_patch_size: int,
        use_channel_embs: bool = True,
    ):
        """Initialize the composite encodings.

        Args:
            embedding_size: Size of token embeddings
            supported_modalities: Which modalities from Modality this model
                instantiation supports
            max_sequence_length: Maximum sequence length
            base_patch_size: Base patch size
            use_channel_embs: Whether to use learnable channel embeddings
        """
        super().__init__()
        self.embedding_size = embedding_size
        self.supported_modality_names = supported_modalities
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
                        len(Modality.get(modality).band_sets),
                        self.embedding_dim_per_embedding_type,
                    ),
                    **args,
                )
                for modality in self.supported_modality_names
            }
        )

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                # TODO: fix the dtype here
                nn.init.constant_(m.bias, 0).to(torch.float32)

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
        if modality == Modality.LATLON.name:
            return modality_tokens
        # TODO: Improve this implementation it is quite bad
        if modality_tokens.ndim == 3:
            # modality_tokens = [B, Band_Sets, D]; static in space, static in time
            b, b_s, _ = modality_tokens.shape
            # Static modality only needs channel embeddings
            modality_channel_embed = self.per_modality_channel_embeddings[modality]
            modality_channel_embed = repeat(
                modality_channel_embed, "b_s d -> b b_s d", b=b
            )
            modality_embed = F.pad(
                modality_channel_embed,
                (0, self.embedding_size - modality_channel_embed.shape[-1]),
            )
        # For temporal modalities like s1/s2
        if timestamps is None or patch_size is None or input_res is None:
            raise ValueError(
                f"timestamps, patch_size and input_res required for modality {modality}"
            )

        if modality_tokens.ndim == 5:
            raise NotImplementedError(
                f"Modality {modality} has no time dimension, not implemented"
            )
            # # worldcover should have no time dimension
            # b, h, w, b_s, _ = modality_tokens.shape
            # modality_channel_embed = self.per_modality_channel_embeddings[modality]
            # modality_channel_embed = repeat(
            #     modality_channel_embed, "b_s d -> b h w b_s d", b=b, h=h, w=w
            # )
            # gsd_ratio = self.calculate_gsd_ratio(input_res, patch_size)
            # current_device = modality_tokens.device
            # spatial_embed = get_2d_sincos_pos_encoding_with_resolution(
            #     grid_size=h,
            #     res=torch.ones(b, device=current_device) * gsd_ratio,
            #     encoding_dim=self.embedding_dim_per_embedding_type,
            #     device=current_device,
            # )
            # sp_zeros = torch.zeros(
            #     b,
            #     h,
            #     w,
            #     b_s,
            #     self.embedding_dim_per_embedding_type * 2,
            #     device=current_device,
            # )
            # spatial_embed = rearrange(spatial_embed, "b (h w) d -> b h w d", h=h, w=w)
            # spatial_embed = repeat(spatial_embed, "b h w d -> b h w b_s d", b_s=b_s)
            # modality_embed = torch.cat(
            #     [modality_channel_embed, sp_zeros, spatial_embed], dim=-1
            # )
        elif modality_tokens.ndim == 6:
            b, h, w, t, b_s, _ = modality_tokens.shape

            # Channel embeddings
            modality_channel_embed = self.per_modality_channel_embeddings[modality]
            modality_channel_embed = repeat(
                modality_channel_embed, "b_s d -> b h w t b_s d", b=b, h=h, w=w, t=t
            )

            # Time position encodings
            modality_pos_embed = repeat(
                self.pos_embed[:t], "t d -> b h w t b_s d", b=b, h=h, w=w, b_s=b_s
            )

            # Month encodings
            months = timestamps[:, :, 1]
            month_embed = self.month_embed(months)
            modality_month_embed = repeat(
                month_embed, "b t d -> b h w t b_s d", h=h, w=w, b_s=b_s
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
            spatial_embed = repeat(
                spatial_embed, "b h w d -> b h w t b_s d", b_s=b_s, t=t
            )

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
        else:
            raise ValueError(
                f"Unsupported tokens shape {modality_tokens.shape} for {modality}"
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
        output_dict = {}
        for modality in self.supported_modality_names:
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
        supported_modalities: list[ModalitySpec],
    ):
        """Initialize the FlexiHeliosBase class."""
        super().__init__()

        self.embedding_size = embedding_size
        self.supported_modality_names = [x.name for x in supported_modalities]
        logger.info(f"modalities being used by model: {self.supported_modality_names}")

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
            self.supported_modality_names,
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
    def collapse_and_combine_hwtc(self, x: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Collapse the tokens and masks, respectively, into two tensors."""
        tokens, masks = [], []
        for modality in self.supported_modality_names:
            masked_modality_name = MaskedHeliosSample.get_masked_modality_name(modality)
            x_modality = x[modality]
            x_modality_mask = x[masked_modality_name]
            flattened_tokens = rearrange(x_modality, "b ... d -> b (...) d")
            flattened_masks = rearrange(x_modality_mask, "b ... -> b (...)")
            num_tokens = flattened_tokens.shape[1]
            logger.debug(f"Modality {modality} has {num_tokens} tokens")
            tokens.append(flattened_tokens)
            masks.append(flattened_masks)
        tokens = torch.cat(tokens, dim=1)
        masks = torch.cat(masks, dim=1)
        # TODO: We should return the number of unmasked tokens processed by the encoder
        return tokens, masks

    def collapse_and_combine_tc(self, x: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Collapse the tokens and masks, respectively, into two tensors.

        This combines the batch/height/width dimensions so that attention is applied
        temporally but not spatially.
        """
        tokens, masks = [], []
        for modality in self.supported_modality_names:
            masked_modality_name = MaskedHeliosSample.get_masked_modality_name(modality)
            x_modality = x[modality]
            x_modality_mask = x[masked_modality_name]
            flattened_tokens = rearrange(
                x_modality, "b h w t b_s d -> (b h w) (t b_s) d"
            )
            flattened_masks = rearrange(
                x_modality_mask, "b h w t b_s -> (b h w) (t b_s)"
            )
            num_tokens = flattened_tokens.shape[1]
            logger.debug(f"Modality {modality} has {num_tokens} tokens")
            tokens.append(flattened_tokens)
            masks.append(flattened_masks)

        # Concatenate along temporal (token) dimension.
        # This only works when h/w are consistent across modalities.
        tokens = torch.cat(tokens, dim=1)
        masks = torch.cat(masks, dim=1)
        return tokens, masks

    def collapse_and_combine_hw(self, x: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Collapse the tokens and masks, respectively, into two tensors.

        This combines the batch/time dimensions so that attention is applied spatially
        but not temporally.
        """
        tokens, masks = [], []
        for modality in self.supported_modality_names:
            masked_modality_name = MaskedHeliosSample.get_masked_modality_name(modality)
            x_modality = x[modality]
            x_modality_mask = x[masked_modality_name]
            flattened_tokens = rearrange(
                x_modality, "b h w t b_s d -> (b t b_s) (h w) d"
            )
            flattened_masks = rearrange(
                x_modality_mask, "b h w t b_s -> (b t b_s) (h w)"
            )
            num_tokens = flattened_tokens.shape[1]
            logger.debug(f"Modality {modality} has {num_tokens} tokens")
            tokens.append(flattened_tokens)
            masks.append(flattened_masks)

        # Concatenate along temporal (batch) dimension.
        # This only works when h/w are consistent across modalities.
        tokens = torch.cat(tokens, dim=0)
        masks = torch.cat(masks, dim=0)
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

    def split_tokens_masks_and_dims(
        self, x: dict[str, Tensor]
    ) -> tuple[dict[str, Tensor], dict[str, Tensor], dict[str, tuple]]:
        """Split the tokens, masks, and dimensions out into separate dicts."""
        tokens_only_dict = {}
        original_masks_dict = {}
        modalities_to_dims_dict = {}
        # TODO: Should I have a dict like object that has methods that can return a mask or atoken here?
        for modality in self.supported_modality_names:
            x_modality = x[modality]
            tokens_only_dict[modality] = x_modality
            modalities_to_dims_dict[modality] = x_modality.shape
            masked_modality_name = MaskedHeliosSample.get_masked_modality_name(modality)
            original_masks_dict[masked_modality_name] = x[masked_modality_name]
        return tokens_only_dict, original_masks_dict, modalities_to_dims_dict

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

    @staticmethod
    def split_and_expand_per_modality_tc(
        x: Tensor, modalities_to_dims_dict: dict[str, tuple]
    ) -> dict[str, Tensor]:
        """Split and expand the tokens per modality.

        This is for tokens that were collapsed using collapse_and_combine_tc (for doing
        temporal attention only).

        Args:
            x: Tokens to split and expand (b*h*w t*b_s d)
            modalities_to_dims_dict: Dictionary mapping modalities to their dimensions
        Returns:
            tokens_only_dict: mapping modalities to their tokens
        """
        tokens_only_dict = {}
        tokens_reshaped = 0
        for modality, dims in modalities_to_dims_dict.items():
            # For now we only support 5D BHWTD data.
            batch, h, w, t, b_s, d = dims

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

            tokens_reshaped += num_tokens_for_modality
            tokens_only_dict[modality] = x_modality

        return tokens_only_dict

    @staticmethod
    def split_and_expand_per_modality_hw(
        x: Tensor, modalities_to_dims_dict: dict[str, tuple]
    ) -> dict[str, Tensor]:
        """Split and expand the tokens per modality.

        This is for tokens that were collapsed using collapse_and_combine_hw (for doing
        spatial attention only).

        Args:
            x: Tokens to split and expand (b*t*b_s h*w d)
            modalities_to_dims_dict: Dictionary mapping modalities to their dimensions
        Returns:
            tokens_only_dict: mapping modalities to their tokens
        """
        tokens_only_dict = {}
        tokens_reshaped = 0
        for modality, dims in modalities_to_dims_dict.items():
            # For now we only support 5D BHWTD data.
            batch, h, w, t, b_s, d = dims

            # Extract tokens for this modality (b*t*b_s h*w d).
            # Modalities are stacked on the temporal axis, which is part of the batch
            # dimension.
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
        supported_modalities: list[ModalitySpec],
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
            supported_modalities: list documenting modalities used in a given model instantiation
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
            supported_modalities=supported_modalities,
        )
        self.patch_embeddings = FlexiHeliosPatchEmbeddings(
            self.supported_modality_names,
            max_patch_size,
            embedding_size,
        )
        self.norm = nn.LayerNorm(embedding_size)
        self.pre_composite_embed = nn.Linear(embedding_size, embedding_size, bias=True)
        self.post_composite_embed = nn.Linear(embedding_size, embedding_size, bias=True)
        self.apply(self._init_weights)

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
        mask = torch.zeros_like(mask)
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
        x: dict[str, Tensor],
        timestamps: Tensor,
        patch_size: int,
        input_res: int,
    ) -> dict[str, Tensor]:
        """Apply the attention to the tokens and masks."""
        tokens_only_dict, original_masks_dict, modalities_to_dims_dict = (
            self.split_tokens_masks_and_dims(x)
        )

        for modality in self.supported_modality_names:
            tokens_only_dict[modality] = self.pre_composite_embed(
                tokens_only_dict[modality]
            )

        tokens_dict = self.composite_encodings.forward(
            tokens_only_dict,
            timestamps,
            patch_size,
            input_res,
        )
        x.update(tokens_dict)

        for modality in self.supported_modality_names:
            x[modality] = self.post_composite_embed(x[modality])

        print(
            "initial",
            x["sentinel2"][0, 0:5, 0:5, 0, 0, 0],
            x["sentinel2_mask"][0, 0:5, 0:5, 0, 0],
        )

        # Apply attn with varying encoder depths
        for block_idx, block in enumerate(self.blocks):
            # On even blocks, do temporal attention.
            # On odd blocks, do spatial attention.
            is_temporal_block = block_idx % 2 == 0

            if is_temporal_block:
                x, mask = self.collapse_and_combine_tc(x)
            else:
                x, mask = self.collapse_and_combine_hw(x)

            new_mask = mask >= MaskValue.TARGET_ENCODER_ONLY.value
            tokens, indices, new_mask = self.remove_masked_tokens(x, new_mask)

            # we take the inverse of the mask because a value
            # of True indicates the value *should* take part in
            # attention
            # WARNING: THIS MAY CHANGE DEPENDING ON THE ATTENTION IMPLEMENTATION
            tokens = block(x=tokens, y=None, attn_mask=~new_mask.bool())

            # we apply the norm before we add the removed tokens,
            # so that the norm is only computed against "real" tokens
            tokens = self.norm(tokens)

            # we don't care about the mask returned by add_removed_tokens, since we will
            # just use the original, unclipped mask here
            tokens, _ = self.add_removed_tokens(tokens, indices, new_mask)

            if is_temporal_block:
                x = self.split_and_expand_per_modality_tc(
                    tokens, modalities_to_dims_dict
                )
            else:
                x = self.split_and_expand_per_modality_hw(
                    tokens, modalities_to_dims_dict
                )

            # merge original masks and the processed tokens
            x.update(original_masks_dict)

            print(
                f"block {block_idx}",
                x["sentinel2"][0, 0:5, 0:5, 0, 0, 0],
                x["sentinel2_mask"][0, 0:5, 0:5, 0, 0],
                x["sentinel2"].shape,
            )

        return x

    def forward(
        self,
        x: MaskedHeliosSample,
        patch_size: int,
        input_res: int = BASE_GSD,
    ) -> TokensAndMasks:
        """Process masked input samples into token representations.

        Args:
            x: Masked input sample containing the data to be encoded
            patch_size: Size of patches to divide the input into
            input_res: Resolution of the input data

        Returns:
            TokensAndMasks containing the encoded representations and their masks
        """
        patchified_tokens_and_masks = self.patch_embeddings.forward(x, patch_size)
        patchified_tokens_and_masks = self.apply_attn(
            x=patchified_tokens_and_masks,
            timestamps=x.timestamps,
            patch_size=patch_size,
            input_res=input_res,
        )
        if any(
            patchified_tokens_and_masks[modality] is None
            for modality in self.supported_modality_names
        ):
            raise ValueError(
                f"Some supported modalities are None {patchified_tokens_and_masks}"
            )
        return TokensAndMasks(**patchified_tokens_and_masks)


class Predictor(FlexiHeliosBase):
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
        max_patch_size: int = 8,
        drop_path: float = 0.0,
        learnable_channel_embeddings: bool = False,
        output_embedding_size: int | None = None,
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
            supported_modalities=supported_modalities,
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

    def add_masks(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        """Replace tokens that should be decoded (MaskValue.DECODER_ONLY) with the learnable mask token.

        in a dimension-agnostic way using einops. We assume the final dimension of each token tensor
        is the embedding dimension matching self.mask_token's size.
        """
        output_dict = {}
        for modality in self.supported_modality_names:
            x_modality = x[modality]
            mask_modality = x[MaskedHeliosSample.get_masked_modality_name(modality)]

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
        x: dict[str, Tensor],
        timestamps: Tensor,
        patch_size: int,
        input_res: int,
    ) -> dict[str, Tensor]:
        """Apply the attention to the tokens and masks."""
        tokens_only_dict, original_masks_dict, modalities_to_dims_dict = (
            self.split_tokens_masks_and_dims(x)
        )
        tokens_dict = self.composite_encodings(
            tokens_only_dict, timestamps, patch_size, input_res
        )
        x.update(tokens_dict)

        for block_idx, block in enumerate(self.blocks):
            # On even blocks, do temporal attention.
            # On odd blocks, do spatial attention.
            is_temporal_block = block_idx % 2 == 0

            if is_temporal_block:
                x, mask = self.collapse_and_combine_tc(x)
            else:
                x, mask = self.collapse_and_combine_hw(x)

            x, y, x_mask, y_mask, indices = self.split_x_y(x, mask)

            # note that we are not taking the inverse of the mask, since split_x_y gives us
            # true values for values we want to take part in attention
            x = block(x=x, y=y, attn_mask=y_mask.bool())

            x = self.combine_x_y(x, y, x_mask, y_mask, indices)

            if is_temporal_block:
                x = self.split_and_expand_per_modality_tc(x, modalities_to_dims_dict)
            else:
                x = self.split_and_expand_per_modality_hw(x, modalities_to_dims_dict)

            x.update(original_masks_dict)

            print(f"predictor block {block_idx}", x["sentinel2"][0, 0:5, 0:5, 0, 0, 0])

        return x

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
        decoder_emedded_dict = x._asdict()
        # Apply Input Norms and encoder to decoder embeds to each modality
        for modality in self.supported_modality_names:
            x_modality = getattr(x, modality)
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
        for modality in self.supported_modality_names:
            masked_modality_name = MaskedHeliosSample.get_masked_modality_name(modality)
            modality_mask = tokens_and_masks[masked_modality_name]
            # patchify masked data
            per_modality_output_tokens = []
            modality_data = tokens_and_masks[modality]
            # modality_specific_dims = self.grab_modality_specific_dims(modality_data)

            band_sets = Modality.get(modality).band_sets
            for idx in range(len(band_sets)):
                per_channel_modality_data = modality_data[..., idx, :]
                output_data = self.to_output_embed(
                    # self.norm(per_channel_modality_data)
                    per_channel_modality_data
                )
                per_modality_output_tokens.append(output_data)
            output_dict[modality] = torch.stack(per_modality_output_tokens, dim=-2)
            output_dict[masked_modality_name] = modality_mask
        return TokensAndMasks(**output_dict)
