"""Model code for the Helios model."""

import logging
import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, NamedTuple

import torch
from einops import rearrange, repeat
from olmo_core.config import Config
from torch import Tensor, nn
from torch.distributed.fsdp import fully_shard

from helios.data.constants import BASE_GSD, Modality, ModalitySpec
from helios.dataset.utils import get_modality_specs_from_names
from helios.nn.attention import Block
from helios.nn.encodings import (
    get_1d_sincos_pos_encoding,
    get_2d_sincos_pos_encoding_with_resolution,
    get_month_encoding_table,
)
from helios.nn.flexi_patch_embed import FlexiPatchEmbed, FlexiPatchReconstruction
from helios.nn.utils import get_cumulative_sequence_lengths
from helios.train.masking import MaskedHeliosSample, MaskValue

logger = logging.getLogger(__name__)


def get_modalities_to_process(
    available_modalities: list[str], supported_modality_names: list[str]
) -> list[str]:
    """Get the modalities to process."""
    modalities_to_process = set(supported_modality_names).intersection(
        set(available_modalities)
    )
    return list(modalities_to_process)


def return_modalities_from_dict(
    per_modality_input_tokens: dict[str, Tensor],
) -> list[str]:
    """Return the modalities from a dictionary of per modality input tokens."""
    return [
        key for key in per_modality_input_tokens.keys() if not key.endswith("_mask")
    ]


class PoolingType(StrEnum):
    """Strategy for pooling the tokens."""

    MAX = "max"
    MEAN = "mean"


class TokensAndMasks(NamedTuple):
    """Output to compute the loss on.

    Args:
        sentinel2: sentinel 2 data of shape (B, P_H, P_W, T, Band_Sets, D)
        sentinel2_mask: sentinel 2 mask indicating which tokens are masked/unmasked (B, P_H, P_W, T, Band_Sets)
        sentinel1: sentinel 1 data of shape (B, P_H, P_W, T, Band_Sets, D)
        sentinel1_mask: sentinel 1 mask indicating which tokens are masked/unmasked (B, P_H, P_W, T, Band_Sets)
        worldcover: worldcover data of shape (B, P_H, P_W, T, Band_Sets, D)
        worldcover_mask: worldcover mask indicating which tokens are masked/unmasked (B, P_H, P_W, T, Band_Sets)
        latlon: lat lon data containing geographical coordinates
        latlon_mask: lat lon mask indicating which coordinates are masked/unmasked
        openstreetmap_raster: openstreetmap raster data of shape (B, P_H, P_W, T, Band_Sets, D)
        openstreetmap_raster_mask: openstreetmap raster mask indicating which tokens are masked/unmasked (B, P_H, P_W, T, Band_Sets)
    """

    sentinel2_l2a: Tensor | None = None
    sentinel2_l2a_mask: Tensor | None = None
    sentinel1: Tensor | None = None
    sentinel1_mask: Tensor | None = None
    worldcover: Tensor | None = None
    worldcover_mask: Tensor | None = None
    latlon: Tensor | None = None
    latlon_mask: Tensor | None = None
    openstreetmap_raster: Tensor | None = None
    openstreetmap_raster_mask: Tensor | None = None
    srtm: Tensor | None = None
    srtm_mask: Tensor | None = None
    landsat: Tensor | None = None
    landsat_mask: Tensor | None = None
    naip: Tensor | None = None
    naip_mask: Tensor | None = None
    naip_10: Tensor | None = None
    naip_10_mask: Tensor | None = None
    gse: Tensor | None = None
    gse_mask: Tensor | None = None
    cdl: Tensor | None = None
    cdl_mask: Tensor | None = None
    worldpop: Tensor | None = None
    worldpop_mask: Tensor | None = None
    worldcereal: Tensor | None = None
    worldcereal_mask: Tensor | None = None
    wri_canopy_height_map: Tensor | None = None
    wri_canopy_height_map_mask: Tensor | None = None
    era5_10: Tensor | None = None
    era5_10_mask: Tensor | None = None

    @property
    def device(self) -> torch.device:
        """Get the device of the tokens and masks."""
        if self.sentinel2_l2a is not None:
            return self.sentinel2_l2a.device
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

    def as_dict(self, return_none: bool = True) -> dict[str, Any]:
        """Convert the namedtuple to a dictionary.

        Returns:
            Dictionary representation of the namedtuple.
        """
        return_dict = {}
        for field in self._fields:
            val = getattr(self, field)
            if return_none:
                return_dict[field] = val
            else:
                if val is not None:
                    return_dict[field] = val
        return return_dict

    @property
    def modalities(self) -> list[str]:
        """Return all data fields."""
        return [
            x
            for x in self._fields
            if not x.endswith("mask") and getattr(self, x) is not None
        ]

    def get_shape_dict(self) -> dict[str, tuple]:
        """Return a dictionary of the shapes of the fields."""
        return {x: getattr(self, x).shape for x in self._fields}

    @staticmethod
    def _flatten(x: Tensor) -> Tensor:
        return rearrange(x, "b ... d -> b (...) d")

    def flatten_tokens_and_masks(
        self, return_lists: bool = False
    ) -> tuple[Tensor, Tensor]:
        """Return the flattened tokens and masks.

        Args:
            return_lists: If True, return the original lists before concatenation.
                          If False, return concatenated tensors.

        Tokens will have shape [B, T, D] and masks will have shape [B, T]
        """
        flattened_x, flattened_masks = [], []
        for attr_name in self.modalities:
            mask_attr_name = self.get_masked_modality_name(attr_name)
            attr = getattr(self, attr_name)
            masked_attr = getattr(self, mask_attr_name)
            if attr is not None:
                if masked_attr is None:
                    raise ValueError(
                        f"Can't have present {attr_name} but None {mask_attr_name}"
                    )
                masked_attr = masked_attr.unsqueeze(dim=-1)
                flattened_x.append(self._flatten(attr))
                flattened_masks.append(self._flatten(masked_attr))

        if return_lists:
            # Remove the extra dimension from the masks
            flattened_masks = [mask[:, :, 0] for mask in flattened_masks]
            return flattened_x, flattened_masks

        x = torch.cat(flattened_x, dim=1)
        masks = torch.cat(flattened_masks, dim=1)[:, :, 0]
        return x, masks

    def pool_unmasked_tokens(
        self,
        pooling_type: PoolingType = PoolingType.MAX,
        spatial_pooling: bool = False,
        concat_features: bool = False,
    ) -> Tensor:
        """Pool the unmasked tokens.

        Args:
            pooling_type: Pooling type for the tokens
            spatial_pooling: Whether to keep the spatial dimensions when pooling. If true,
                this expects the masks within a spatial modality to be consistent (e.g. all
                s2 tokens would have the same mask.)
            concat_features: Whether to concatenate the features instead of averaging them, only enabled for spatial pooling as of now,
            requires no masked out tokens
        """
        if concat_features and spatial_pooling:
            spatial_stacked_features = []
            for attr_name in self.modalities:
                if Modality.get(attr_name).is_spatial:
                    mask_attr_name = self.get_masked_modality_name(attr_name)
                    masked_attr = getattr(self, mask_attr_name)
                    if masked_attr is None:
                        continue
                    if (masked_attr == MaskValue.ONLINE_ENCODER.value).all():
                        attr = getattr(self, attr_name)
                        # only mean in temporal dimension
                        pooled_attr = torch.mean(attr, dim=(-3))
                        spatial_stacked_features.append(pooled_attr)
            if len(spatial_stacked_features) == 0:
                raise ValueError(
                    "Missing unmasked spatial modalities for spatial pooling."
                )
            # Concatenate along the band sets dimension instead of stacking
            spatial_stacked_features = torch.cat(spatial_stacked_features, dim=-2)
            # flatten the last 3 dimensions
            return rearrange(spatial_stacked_features, "b h w c d-> b h w c d")
        if concat_features:
            raise ValueError("concat_features is not supported for non-spatial pooling")
        if not spatial_pooling:
            x, mask = self.flatten_tokens_and_masks()
            # 1s for online encoder, 0s elsewhere
            mask = (mask == MaskValue.ONLINE_ENCODER.value).long()
            x_for_pooling = x * mask.unsqueeze(-1)
            if pooling_type == PoolingType.MAX:
                x_for_pooling = x_for_pooling.masked_fill(
                    ~mask.bool().unsqueeze(-1), -float("inf")
                )
                return x_for_pooling.max(dim=1).values
            elif pooling_type == PoolingType.MEAN:
                num_encoded_tokens = torch.sum(mask, -1, keepdim=True)
                logger.debug(f"num_encoded_tokens: {num_encoded_tokens}")
                if (num_encoded_tokens == 0).any():
                    raise ValueError(
                        f"num_encoded_tokens is 0 for some samples {num_encoded_tokens}"
                    )
                return x_for_pooling.sum(dim=1) / num_encoded_tokens
            else:
                raise ValueError(f"Invalid pooling type: {pooling_type}")
        else:
            spatial_average = []
            for attr_name in self.modalities:
                if Modality.get(attr_name).is_spatial:
                    mask_attr_name = self.get_masked_modality_name(attr_name)
                    masked_attr = getattr(self, mask_attr_name)
                    if masked_attr is None:
                        continue
                    if (masked_attr == MaskValue.ONLINE_ENCODER.value).all():
                        attr = getattr(self, attr_name)
                        # pool across time and bandset dimensions
                        if pooling_type == PoolingType.MEAN:
                            spatial_average.append(torch.mean(attr, dim=(-2, -3)))
                        else:
                            spatial_average.append(
                                torch.max(torch.max(attr, dim=-2).values, dim=-2).values
                            )
            if len(spatial_average) == 0:
                raise ValueError(
                    "Missing unmasked spatial modalities for spatial pooling."
                )
            spatial_average_t = torch.stack(spatial_average, dim=-1)
            if pooling_type == PoolingType.MEAN:
                return spatial_average_t.mean(dim=-1)
            else:
                return spatial_average_t.max(dim=-1).values


class ProjectAndAggregate(nn.Module):
    """Module that applies a linear projection to tokens and masks."""

    def __init__(
        self,
        embedding_size: int,
        num_layers: int,
        aggregate_then_project: bool = True,
    ):
        """Initialize the linear module.

        embedding_size: The embedding size of the input TokensAndMasks
        num_layers: The number of layers to use in the projection. If >1, then
            a ReLU activation will be applied between layers
        aggregate_then_project: If True, then we will average the tokens before applying
            the projection. If False, we will apply the projection first.
        """
        super().__init__()
        projections = [nn.Linear(embedding_size, embedding_size)]
        for _ in range(1, num_layers):
            projections.append(nn.ReLU())
            projections.append(nn.Linear(embedding_size, embedding_size))
        self.projection = nn.Sequential(*projections)
        self.aggregate_then_project = aggregate_then_project

    def forward(self, x: TokensAndMasks) -> torch.Tensor:
        """Apply a (non)linear projection to an input TokensAndMasks.

        This can be applied either before or after pooling the tokens.
        """
        if self.aggregate_then_project:
            pooled_for_contrastive = x.pool_unmasked_tokens(
                PoolingType.MEAN, spatial_pooling=False
            )
            return self.projection(pooled_for_contrastive)
        else:
            decoder_emedded_dict = x._asdict()
            for modality in x.modalities:
                x_modality = getattr(x, modality)
                # Are these normalizations masked correctly?
                x_modality = self.projection(x_modality)
                masked_modality_name = x.get_masked_modality_name(modality)
                decoder_emedded_dict[modality] = x_modality
                decoder_emedded_dict[masked_modality_name] = getattr(
                    x, masked_modality_name
                )
            x_projected = TokensAndMasks(**decoder_emedded_dict)
            return x_projected.pool_unmasked_tokens(
                PoolingType.MEAN, spatial_pooling=False
            )


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

        # For every patch embedding module we want to create a unique
        for modality in self.supported_modality_names:
            for idx, bandset_indices in enumerate(
                Modality.get(modality).bandsets_as_indices()
            ):
                buffer_name = self._get_buffer_name(modality, idx)
                banset_indices_tensor = torch.tensor(bandset_indices, dtype=torch.long)
                self.register_buffer(
                    buffer_name, banset_indices_tensor, persistent=False
                )

        # Create a dictionary of per modality index tensors to do  index select with registered buffer

    @staticmethod
    def _get_buffer_name(modality: str, idx: int) -> str:
        """Get the buffer name."""
        return f"{modality}__{idx}_buffer"

    @staticmethod
    def _get_embedding_module_name(modality: str, idx: int) -> str:
        """Get the embedding module name.

        Module Dicts require string keys
        """
        return f"{modality}__{idx}"

    def _get_patch_embedding_module_for_modality(self, modality: str) -> nn.Module:
        """Get the patch embedding module for a modality."""
        modality_spec = Modality.get(modality)
        # Based on the modality name we choose the way to embed the data

        # I likely will need to know about what the embedding strategy is in the forward as well
        # Static modality
        if not modality_spec.is_spatial:
            # static in space
            return nn.ModuleDict(
                {
                    self._get_embedding_module_name(modality, idx): nn.Linear(
                        len(channel_set_idxs), self.embedding_size
                    )
                    for idx, channel_set_idxs in enumerate(
                        modality_spec.bandsets_as_indices()
                    )
                }
            )
        else:
            return nn.ModuleDict(
                {
                    self._get_embedding_module_name(modality, idx): FlexiPatchEmbed(
                        in_chans=len(channel_set_idxs),
                        embedding_size=self.embedding_size,
                        patch_size_at_16=self.max_patch_size,
                        modality_spec=modality_spec,
                    )
                    for idx, channel_set_idxs in enumerate(
                        modality_spec.bandsets_as_indices()
                    )
                }
            )

    def apply_embedding_to_modality(
        self,
        modality: str,
        input_data: MaskedHeliosSample,
        patch_size: int,
        fast_pass: bool = False,
    ) -> tuple[Tensor, Tensor]:
        """Apply embedding to a modality."""
        logger.debug(f"applying embedding to modality:{modality}")
        masked_modality_name = input_data.get_masked_modality_name(modality)
        modality_mask = getattr(input_data, masked_modality_name)
        modality_data = getattr(input_data, modality)

        modality_spec = Modality.get(modality)

        modality_tokens, modality_masks = [], []
        for idx in range(modality_spec.num_band_sets):
            modality_specific_kwargs = {}
            if not modality_spec.is_spatial:
                # static in time
                token_mask = modality_mask[..., idx]
            else:
                token_mask = modality_mask[
                    :,
                    0 :: patch_size * modality_spec.image_tile_size_factor,
                    0 :: patch_size * modality_spec.image_tile_size_factor,
                    ...,
                    idx,
                ]
                modality_specific_kwargs = {"patch_size": patch_size}
            # In the fast pass we want to the sync that comes with checking for online encoder
            if fast_pass or (token_mask == MaskValue.ONLINE_ENCODER.value).any():
                buffer_name = self._get_buffer_name(modality, idx)
                patchified_data = torch.index_select(
                    modality_data, -1, getattr(self, buffer_name)
                )
                embedding_module = self.per_modality_embeddings[modality][
                    self._get_embedding_module_name(modality, idx)
                ]
                patchified_data = embedding_module(
                    patchified_data, **modality_specific_kwargs
                )
            else:
                mask_shape = token_mask.shape + (self.embedding_size,)
                patchified_data = torch.zeros(
                    mask_shape, dtype=token_mask.dtype, device=token_mask.device
                )

            modality_tokens.append(patchified_data)
            modality_masks.append(token_mask)
        return torch.stack(modality_tokens, dim=-2), torch.stack(modality_masks, dim=-1)

    @staticmethod
    def is_any_data_seen_by_encoder(modality_mask: Tensor) -> bool:
        """Check if any data is seen by the encoder."""
        return (MaskValue.ONLINE_ENCODER.value == modality_mask).any()

    def apply_compile(self) -> None:
        """Apply torch.compile to the model."""
        self.compile(dynamic=False, mode="max-autotune-no-cudagraphs", fullgraph=True)

    def forward(
        self,
        input_data: MaskedHeliosSample,
        patch_size: int,
        fast_pass: bool = False,
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
        modalities_to_process = get_modalities_to_process(
            input_data.modalities, self.supported_modality_names
        )
        for modality in modalities_to_process:
            modality_tokens, modality_masks = self.apply_embedding_to_modality(
                modality, input_data, patch_size, fast_pass
            )
            output_dict[modality] = modality_tokens
            modality_mask_name = input_data.get_masked_modality_name(modality)
            output_dict[modality_mask_name] = modality_masks
        return output_dict


class Reconstructor(nn.Module):
    """Module that patchifies and encodes the input data."""

    def __init__(
        self,
        decoder: nn.Module,
        supported_modalities: list[ModalitySpec],
        max_patch_size: int,
    ):
        """Initialize the patch embeddings.

        Args:
            decoder: Predictor nn module to use on before reconstructor on input
            supported_modalities: Which modalities from Modality this model
                instantiation supports
            max_patch_size: Maximum size of patches
        """
        super().__init__()
        self.max_patch_size = max_patch_size
        self.embedding_size = decoder.output_embedding_size
        self.supported_modalities = supported_modalities
        self.decoder = decoder
        # TODO: want to be able to remove certain bands and modalities
        self.per_modality_reconstructions = nn.ModuleDict({})
        for modality in self.supported_modalities:
            self.per_modality_reconstructions[modality.name] = (
                self._get_patch_reconstruction_module_for_modality(modality)
            )

    def apply_compile(self) -> None:
        """Apply torch.compile to the model."""
        self.decoder.apply_compile()

    def apply_fsdp(self, **fsdp_kwargs: Any) -> None:
        """Apply FSDP to the model."""
        self.decoder.apply_fsdp(**fsdp_kwargs)

    @staticmethod
    def _get_reconstruction_module_name(modality: str, idx: int) -> str:
        """Get the reconstruction module name.

        Module Dicts require string keys
        """
        return f"{modality}__{idx}"

    def _get_patch_reconstruction_module_for_modality(
        self, modality: ModalitySpec
    ) -> nn.Module:
        """Get the patch reconstruction module for a modality."""
        # Based on the modality name we choose the way to embed the data

        # I likely will need to know about what the embedding strategy is in the forward as well
        # Static modality
        if modality.get_tile_resolution() == 0:
            # static in space
            return nn.ModuleDict(
                {
                    self._get_reconstruction_module_name(modality.name, idx): nn.Linear(
                        self.embedding_size, len(channel_set_idxs)
                    )
                    for idx, channel_set_idxs in enumerate(
                        modality.bandsets_as_indices()
                    )
                }
            )
        else:
            return nn.ModuleDict(
                {
                    self._get_reconstruction_module_name(
                        modality.name, idx
                    ): FlexiPatchReconstruction(
                        out_chans=len(channel_set_idxs),
                        embedding_size=self.embedding_size,
                        max_patch_size=self.max_patch_size,
                    )
                    for idx, channel_set_idxs in enumerate(
                        modality.bandsets_as_indices()
                    )
                }
            )

    # TODO: Likely we want a single object that stores all the data related configuration etc per modality including channel grous bands patch size etc
    def apply_reconstruction_to_modality(
        self, modality: str, input_data: TokensAndMasks, patch_size: int
    ) -> tuple[Tensor, Tensor]:
        """Apply reconstruction to a modality."""
        masked_modality_name = input_data.get_masked_modality_name(modality)
        modality_mask = getattr(input_data, masked_modality_name)
        modality_data = getattr(input_data, modality)

        modality_spec = Modality.get(modality)

        # x: Input tensor with shape [b, h, w, (t), b_s, d]
        modality_tokens, modality_masks = [], []
        for idx, channel_set_indices in enumerate(modality_spec.bandsets_as_indices()):
            data = modality_data[..., idx, :]
            masks = modality_mask[..., idx]
            r_model = self.per_modality_reconstructions[modality][
                self._get_reconstruction_module_name(modality, idx)
            ]
            if modality_spec.get_tile_resolution() == 0:
                data = r_model(data)
            else:
                data = r_model(data, patch_size=patch_size)
            modality_tokens.append(data)
            masks = repeat(
                masks,
                "b h w ... -> b (h p_h) (w p_w) ...",
                p_h=patch_size,
                p_w=patch_size,
            )
            modality_masks.append(masks)
        modality_mask = repeat(
            modality_mask,
            "b h w ... -> b (h p_h) (w p_w) ...",
            p_h=patch_size,
            p_w=patch_size,
        )
        return torch.cat(modality_tokens, dim=-1), modality_mask

    def forward(
        self,
        x: TokensAndMasks,
        timestamps: Tensor,
        patch_size: int,
        input_res: int = BASE_GSD,
    ) -> TokensAndMasks:
        """Return flexibly patchified reconstruction for each modality of the input data.

        Given a [B, H, W, (T), b_s, D] inputs, returns a [B, H, W, (T), C] output.
        """
        input_data = self.decoder(x, timestamps, patch_size, input_res)
        output_dict = {}
        modalities_to_process = get_modalities_to_process(
            input_data.modalities, [m.name for m in self.supported_modalities]
        )
        for modality in modalities_to_process:
            modality_tokens, modality_masks = self.apply_reconstruction_to_modality(
                modality, input_data, patch_size
            )
            output_dict[modality] = modality_tokens
            modality_mask_name = input_data.get_masked_modality_name(modality)
            output_dict[modality_mask_name] = modality_masks
        return TokensAndMasks(**output_dict)


@dataclass
class ReconstructorConfig(Config):
    """Configuration for the Reconstructor."""

    decoder_config: "Config"
    supported_modality_names: list[str]
    max_patch_size: int = 8

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

    def build(self) -> "Reconstructor":
        """Build the reconstructor."""
        self.validate()
        kwargs = self.as_dict(exclude_none=True, recurse=False)
        kwargs.pop("supported_modality_names")
        kwargs["supported_modalities"] = self.supported_modalities
        kwargs.pop("decoder_config")
        kwargs["decoder"] = self.decoder_config.build()
        logger.info(f"Predictor kwargs: {kwargs}")
        return Reconstructor(**kwargs)


class FlexiHeliosCompositeEncodings(nn.Module):
    """Composite encodings for the FlexiHelios model."""

    def __init__(
        self,
        embedding_size: int,
        supported_modalities: list[ModalitySpec],
        max_sequence_length: int,
        learnable_channel_embeddings: bool = True,
        random_channel_embeddings: bool = False,
    ):
        """Initialize the composite encodings.

        Args:
            embedding_size: Size of token embeddings
            supported_modalities: Which modalities from Modality this model
                instantiation supports
            max_sequence_length: Maximum sequence length
            learnable_channel_embeddings: Whether to use learnable channel embeddings
            random_channel_embeddings: Initialize channel embeddings randomly (zeros if False)
        """
        super().__init__()
        self.embedding_size = embedding_size
        self.supported_modalities = supported_modalities
        self.supported_modality_names = [
            modality.name for modality in supported_modalities
        ]
        self.embedding_size = embedding_size
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
        # Month encodings
        month_tab = get_month_encoding_table(self.embedding_dim_per_embedding_type)
        self.month_embed = nn.Embedding.from_pretrained(month_tab, freeze=True)
        if not learnable_channel_embeddings and not random_channel_embeddings:
            self.per_modality_channel_embeddings = nn.ParameterDict()
            for modality in self.supported_modalities:
                shape = (len(modality.band_sets), self.embedding_dim_per_embedding_type)
                channel_embeddings = nn.Parameter(
                    torch.zeros(shape), requires_grad=False
                )
                self.per_modality_channel_embeddings[modality.name] = channel_embeddings
        else:
            # Channel embeddings
            if learnable_channel_embeddings:
                args = {"requires_grad": True}
            else:
                args = {"requires_grad": False}

            self.per_modality_channel_embeddings = nn.ParameterDict()
            for modality in self.supported_modalities:
                shape = (len(modality.band_sets), self.embedding_dim_per_embedding_type)
                if random_channel_embeddings:
                    channel_embeddings = nn.Parameter(torch.rand(shape), **args)
                else:
                    channel_embeddings = nn.Parameter(torch.zeros(shape), **args)
                self.per_modality_channel_embeddings[modality.name] = channel_embeddings

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
        modality_name: str,
        modality_tokens: Tensor,
        timestamps: Tensor | None = None,
        patch_size: int | None = None,
        input_res: int | None = None,
        use_modality_encodings: bool = True,
        use_temporal_encodings: bool = True,
    ) -> Tensor:
        """Apply the encodings to the patchified data based on modality type.

        Args:
            modality_name: Name of the modality being processed
            modality_tokens: Token embeddings for the modality
            timestamps: Optional timestamps for temporal encodings
            patch_size: Optional patch size for spatial encodings
            input_res: Optional input resolution for spatial encodings
            use_modality_encodings: Whether to use modality encodings
            use_temporal_encodings: Whether to use temporal encodings

        Returns:
            Tensor with encodings applied based on modality type
        """
        logger.debug(
            f"use_modality_encodings: {use_modality_encodings}, use_temporal_encodings: {use_temporal_encodings}"
        )
        # TODO: Improve this implementation it is quite bad

        modality = Modality.get(modality_name)
        logger.debug(f"Applying encodings to modality {modality}")
        if not use_modality_encodings and use_temporal_encodings:
            b, h, w, t, _ = modality_tokens.shape
            ein_string, ein_dict = (
                "b h w t d",
                {"b": b, "h": h, "w": w, "t": t},
            )
        elif not use_temporal_encodings and not use_modality_encodings:
            b, h, w, _ = modality_tokens.shape
            ein_string, ein_dict = (
                "b h w d",
                {"b": b, "h": h, "w": w},
            )
        elif not use_temporal_encodings and use_modality_encodings:
            raise NotImplementedError("Not implemented")
        else:
            if modality_tokens.ndim == 3:
                # modality_tokens = [B, Band_Sets, D]; static in space, static in time
                b, b_s, _ = modality_tokens.shape
                ein_string, ein_dict = "b b_s d", {"b": b, "b_s": b_s}
            elif modality_tokens.ndim == 4:
                b, t, b_s, _ = modality_tokens.shape
                ein_string, ein_dict = "b t b_s d", {"b": b, "t": t, "b_s": b_s}
            elif modality_tokens.ndim == 5:
                b, h, w, b_s, _ = modality_tokens.shape
                ein_string, ein_dict = (
                    "b h w b_s d",
                    {"b": b, "h": h, "w": w, "b_s": b_s},
                )
            elif modality_tokens.ndim == 6:
                b, h, w, t, b_s, _ = modality_tokens.shape
                ein_string, ein_dict = (
                    "b h w t b_s d",
                    {"b": b, "h": h, "w": w, "t": t, "b_s": b_s},
                )
            else:
                raise ValueError(f"Unsupported tokens shape: {modality_tokens.shape}")

        device = modality_tokens.device
        modality_embed = torch.zeros(modality_tokens.shape, device=device)
        n = self.embedding_dim_per_embedding_type

        # Channel embeddings
        if use_modality_encodings:
            channel_embed = self.per_modality_channel_embeddings[modality.name]
            channel_embed = repeat(
                channel_embed, f"b_s d -> {ein_string}", **ein_dict
            ).to(device)
            modality_embed[..., :n] += channel_embed

        if modality.is_multitemporal and use_temporal_encodings:
            # Time position encodings
            time_embed = repeat(self.pos_embed[:t], f"t d -> {ein_string}", **ein_dict)
            modality_embed[..., n : n * 2] += time_embed.to(device)

            # Month encodings
            assert timestamps is not None
            months = timestamps[:, :, 1]
            month_embed = self.month_embed(months)
            month_embed = repeat(month_embed, f"b t d -> {ein_string}", **ein_dict)
            modality_embed[..., n * 2 : n * 3] += month_embed.to(device)
        if modality.is_spatial:
            # Spatial encodings
            assert input_res is not None
            assert patch_size is not None
            gsd_ratio = self.calculate_gsd_ratio(input_res, patch_size)
            spatial_embed = get_2d_sincos_pos_encoding_with_resolution(
                grid_size=h,
                res=torch.ones(b, device=device) * gsd_ratio,
                encoding_dim=self.embedding_dim_per_embedding_type,
                device=device,
            )
            spatial_embed = rearrange(spatial_embed, "b (h w) d -> b h w d", h=h, w=w)
            spatial_embed = repeat(
                spatial_embed, f"b h w d -> {ein_string}", **ein_dict
            )
            modality_embed[..., n * 3 : n * 4] += spatial_embed
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
        available_modalities = return_modalities_from_dict(per_modality_input_tokens)
        modalities_to_process = get_modalities_to_process(
            available_modalities, self.supported_modality_names
        )
        for modality_name in modalities_to_process:
            output_dict[modality_name] = self._apply_encodings_per_modality(
                modality_name,
                per_modality_input_tokens[modality_name],
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
        num_heads: int,
        mlp_ratio: float,
        depth: int,
        drop_path: float,
        supported_modalities: list[ModalitySpec],
        learnable_channel_embeddings: bool = True,
        random_channel_embeddings: bool = False,
        use_flash_attn: bool = False,
        qk_norm: bool = False,
    ) -> None:
        """Initialize the FlexiHeliosBase class."""
        super().__init__()

        self.embedding_size = embedding_size
        self.supported_modalities = supported_modalities
        self.supported_modality_names = [x.name for x in supported_modalities]
        logger.info(f"modalities being used by model: {self.supported_modality_names}")

        self.max_sequence_length = max_sequence_length

        self.use_flash_attn = use_flash_attn
        self.learnable_channel_embeddings = learnable_channel_embeddings
        self.random_channel_embeddings = random_channel_embeddings
        self.blocks = nn.ModuleList(
            [
                Block(
                    embedding_size,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    qk_norm=qk_norm,
                    norm_layer=nn.LayerNorm,  # TODO: This should be configurable
                    cross_attn=self.cross_attn,
                    drop_path=drop_path,
                    use_flash_attn=self.use_flash_attn,
                )
                for _ in range(depth)
            ]
        )

        self.composite_encodings = FlexiHeliosCompositeEncodings(
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

    # is naming here confusing if one of these channels can be missing?
    def collapse_and_combine_hwtc(self, x: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Collapse the tokens and masks, respectively, into two tensors."""
        tokens, masks = [], []
        available_modalities = return_modalities_from_dict(x)
        modalities_to_process = get_modalities_to_process(
            available_modalities, self.supported_modality_names
        )
        for modality in modalities_to_process:
            masked_modality_name = MaskedHeliosSample.get_masked_modality_name(modality)
            x_modality = x[modality]
            x_modality_mask = x[masked_modality_name]
            tokens.append(rearrange(x_modality, "b ... d -> b (...) d"))
            masks.append(rearrange(x_modality_mask, "b ... -> b (...)"))
        tokens = torch.cat(tokens, dim=1)
        masks = torch.cat(masks, dim=1)

        return tokens, masks

    def stack_spatial_modalities_and_masks(
        self, tokens_dict: dict[str, Tensor]
    ) -> Tensor:
        """Stack the spatial modalities together."""
        available_modalities = return_modalities_from_dict(tokens_dict)
        modalities_to_process = get_modalities_to_process(
            available_modalities, self.supported_modality_names
        )
        mask_list = []
        data_list = []
        for modality in modalities_to_process:
            if Modality.get(modality).is_spatial:
                masked_modality_name = MaskedHeliosSample.get_masked_modality_name(
                    modality
                )
                logger.info(
                    f"modality: {modality}, masked_modality_name: {masked_modality_name}"
                )
                data = tokens_dict[modality]
                mask = tokens_dict[masked_modality_name]
                data_list.append(data)
                mask_list.append(mask)
        # stack in the modality dimension
        return torch.cat(data_list, dim=4), torch.cat(mask_list, dim=4)

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
            masked_modality_name = MaskedHeliosSample.get_masked_modality_name(modality)
            original_masks_dict[masked_modality_name] = x[masked_modality_name]
        return tokens_only_dict, original_masks_dict, modalities_to_dims_dict

    @staticmethod
    def split_and_expand_per_modality(
        x: Tensor, modalities_to_dims_dict: dict
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
    def pack_tokens(tokens: Tensor, mask: Tensor) -> Tensor:
        """Pack the Batch and sequence length dimensions of tokens and mask into a single tensor.

        Args:
            tokens: Tokens to pack
            mask: Mask to pack

        Returns:
            Packed tokens enabling varlen flash attention
        """
        tokens_packed = torch.flatten(tokens, end_dim=1)
        mask = torch.flatten(mask)
        tokens = tokens_packed[mask]
        return tokens

    @staticmethod
    def unpack_tokens(tokens: Tensor, mask: Tensor, og_shape: tuple) -> Tensor:
        """Unpack the Batch and sequence length dimensions of tokens and mask into a single tensor.

        Args:
            tokens: Tokens to unpack
            mask: Mask to unpack
            og_shape: Original shape of the tokens
        """
        tokens_new = tokens.new_zeros(og_shape[0] * og_shape[1], og_shape[2])
        mask = torch.flatten(mask)
        tokens_new[mask] = tokens
        tokens = tokens_new.reshape(og_shape[0], og_shape[1], -1)
        return tokens

    def apply_fsdp(self, **fsdp_kwargs: Any) -> None:
        """Apply FSDP to the model."""
        for block in self.blocks:
            block.apply_fsdp(**fsdp_kwargs)

    def apply_compile(self) -> None:
        """Apply torch.compile to the model."""
        for block in self.blocks:
            block.apply_compile()


class Encoder(FlexiHeliosBase):
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
        learnable_channel_embeddings: bool = True,
        random_channel_embeddings: bool = False,
        num_projection_layers: int = 1,
        aggregate_then_project: bool = True,
        use_flash_attn: bool = False,
        frozen_patch_embeddings: bool = False,
        qk_norm: bool = False,
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
            learnable_channel_embeddings: Whether to use learnable channel embeddings
            random_channel_embeddings: Initialize channel embeddings randomly (zeros if False)
            num_projection_layers: The number of layers to use in the projection. If >1, then
                a ReLU activation will be applied between layers
            aggregate_then_project: If True, then we will average the tokens before applying
                the projection. If False, we will apply the projection first.
            use_flash_attn: Whether to use flash attention
            frozen_patch_embeddings: If True, we freeze the embedding layer, as recommended in
                https://arxiv.org/pdf/2104.02057, Section 4.2
            qk_norm: Whether to apply normalization to Q and K in attention
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
            use_flash_attn=use_flash_attn,
            random_channel_embeddings=random_channel_embeddings,
            qk_norm=qk_norm,
        )
        self.min_patch_size = min_patch_size
        self.max_patch_size = max_patch_size
        self.embedding_size = embedding_size
        self.patch_embeddings = FlexiHeliosPatchEmbeddings(
            self.supported_modality_names,
            self.max_patch_size,
            self.embedding_size,
        )
        self.project_and_aggregate = ProjectAndAggregate(
            embedding_size=self.embedding_size,
            num_layers=num_projection_layers,
            aggregate_then_project=aggregate_then_project,
        )
        self.norm = nn.LayerNorm(self.embedding_size)
        self.apply(self._init_weights)

        if frozen_patch_embeddings:
            for p in self.patch_embeddings.parameters():
                p.requires_grad = False

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
            exit_ids_seq, _ = self.collapse_and_combine_hwtc(exit_ids_per_modality)
        else:
            exit_ids_seq = None
        return exit_ids_seq

    def _maybe_get_attn_mask(
        self,
        new_mask: Tensor,
        fast_pass: bool,
    ) -> Tensor | None:
        """Get the attention mask or None if we should pass None to the transformer."""
        if fast_pass or not self.training:
            return None
        else:
            return new_mask

    def _maybe_remove_masked_tokens(
        self,
        tokens: Tensor,
        mask: Tensor,
        fast_pass: bool,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Remove masked tokens from the tokens and masks."""
        if fast_pass:
            # This is the inference fast pass
            indices = None
            new_mask = None
            seq_lengths = None
            max_seqlen = None
            bool_mask = None
        else:
            bool_mask = mask == MaskValue.ONLINE_ENCODER.value
            tokens, indices, new_mask, seq_lengths, max_seqlen = (
                self.remove_masked_tokens(tokens, bool_mask)
            )
        return tokens, indices, new_mask, seq_lengths, max_seqlen, bool_mask

    def _maybe_add_removed_tokens(
        self,
        tokens: Tensor,
        indices: Tensor,
        mask: Tensor,
        fast_pass: bool,
    ) -> Tensor:
        """Add removed tokens to the tokens and masks."""
        if not fast_pass:
            tokens, _ = self.add_removed_tokens(tokens, indices, mask)
        return tokens

    def apply_attn(
        self,
        x: dict[str, Tensor],
        timestamps: Tensor,
        patch_size: int,
        input_res: int,
        token_exit_cfg: dict[str, int] | None = None,
        fast_pass: bool = False,
    ) -> dict[str, Tensor]:
        """Apply the attention to the tokens and masks."""
        tokens_only_dict, original_masks_dict, modalities_to_dims_dict = (
            self.split_tokens_masks_and_dims(x)
        )
        # already a no-op but we could remove entirely
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
        tokens, mask = self.collapse_and_combine_hwtc(tokens_dict)

        tokens, indices, new_mask, seq_lengths, max_seqlen, bool_mask = (
            self._maybe_remove_masked_tokens(tokens, mask, fast_pass)
        )

        if exit_ids_seq is not None:
            exit_ids_seq, _, _, _, _ = self.remove_masked_tokens(
                exit_ids_seq, bool_mask
            )
            # still linear projections
            exited_tokens, _, _, _, _ = self.remove_masked_tokens(
                exited_tokens, bool_mask
            )

        # Pack x tokens
        if self.use_flash_attn and not fast_pass:
            cu_seqlens = get_cumulative_sequence_lengths(seq_lengths)
            og_shape = tokens.shape
            tokens = self.pack_tokens(tokens, new_mask)
        else:
            cu_seqlens = None

        attn_mask = self._maybe_get_attn_mask(
            new_mask,
            fast_pass=fast_pass,
        )
        # Apply attn with varying encoder depths
        for i_blk, blk in enumerate(self.blocks):
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

        # TODO: Integrate fast pass with non torch flash attention
        if self.use_flash_attn and not fast_pass:
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
        # we apply the norm before we add the removed tokens,
        # so that the norm is only computed against "real" tokens
        tokens = self.norm(tokens)
        # we don't care about the mask returned by add_removed_tokens, since we will
        # just use the original, unclipped mask here
        tokens = self._maybe_add_removed_tokens(tokens, indices, new_mask, fast_pass)

        tokens_per_modality_dict = self.split_and_expand_per_modality(
            tokens, modalities_to_dims_dict
        )
        # merge original masks and the processed tokens
        tokens_per_modality_dict.update(original_masks_dict)
        return tokens_per_modality_dict

    def forward(
        self,
        x: MaskedHeliosSample,
        patch_size: int,
        input_res: int = BASE_GSD,
        token_exit_cfg: dict | None = None,
        fast_pass: bool = False,
    ) -> dict[str, Any]:
        """Process masked input samples into token representations.

        Args:
            x: Masked input sample containing the data to be encoded
            patch_size: Size of patches to divide the input into
            input_res: Resolution of the input data
            token_exit_cfg: Configuration for token exit
            fast_pass: Whether to always pass None as the mask to the transformer, this enables torch based flash attention, and skips mask construciton and sorting

        Returns:
            TokensAndMasks containing the encoded representations and their masks
        """
        if fast_pass and token_exit_cfg is not None:
            raise ValueError("token_exit_cfg cannot be set when fast_pass is True")

        patchified_tokens_and_masks = self.patch_embeddings.forward(
            x, patch_size, fast_pass=fast_pass
        )
        if token_exit_cfg is None or any(
            [exit_depth > 0 for exit_depth in token_exit_cfg.values()]
        ):
            patchified_tokens_and_masks = self.apply_attn(
                x=patchified_tokens_and_masks,
                timestamps=x.timestamps,
                patch_size=patch_size,
                input_res=input_res,
                token_exit_cfg=token_exit_cfg,
                fast_pass=fast_pass,
            )
        output = TokensAndMasks(**patchified_tokens_and_masks)
        output_dict = {
            "tokens_and_masks": output,
        }

        if not fast_pass:
            output_dict["project_aggregated"] = self.project_and_aggregate(output)
        return output_dict

    def apply_fsdp(self, **fsdp_kwargs: Any) -> None:
        """Apply FSDP to the model."""
        super().apply_fsdp(**fsdp_kwargs)
        # Don't Shard the small layers
        # fully_shard(self.patch_embeddings, **fsdp_kwargs)
        # register_fsdp_forward_method(self.patch_embeddings, "forward")
        # fully_shard(self.project_and_aggregate, **fsdp_kwargs)
        # register_fsdp_forward_method(self.project_and_aggregate, "forward")
        fully_shard(self, **fsdp_kwargs)

    def apply_compile(self) -> None:
        """Apply torch.compile to the model."""
        # self.compile(mode="max-autotune", dynamic=False, fullgraph=True)
        logger.info("Compiling blocks")
        # torch.compile(self.blocks, dynamic=False, mode="max-autotune", fullgraph=True)
        # individual block compile is still a lot slower
        for block in self.blocks:
            block.apply_compile()
        # torch.compile(self.patch_embeddings, dynamic=False, mode="max-autotune-no-cudagraphs", fullgraph=True)


class PredictorBase(FlexiHeliosBase):
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
        use_flash_attn: bool = False,
        qk_norm: bool = False,
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
            use_flash_attn: Whether to use flash attention
            qk_norm: Whether to apply normalization to Q and K in attention
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
            use_flash_attn=use_flash_attn,
            qk_norm=qk_norm,
        )
        self.learnable_channel_embeddings = learnable_channel_embeddings
        self.random_channel_embeddings = random_channel_embeddings
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
            mask_name = MaskedHeliosSample.get_masked_modality_name(modality)
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

    # TODO: GIVE more explicit function names
    @staticmethod
    def split_x_y(tokens: Tensor, mask: Tensor) -> tuple[Tensor, ...]:
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
            seqlens_tokens_to_decode: Sequence lengths of tokens to decode of shape [B].
            seqlens_unmasked_tokens: Sequence lengths of unmasked tokens of shape [B].
            max_length_of_decoded_tokens: Maximum length of decoded tokens of shape [1].
            max_length_of_unmasked_tokens: Maximum length of unmasked tokens of shape [1].
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

        seqlens_unmasked_tokens = binarized_online_encoder_mask.sum(dim=-1)
        max_length_of_unmasked_tokens = seqlens_unmasked_tokens.max()
        seqlens_tokens_to_decode = binarized_decoder_mask.sum(dim=-1)
        max_length_of_decoded_tokens = seqlens_tokens_to_decode.max()

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
            seqlens_tokens_to_decode,
            seqlens_unmasked_tokens,
            max_length_of_decoded_tokens,
            max_length_of_unmasked_tokens,
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

    def is_any_data_to_be_decoded(self, modality_mask: Tensor) -> bool:
        """Check if any data is to be decoded for a given modality."""
        return (MaskValue.DECODER.value == modality_mask).any()

    def apply_fsdp(self, **fsdp_kwargs: Any) -> None:
        """Apply FSDP to the model."""
        super().apply_fsdp(**fsdp_kwargs)
        fully_shard(self, **fsdp_kwargs)


class Predictor(PredictorBase):
    """Predictor module that generates predictions from encoded tokens."""

    cross_attn = True

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
                y=unmasked_tokens,
                attn_mask=(
                    unmasked_tokens_mask.bool() if not self.use_flash_attn else None
                ),  # only for flash attn though this should not be left in
                cu_seqlens_q=cu_seqlens_tokens_to_decode,
                cu_seqlens_k=cu_seqlens_unmasked_tokens,
                max_seqlen_q=max_length_of_tokens_to_decode,
                max_seqlen_k=max_length_of_unmasked_tokens,
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
            # Although, we do not account for missing tokens both proj and normalize are on token dimension so there is no mixing with real tokens
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
class EncoderConfig(Config):
    """Configuration for the Encoder."""

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
    learnable_channel_embeddings: bool = True
    random_channel_embeddings: bool = False
    num_projection_layers: int = 1
    aggregate_then_project: bool = True
    use_flash_attn: bool = False
    frozen_patch_embeddings: bool = False
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

    def build(self) -> "Encoder":
        """Build the encoder."""
        self.validate()
        kwargs = self.as_dict(exclude_none=True, recurse=False)
        # supported_modality_names is replaced by supported_modalities
        kwargs.pop("supported_modality_names")
        kwargs["supported_modalities"] = self.supported_modalities
        logger.info(f"Encoder kwargs: {kwargs}")
        return Encoder(**kwargs)


@dataclass
class PredictorConfig(Config):
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

    def build(self) -> "PredictorBase":
        """Build the predictor."""
        self.validate()
        kwargs = self.as_dict(exclude_none=True, recurse=False)
        # supported_modality_names is replaced by supported_modalities
        kwargs.pop("supported_modality_names")
        kwargs["supported_modalities"] = self.supported_modalities
        logger.info(f"Predictor kwargs: {kwargs}")
        return Predictor(**kwargs)
