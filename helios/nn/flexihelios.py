"""Model code for the Helios model."""

import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, NamedTuple

import torch
from einops import rearrange, repeat
from olmo_core.config import Config
from torch import Tensor, nn
from torch.distributed.fsdp import fully_shard

from helios.data.constants import Modality, ModalitySpec
from helios.dataset.utils import get_modality_specs_from_names
from helios.nn.attention import Block
from helios.nn.encodings import (
    get_1d_sincos_pos_encoding,
    get_2d_sincos_pos_encoding_with_resolution,
    get_month_encoding_table,
)
from helios.nn.flexi_patch_embed import FlexiPatchEmbed, FlexiPatchReconstruction
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


# Resolution of the input data in meters
BASE_GSD = 10


class PoolingType(str, Enum):
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

    def flatten_tokens_and_masks(self) -> tuple[Tensor, Tensor]:
        """Return the flattened tokens and masks.

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

        x = torch.cat(flattened_x, dim=1)
        masks = torch.cat(flattened_masks, dim=1)[:, :, 0]
        return x, masks

    def pool_unmasked_tokens(
        self, pooling_type: PoolingType = PoolingType.MAX, spatial_pooling: bool = False
    ) -> Tensor:
        """Pool the unmasked tokens.

        Args:
            pooling_type: Pooling type for the tokens
            spatial_pooling: Whether to keep the spatial dimensions when pooling. If true,
                this expects the masks within a spatial modality to be consistent (e.g. all
                s2 tokens would have the same mask.)
        """
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
                return x_for_pooling.sum(dim=1) / torch.sum(mask, -1, keepdim=True)
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
        if modality_spec.get_tile_resolution() == 0:
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
                        patch_size=self.max_patch_size,
                    )
                    for idx, channel_set_idxs in enumerate(
                        modality_spec.bandsets_as_indices()
                    )
                }
            )

    def apply_embedding_to_modality(
        self, modality: str, input_data: MaskedHeliosSample, patch_size: int
    ) -> tuple[Tensor, Tensor]:
        """Apply embedding to a modality."""
        logger.debug(f"applying embedding to modality:{modality}")
        masked_modality_name = input_data.get_masked_modality_name(modality)
        modality_mask = getattr(input_data, masked_modality_name)
        modality_data = getattr(input_data, modality)

        modality_spec = Modality.get(modality)

        modality_tokens, modality_masks = [], []
        for idx, channel_set_indices in enumerate(modality_spec.bandsets_as_indices()):
            modality_specific_kwargs = {}
            if not modality_spec.is_spatial:
                # static in time
                token_mask = modality_mask[..., idx]
            else:
                token_mask = modality_mask[:, 0::patch_size, 0::patch_size, ..., idx]
                modality_specific_kwargs = {"patch_size": patch_size}
            # Now apply the embedding to the patchified data
            patchified_data = modality_data[..., channel_set_indices]
            embedding_module = self.per_modality_embeddings[modality][
                self._get_embedding_module_name(modality, idx)
            ]
            patchified_data = embedding_module(
                patchified_data, **modality_specific_kwargs
            )
            modality_tokens.append(patchified_data)
            modality_masks.append(token_mask)
        return torch.stack(modality_tokens, dim=-2), torch.stack(modality_masks, dim=-1)

    @staticmethod
    def is_any_data_seen_by_encoder(modality_mask: Tensor) -> bool:
        """Check if any data is seen by the encoder."""
        return (MaskValue.ONLINE_ENCODER.value == modality_mask).any()

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
        modalities_to_process = get_modalities_to_process(
            input_data.modalities, self.supported_modality_names
        )
        for modality in modalities_to_process:
            modality_tokens, modality_masks = self.apply_embedding_to_modality(
                modality, input_data, patch_size
            )
            output_dict[modality] = modality_tokens
            modality_mask_name = input_data.get_masked_modality_name(modality)
            output_dict[modality_mask_name] = modality_masks
        return output_dict


class Reconstructor(nn.Module):
    """Module that patchifies and encodes the input data."""

    def __init__(
        self,
        decoder: "Predictor",
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

    decoder_config: "PredictorConfig"
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
        use_channel_embs: bool = True,
        random_channel_embs: bool = False,
        use_learnable_ape: bool = False,
        no_ape: bool = True,  # be careful here
        max_height_patch: int = 12,  # need to be here but it is probably not optimal
    ):
        """Initialize the composite encodings.

        Args:
            embedding_size: Size of token embeddings
            supported_modalities: Which modalities from Modality this model
                instantiation supports
            max_sequence_length: Maximum sequence length
            use_channel_embs: Whether to use learnable channel embeddings
            random_channel_embs: Initialize channel embeddings randomly (zeros if False)
            use_learnable_ape: Whether to use learnable spatial embeddings
            max_height_patch: Maximum height of the patch
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
        self.use_learnable_ape = use_learnable_ape
        self.max_height_patch = max_height_patch
        self.no_ape = no_ape
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

        if use_learnable_ape and not self.no_ape:
            self.learnable_spatial_embed = nn.Parameter(
                torch.zeros(
                    self.embedding_dim_per_embedding_type,
                    max_height_patch,
                    max_height_patch,
                ),
                requires_grad=True,
            )

        # M
        month_tab = get_month_encoding_table(self.embedding_dim_per_embedding_type)
        self.month_embed = nn.Embedding.from_pretrained(month_tab, freeze=True)
        if use_channel_embs:
            args = {"requires_grad": True}
        else:
            args = {"requires_grad": False}

        self.per_modality_channel_embeddings = nn.ParameterDict()
        for modality in self.supported_modalities:
            shape = (len(modality.band_sets), self.embedding_dim_per_embedding_type)
            if random_channel_embs:
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

        if self.use_learnable_ape and not self.no_ape:
            nn.init.trunc_normal_(self.learnable_spatial_embed, std=0.02)

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
    ) -> Tensor:
        """Apply the encodings to the patchified data based on modality type.

        Args:
            modality_name: Name of the modality being processed
            modality_tokens: Token embeddings for the modality
            timestamps: Optional timestamps for temporal encodings
            patch_size: Optional patch size for spatial encodings
            input_res: Optional input resolution for spatial encodings

        Returns:
            Tensor with encodings applied based on modality type
        """
        # TODO: Improve this implementation it is quite bad

        modality = Modality.get(modality_name)
        logger.debug(f"Applying encodings to modality {modality}")

        if modality_tokens.ndim == 3:
            # modality_tokens = [B, Band_Sets, D]; static in space, static in time
            b, b_s, _ = modality_tokens.shape
            ein_string, ein_dict = "b b_s d", {"b": b, "b_s": b_s}
        elif modality_tokens.ndim == 4:
            b, t, b_s, _ = modality_tokens.shape
            ein_string, ein_dict = "b t b_s d", {"b": b, "t": t, "b_s": b_s}
        elif modality_tokens.ndim == 5:
            b, h, w, b_s, _ = modality_tokens.shape
            ein_string, ein_dict = "b h w b_s d", {"b": b, "h": h, "w": w, "b_s": b_s}
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
        channel_embed = self.per_modality_channel_embeddings[modality.name]
        channel_embed = repeat(channel_embed, f"b_s d -> {ein_string}", **ein_dict).to(
            device
        )
        modality_embed[..., :n] += channel_embed

        if modality.is_multitemporal:
            # Time position encodings
            time_embed = repeat(self.pos_embed[:t], f"t d -> {ein_string}", **ein_dict)
            modality_embed[..., n : n * 2] += time_embed.to(device)

            # Month encodings
            assert timestamps is not None
            months = timestamps[:, :, 1]
            month_embed = self.month_embed(months)
            month_embed = repeat(month_embed, f"b t d -> {ein_string}", **ein_dict)
            modality_embed[..., n * 2 : n * 3] += month_embed.to(device)

        if self.no_ape:
            logger.info("No APE applied")
            return modality_tokens + modality_embed

        if modality.is_spatial:
            # Spatial encodings
            assert input_res is not None
            assert patch_size is not None
            gsd_ratio = self.calculate_gsd_ratio(input_res, patch_size)
            if not self.use_learnable_ape:
                logger.info("Using 2d sincos pos encoding")
                spatial_embed = get_2d_sincos_pos_encoding_with_resolution(
                    grid_size=h,
                    res=torch.ones(b, device=device) * gsd_ratio,
                    encoding_dim=self.embedding_dim_per_embedding_type,
                    device=device,
                )
                spatial_embed = rearrange(
                    spatial_embed, "b (h w) d -> b h w d", h=h, w=w
                )
                spatial_embed = repeat(
                    spatial_embed, f"b h w d -> {ein_string}", **ein_dict
                )
            else:
                logger.info("Using learnable APE")
                interp_pos = (
                    torch.nn.functional.interpolate(
                        self.learnable_spatial_embed.unsqueeze(0),
                        size=(h, w),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0)
                ) * gsd_ratio
                interp_pos = rearrange(interp_pos, "d h w -> h w d")
                spatial_embed = repeat(interp_pos, f"h w d -> {ein_string}", **ein_dict)

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
        use_channel_embs: bool,
        num_heads: int,
        mlp_ratio: float,
        depth: int,
        drop_path: float,
        supported_modalities: list[ModalitySpec],
        random_channel_embs: bool = False,
        no_ape: bool = True,
        non_spatial_coord_value: int = 0,  # upper left corner of the image?
    ) -> None:
        """Initialize the FlexiHeliosBase class."""
        super().__init__()

        self.embedding_size = embedding_size
        self.supported_modalities = supported_modalities
        self.supported_modality_names = [x.name for x in supported_modalities]
        logger.info(f"modalities being used by model: {self.supported_modality_names}")

        self.max_sequence_length = max_sequence_length
        self.use_channel_embs = use_channel_embs
        self.random_channel_embs = random_channel_embs
        self.non_spatial_coord_value = non_spatial_coord_value

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
            self.supported_modalities,
            max_sequence_length,
            use_channel_embs,
            random_channel_embs,
            use_learnable_ape=False,  # True,  # TODO: make this configurable
            no_ape=no_ape,
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
    def collapse_and_combine_hwtc(
        self, x: dict[str, Tensor]
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Collapse the tokens and masks, respectively, into two tensors.
        Also returns x and y coordinates for spatial tokens, and a boolean spatial mask.
        Coordinates are `self.non_spatial_coord_value` for non-spatial tokens.
        """
        tokens, masks = [], []
        x_coords_list, y_coords_list = [], []
        is_spatial_list = []

        available_modalities = return_modalities_from_dict(x)
        modalities_to_process = get_modalities_to_process(
            available_modalities, self.supported_modality_names
        )
        for modality in modalities_to_process:
            masked_modality_name = MaskedHeliosSample.get_masked_modality_name(modality)
            x_modality = x[modality]
            x_modality_mask = x[masked_modality_name]

            tokens.append(rearrange(x_modality, "b ... d -> b (...) d"))
            rearranged_current_mask = rearrange(x_modality_mask, "b ... -> b (...)")
            masks.append(rearranged_current_mask)

            modality_spec = Modality.get(modality)
            device = x_modality.device
            coord_dtype = torch.long

            if modality_spec.is_spatial:
                if x_modality_mask.ndim < 3:
                    raise ValueError(
                        f"Spatial modality {modality} has mask with ndim < 3: {x_modality_mask.shape}"
                    )

                mask_shape = x_modality_mask.shape
                _B, H, W = mask_shape[0], mask_shape[1], mask_shape[2]

                h_indices = torch.arange(H, device=device, dtype=coord_dtype)
                reshape_dims_h = [1] * x_modality_mask.ndim
                reshape_dims_h[1] = H
                h_coords_broadcastable = h_indices.reshape(*reshape_dims_h)
                h_coords_tensor = h_coords_broadcastable.expand_as(x_modality_mask)

                w_indices = torch.arange(W, device=device, dtype=coord_dtype)
                reshape_dims_w = [1] * x_modality_mask.ndim
                reshape_dims_w[2] = W
                w_coords_broadcastable = w_indices.reshape(*reshape_dims_w)
                w_coords_tensor = w_coords_broadcastable.expand_as(x_modality_mask)

                x_coords_list.append(rearrange(h_coords_tensor, "b ... -> b (...)"))
                y_coords_list.append(rearrange(w_coords_tensor, "b ... -> b (...)"))

                is_spatial_modality_tensor = torch.ones_like(
                    x_modality_mask, dtype=torch.bool, device=device
                )
                is_spatial_list.append(
                    rearrange(is_spatial_modality_tensor, "b ... -> b (...)")
                )

            else:
                num_flat_tokens_modality = rearranged_current_mask.shape[1]
                batch_size_modality = rearranged_current_mask.shape[0]

                placeholder_coords_modality = torch.full(
                    (batch_size_modality, num_flat_tokens_modality),
                    self.non_spatial_coord_value,
                    device=device,
                    dtype=coord_dtype,
                )
                x_coords_list.append(placeholder_coords_modality)
                y_coords_list.append(placeholder_coords_modality)

                is_spatial_modality_flat = torch.full(
                    (batch_size_modality, num_flat_tokens_modality),
                    False,
                    device=device,
                    dtype=torch.bool,
                )
                is_spatial_list.append(is_spatial_modality_flat)

        tokens_cat = torch.cat(tokens, dim=1)
        masks_cat = torch.cat(masks, dim=1)
        x_coords_cat = torch.cat(x_coords_list, dim=1)
        y_coords_cat = torch.cat(y_coords_list, dim=1)
        is_spatial_cat = torch.cat(is_spatial_list, dim=1)

        return tokens_cat, masks_cat, x_coords_cat, y_coords_cat, is_spatial_cat

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

    def collapse_and_combine_alibi_masks(
        self, alibi_masks: dict[str, Tensor]
    ) -> Tensor:
        """Collapse and combine the alibi masks."""
        # THis could be added to the other collapse and combine function
        alibi_mask_list = []
        available_modalities = return_modalities_from_dict(alibi_masks)
        modalities_to_process = get_modalities_to_process(
            available_modalities, self.supported_modality_names
        )
        for modality in modalities_to_process:
            alibi_mask_list.append(rearrange(alibi_masks[modality], "b ... -> b (...)"))
        alibi_masks = torch.cat(alibi_mask_list, dim=1)
        return alibi_masks

    def create_alibi_mask(self, padding_mask: Tensor, alibi_mask: Tensor) -> Tensor:
        """Create the alibi mask."""
        # fill the padding mask so that values that are not to take part in attention are set to -inf
        float_padding_mask = torch.zeros_like(
            padding_mask, dtype=torch.float32, device=padding_mask.device
        )
        float_padding_mask = float_padding_mask.masked_fill(
            ~padding_mask, float("-inf")
        )
        # this mask still needs to be scaled by the slope
        logger.info(f"padding mask dtype: {float_padding_mask.dtype}")
        logger.info(f"alibi mask dtype: {alibi_mask.dtype}")
        return float_padding_mask - alibi_mask

    def compute_alibi_masks(
        self, masks: dict[str, Tensor], patch_size: int
    ) -> dict[str, Tensor]:
        """Compute the alibi masks without per head slope scaling."""
        # assume that all spatial modalities have the same patch size and dimensions
        # we cna alter this later if needed
        alibi_masks = {}
        for modality in self.supported_modality_names:
            masked_modality_name = MaskedHeliosSample.get_masked_modality_name(modality)
            if masked_modality_name not in masks:
                continue
            mask = masks[masked_modality_name]

            modality_spec = Modality.get(modality)
            if not modality_spec.is_spatial:
                # Make an all zeros alibi mask in the same shape as the mask
                alibi_masks[modality] = torch.zeros_like(mask)
                continue
            B, H, W, T, B_S = mask.shape
            # TODO: These masks can easily be shared across modalities
            # we want the mask to be of Shape H, W and then to repeat it across the other dimensions
            # Why do we need attnetion heads
            # 1) build patch-grid coordinates of shape (N, 2)
            grid_size = int(H)
            coords = (
                torch.stack(
                    torch.meshgrid(
                        torch.arange(grid_size, device=mask.device),
                        torch.arange(grid_size, device=mask.device),
                        indexing="ij",
                    ),
                    dim=-1,
                ).view(-1, 2)
                * patch_size
            )  # → [N, 2]
            # TODO: We may or may not want to scale based on patch size

            # 2) compute pairwise Euclidean distances [N, N]
            #    (coords[:,None] - coords[None,:])² → [N, N, 2]
            diff = coords[:, None, :] - coords[None, :, :]
            dists = diff.pow(2).sum(-1).sqrt()  # → [N, N]
            # repeat this back over the other dimensions
            dists = repeat(dists, "h w -> b h w t b_s", t=T, b_s=B_S, b=B)
            alibi_masks[modality] = dists
        return alibi_masks

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
        use_channel_embs: bool = True,
        random_channel_embs: bool = False,
        num_projection_layers: int = 1,
        aggregate_then_project: bool = True,
        use_alibi: bool = True,  # DOn't leave this false when we keep it
        no_ape: bool = True,
        use_rope: bool = True,
        non_spatial_coord_value: int = -1,
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
            use_channel_embs: Whether to use learnable channel embeddings
            random_channel_embs: Initialize channel embeddings randomly (zeros if False)
            num_projection_layers: The number of layers to use in the projection. If >1, then
                a ReLU activation will be applied between layers
            aggregate_then_project: If True, then we will average the tokens before applying
                the projection. If False, we will apply the projection first.
        """
        super().__init__(
            embedding_size=embedding_size,
            depth=depth,
            mlp_ratio=mlp_ratio,
            num_heads=num_heads,
            max_sequence_length=max_sequence_length,
            use_channel_embs=use_channel_embs,
            drop_path=drop_path,
            supported_modalities=supported_modalities,
            random_channel_embs=random_channel_embs,
            no_ape=no_ape,
            non_spatial_coord_value=non_spatial_coord_value,
        )
        self.use_alibi = use_alibi
        self.use_rope = use_rope
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
        x: Tensor, mask: Tensor, alibi_mask: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
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
        if x.ndim == 2:
            # This is to use this on the coords this is very messy
            x = x.gather(1, indices)
        elif x.ndim == 3:
            x = x.gather(1, indices[:, :, None].expand_as(x))

        # Now all tokens that should be kept are first in the tensor
        if alibi_mask is not None:
            alibi_mask = alibi_mask.gather(1, indices)

        # set masked values to 0 (not really necessary since we'll ignore them anyway)
        if not x.ndim == 2:
            x = x * sorted_mask.unsqueeze(-1)
        # cut off to the length of the longest sequence
        max_length = sorted_mask.sum(-1).max()
        if alibi_mask is not None:
            alibi_mask = alibi_mask[:, :max_length]
        x = x[:, :max_length]
        # New mask chopped to the longest sequence
        updated_mask = sorted_mask[:, :max_length]
        return x, indices, updated_mask, alibi_mask

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
        assert all(
            not key.endswith("_mask") for key in tokens_only_dict
        ), "tokens_only_dict should not contain mask keys"
        if token_exit_cfg:
            exit_ids_per_modality = self.create_token_exit_ids(
                tokens_only_dict, token_exit_cfg
            )
            exit_ids_per_modality.update(mask_only_dict)
            # Exit ids seqs tells us which layer to exit each token
            exit_ids_seq, _, _, _, _ = self.collapse_and_combine_hwtc(
                exit_ids_per_modality
            )
        else:
            exit_ids_seq = None
        return exit_ids_seq

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
        # exited tokens are just the linear projection
        exited_tokens, _, _, _, _ = self.collapse_and_combine_hwtc(x)

        tokens_dict = self.composite_encodings.forward(
            tokens_only_dict,
            timestamps,
            patch_size,
            input_res,
        )
        if self.use_alibi:
            alibi_masks = self.compute_alibi_masks(original_masks_dict, patch_size)
            alibi_mask = self.collapse_and_combine_alibi_masks(alibi_masks)
        else:
            alibi_mask = None
        # logger.info(f"Alibi masks: {alibi_masks}")

        tokens_dict.update(original_masks_dict)

        # We need to create the alibi mask here
        # Then we need to collapse and combine it so it aligns with the other masks and x
        x, mask, x_coords, y_coords, is_spatial_mask = self.collapse_and_combine_hwtc(
            tokens_dict
        )
        # GSD scaling all same base resolution
        # x_coords = x_coords * patch_size
        # y_coords = y_coords * patch_size

        bool_mask = mask == MaskValue.ONLINE_ENCODER.value
        # We need to remove masked tokens from the alibi mask as well to keep it aligned
        tokens, indices, new_mask, alibi_mask = self.remove_masked_tokens(
            x, bool_mask, alibi_mask
        )
        if self.use_rope:
            # we need to filter the x_coords and y_coords to only include the tokens that are not masked but this may differ across samples
            x_coords, _, _, _ = self.remove_masked_tokens(x_coords, bool_mask)
            y_coords, _, _, _ = self.remove_masked_tokens(y_coords, bool_mask)
            is_spatial_mask, _, _, _ = self.remove_masked_tokens(
                is_spatial_mask, bool_mask
            )
        else:
            x_coords = None
            y_coords = None
            is_spatial_mask = None
        # remove the masked tokens from the alibi mask
        if self.use_alibi:
            alibi_mask = self.create_alibi_mask(new_mask, alibi_mask)
        if exit_ids_seq is not None:
            exit_ids_seq, _, _, _ = self.remove_masked_tokens(exit_ids_seq, bool_mask)
            # still linear projections
            exited_tokens, _, _, _ = self.remove_masked_tokens(exited_tokens, bool_mask)
        # we need to apply the alibi mask to the padding mask here so that we can use it in attention
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
            # Tell the block the mask is an alibi mask so we can do the slope scaling for each head
            tokens = blk(
                x=tokens,
                y=None,
                attn_mask=new_mask if not self.use_alibi else alibi_mask,
                x_x_coords=x_coords,
                x_y_coords=y_coords,
                is_spatial_mask_x=is_spatial_mask,
                y_x_coords=None,
                y_y_coords=None,
                is_spatial_mask_y=None,
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
        tokens_per_modality_dict = self.split_and_expand_per_modality(
            tokens, modalities_to_dims_dict
        )
        # merge original masks and the processed tokens
        tokens_per_modality_dict.update(original_masks_dict)
        return tokens_per_modality_dict

    # TODO: we want to have a single API for the encoder and decoder
    def forward(
        self,
        x: MaskedHeliosSample,
        patch_size: int,
        input_res: int = BASE_GSD,
        token_exit_cfg: dict | None = None,
    ) -> tuple[TokensAndMasks, torch.Tensor]:
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
        # Don't Shard the small layers
        # fully_shard(self.patch_embeddings, **fsdp_kwargs)
        # register_fsdp_forward_method(self.patch_embeddings, "forward")
        # fully_shard(self.project_and_aggregate, **fsdp_kwargs)
        # register_fsdp_forward_method(self.project_and_aggregate, "forward")
        fully_shard(self, **fsdp_kwargs)


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
        drop_path: float = 0.0,
        learnable_channel_embeddings: bool = True,
        random_channel_embeddings: bool = False,
        output_embedding_size: int | None = None,
        use_alibi: bool = True,
        no_ape: bool = True,
        non_spatial_coord_value: int = -1,
        use_rope: bool = True,
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
            use_alibi: Whether to use alibi mask
            no_ape: Whether to not use ape
        """
        super().__init__(
            embedding_size=decoder_embedding_size,
            depth=depth,
            mlp_ratio=mlp_ratio,
            num_heads=num_heads,
            max_sequence_length=max_sequence_length,
            drop_path=drop_path,
            use_channel_embs=learnable_channel_embeddings,
            random_channel_embs=random_channel_embeddings,
            supported_modalities=supported_modalities,
            no_ape=no_ape,
            non_spatial_coord_value=non_spatial_coord_value,
        )
        self.use_rope = use_rope
        # TODO: Rename this weird misname
        self.use_alibi = use_alibi
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
    def split_x_y(
        tokens: Tensor, mask: Tensor, alibi_mask: Tensor | None = None
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
        if tokens.ndim == 3:
            tokens = tokens.gather(1, indices[:, :, None].expand_as(tokens))
        elif tokens.ndim == 2:
            tokens = tokens.gather(1, indices)
        else:
            raise ValueError(
                f"Tokens must be of shape [B, T, D] or [B, T], got {tokens.shape}"
            )
        if alibi_mask is not None:
            alibi_mask = alibi_mask.gather(1, indices)

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

        if alibi_mask is not None:
            alibi_mask = alibi_mask[:, -max_length_of_unmasked_tokens:]

        return (
            tokens_to_decode,
            unmasked_tokens,
            tokens_to_decode_mask,
            unmasked_tokens_mask,
            alibi_mask,
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
        if self.use_alibi:
            alibi_masks = self.compute_alibi_masks(original_masks_dict, patch_size)
        tokens_dict.update(original_masks_dict)
        x, mask, x_coords, y_coords, is_spatial_mask = self.collapse_and_combine_hwtc(
            tokens_dict
        )
        # GSD scaling all same base resolution
        # x_coords = x_coords * patch_size
        # y_coords = y_coords * patch_size
        if self.use_alibi:
            alibi_mask = self.collapse_and_combine_alibi_masks(alibi_masks)
        else:
            alibi_mask = None
        # X contains the tokens to decode, Y contains the tokens to attend to for context
        x, y, x_mask, y_mask, alibi_mask, indices = self.split_x_y(x, mask, alibi_mask)
        if self.use_rope:
            # I need both coords for the tokens to decode and the coords for the y tokens
            decode_x_coords, y_x_coords, _, _, _, _ = self.split_x_y(x_coords, mask)
            decode_y_coords, y_y_coords, _, _, _, _ = self.split_x_y(y_coords, mask)
            is_spatial_x, is_spatial_y, _, _, _, _ = self.split_x_y(
                is_spatial_mask, mask
            )
        else:
            decode_x_coords = None
            decode_y_coords = None
            y_x_coords = None
            y_y_coords = None
            is_spatial_x = None
            is_spatial_y = None
            # I can stack the x coords and the y coords
            # I can stack the x coords and the y coords

        # so the y mask is where want to add the alibi mask so we want to be able to
        if self.use_alibi:
            alibi_mask = self.create_alibi_mask(y_mask.bool(), alibi_mask)
        for blk in self.blocks:
            # note that we are not taking the inverse of the mask, since split_x_y gives us
            # true values for values we want to take part in attention
            if self.use_alibi:
                logger.info("Using alibi mask")
            x = blk(
                x=x,
                y=y,
                attn_mask=y_mask.bool() if not self.use_alibi else alibi_mask,
                x_x_coords=decode_x_coords,
                x_y_coords=decode_y_coords,
                y_x_coords=y_x_coords,
                y_y_coords=y_y_coords,
                is_spatial_mask_x=is_spatial_x,
                is_spatial_mask_y=is_spatial_y,
            )
        x = self.combine_x_y(
            tokens_to_decode=x,
            unmasked_tokens=y,
            tokens_to_decode_mask=x_mask,
            unmasked_tokens_mask=y_mask,
            indices=indices,
        )
        tokens_per_modality_dict = self.split_and_expand_per_modality(
            x, modalities_to_dims_dict
        )
        tokens_per_modality_dict.update(original_masks_dict)
        return tokens_per_modality_dict

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

    def apply_fsdp(self, **fsdp_kwargs: Any) -> None:
        """Apply FSDP to the model."""
        super().apply_fsdp(**fsdp_kwargs)
        fully_shard(self, **fsdp_kwargs)


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
    use_channel_embs: bool = True
    random_channel_embs: bool = False
    num_projection_layers: int = 1
    aggregate_then_project: bool = True
    use_alibi: bool = True
    no_ape: bool = True
    non_spatial_coord_value: int = 0  # upper left corner of the image?
    use_rope: bool = True

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
    learnable_channel_embeddings: bool = True  # why are there two?
    random_channel_embeddings: bool = False
    output_embedding_size: int | None = None
    use_alibi: bool = True
    no_ape: bool = True
    non_spatial_coord_value: int = 0  # upper left corner of the image?
    use_rope: bool = True

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
        return Predictor(**kwargs)
