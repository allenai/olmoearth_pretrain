"""Model code for the OlmoEarth Pretrain model."""

import logging
import math
from dataclasses import dataclass
from typing import Any

import torch
from einops import rearrange, reduce, repeat
from torch import Tensor, nn
from torch.distributed.fsdp import fully_shard

from olmoearth_pretrain.config import Config
from olmoearth_pretrain.data.constants import (
    BASE_GSD,
    Modality,
    ModalitySpec,
    get_modality_specs_from_names,
)
from olmoearth_pretrain.datatypes import (
    MaskedOlmoEarthSample,
    MaskValue,
    TokensAndMasks,
)
from olmoearth_pretrain.nn.attention import Block
from olmoearth_pretrain.nn.encodings import (
    PositionEncoding,
    WindowSpec,
    axial_3d_dim_split,
    get_1d_sincos_pos_encoding,
    get_2d_sincos_pos_encoding_with_resolution,
    get_month_encoding_table,
    resolve_position_encoding,
    timestamps_to_days,
)
from olmoearth_pretrain.nn.flexi_patch_embed import (
    FlexiPatchEmbed,
    FlexiPatchReconstruction,
)
from olmoearth_pretrain.nn.pooling import PoolingType, pool_unmasked_tokens
from olmoearth_pretrain.nn.tokenization import TokenizationConfig
from olmoearth_pretrain.nn.utils import get_cumulative_sequence_lengths

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


# TokensAndMasks is imported from datatypes and re-exported here for backwards compatibility
# See olmoearth_pretrain.datatypes.TokensAndMasks for the implementation


def validate_position_encoding(
    position_encoding: str,
    head_dim: int,
    temporal_rope_dim_frac: float,
) -> None:
    """Validate a position encoding mode for a given attention head size."""
    if position_encoding not in PositionEncoding.values():
        raise ValueError(
            f"position_encoding must be one of {PositionEncoding.values()}, "
            f"got {position_encoding}"
        )
    if PositionEncoding.is_2d_rope(position_encoding) and head_dim % 4 != 0:
        raise ValueError(
            f"2D RoPE / RoPE-Mixed require head_dim divisible by 4, got {head_dim}"
        )
    if position_encoding == PositionEncoding.AXIAL_3D_ROPE:
        # Validates that head_dim splits cleanly into (d_t, d_x, d_y).
        axial_3d_dim_split(head_dim, temporal_rope_dim_frac)
    if position_encoding == PositionEncoding.MIXED_3D_ROPE and head_dim % 4 != 0:
        raise ValueError(
            f"3D RoPE-Mixed requires head_dim divisible by 4, got {head_dim}"
        )


class ProjectAndAggregate(nn.Module):
    """Module that applies a linear projection to tokens and masks."""

    def __init__(
        self,
        embedding_size: int,
        num_layers: int,
        aggregate_then_project: bool = True,
        output_embedding_size: int | None = None,
        only_project: bool = False,
    ):
        """Initialize the linear module.

        embedding_size: The embedding size of the input TokensAndMasks
        num_layers: The number of layers to use in the projection. If >1, then
            a ReLU activation will be applied between layers
        aggregate_then_project: If True, then we will average the tokens before applying
            the projection. If False, we will apply the projection first.
        output_embedding_size: If provided, the final layer will output this size instead
            of embedding_size.
        only_project: If True, only project the tokens without aggregation.
        """
        super().__init__()
        self.only_project = only_project
        out_size = (
            output_embedding_size
            if output_embedding_size is not None
            else embedding_size
        )
        # Build projection layers: all intermediate layers use embedding_size, final uses out_size
        if num_layers == 1:
            projections = [nn.Linear(embedding_size, out_size)]
        else:
            projections = [nn.Linear(embedding_size, embedding_size)]
            for _ in range(1, num_layers - 1):
                projections.append(nn.ReLU())
                projections.append(nn.Linear(embedding_size, embedding_size))
            projections.append(nn.ReLU())
            projections.append(nn.Linear(embedding_size, out_size))
        self.projection = nn.Sequential(*projections)
        self.aggregate_then_project = aggregate_then_project

    def apply_aggregate_then_project(
        self, x: TokensAndMasks | torch.Tensor
    ) -> torch.Tensor:
        """Apply the aggregate operation to the input."""
        if isinstance(x, TokensAndMasks):
            pooled_for_contrastive = pool_unmasked_tokens(
                x, PoolingType.MEAN, spatial_pooling=False
            )
        elif isinstance(x, torch.Tensor):
            pooled_for_contrastive = reduce(x, "b ... d -> b  d", "mean")
        else:
            raise ValueError(f"Invalid input type: {type(x)}")
        return self.projection(pooled_for_contrastive)

    def apply_project_then_aggregate(
        self, x: TokensAndMasks | torch.Tensor
    ) -> torch.Tensor:
        """Apply the project operation to the input then aggregate."""
        if isinstance(x, TokensAndMasks):
            decoder_emedded_dict = x.as_dict(include_nones=True)
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
            projected_pooled = pool_unmasked_tokens(
                x_projected, PoolingType.MEAN, spatial_pooling=False
            )
        elif isinstance(x, torch.Tensor):
            x_projected = self.projection(x)
            projected_pooled = reduce(x_projected, "b ... d -> b  d", "mean")
        else:
            raise ValueError(f"Invalid input type: {type(x)}")
        return projected_pooled

    def apply_project_only(
        self, x: TokensAndMasks | torch.Tensor
    ) -> TokensAndMasks | torch.Tensor:
        """Apply projection without aggregation, preserving token structure."""
        if isinstance(x, TokensAndMasks):
            decoder_emedded_dict = x._asdict()
            for modality in x.modalities:
                x_modality = getattr(x, modality)
                x_modality = self.projection(x_modality)
                masked_modality_name = x.get_masked_modality_name(modality)
                decoder_emedded_dict[modality] = x_modality
                decoder_emedded_dict[masked_modality_name] = getattr(
                    x, masked_modality_name
                )
            return TokensAndMasks(**decoder_emedded_dict)
        elif isinstance(x, torch.Tensor):
            return self.projection(x)
        else:
            raise ValueError(f"Invalid input type: {type(x)}")

    def forward(
        self, x: TokensAndMasks | torch.Tensor
    ) -> torch.Tensor | TokensAndMasks:
        """Apply a (non)linear projection to an input TokensAndMasks.

        This can be applied either before or after pooling the tokens.
        If only_project is True, returns projected tokens without aggregation.
        """
        if self.only_project:
            return self.apply_project_only(x)
        elif self.aggregate_then_project:
            return self.apply_aggregate_then_project(x)
        else:
            return self.apply_project_then_aggregate(x)


class MultiModalPatchEmbeddings(nn.Module):
    """Module that patchifies and encodes the input data for multiple modalities."""

    def __init__(
        self,
        supported_modality_names: list[str],
        max_patch_size: int,
        embedding_size: int,
        tokenization_config: TokenizationConfig | None = None,
        use_linear_patch_embed: bool = True,
        band_dropout_rate: float = 0.0,
        random_band_dropout: bool = False,
        band_dropout_modalities: list[str] | None = None,
        patch_embed_hidden_sizes: list[int] | None = None,
        post_proj_hidden_sizes: list[int] | None = None,
    ):
        """Initialize the patch embeddings.

        Args:
            supported_modality_names: Which modalities from Modality this model
                instantiation supports
            max_patch_size: Maximum size of patches
            embedding_size: Size of embeddings
            tokenization_config: Optional config for custom band groupings
            use_linear_patch_embed: Passed through to FlexiPatchEmbed. Set False to load
                checkpoints trained before this flag existed (which used Conv2d).
            band_dropout_rate: Probability of dropping each band channel during training.
                When > 0, randomly zeroes out bands before the patch embedding Conv2d,
                forcing the model to learn cross-spectral representations. Only active
                during training (self.training=True). Default: 0.0 (no dropout).
            random_band_dropout: If True, sample the dropout rate per forward call from
                Uniform(0, band_dropout_rate). This reduces train-inference mismatch
                and acts as stronger augmentation. Default: False (fixed rate).
            band_dropout_modalities: If provided, only apply band dropout to these
                modalities. If None, apply to all modalities. Default: None.
            patch_embed_hidden_sizes: Optional list of hidden layer widths for a
                per-pixel MLP applied BEFORE patchification in the spatial
                FlexiPatchEmbed. If None or empty, the projection is a single nn.Linear
                over the flattened patch (current behavior). Otherwise, each pixel's
                channel vector is mapped via an MLP with ReLU activations (weights
                shared across all pixels), producing an H x W x h[-1] feature map
                that is then patchified and projected to embedding_size. Only applies
                to the spatial branch (FlexiPatchEmbed); the non-spatial nn.Linear
                branch is unaffected.
            post_proj_hidden_sizes: Optional list of hidden layer widths for an MLP
                applied AFTER the patch projection. Each entry adds a
                ReLU -> Linear(prev, h) layer, applied before the norm. Only applies
                to the spatial branch (FlexiPatchEmbed).
        """
        super().__init__()
        self.max_patch_size = max_patch_size
        self.embedding_size = embedding_size
        self.supported_modality_names = supported_modality_names
        self.tokenization_config = tokenization_config or TokenizationConfig()
        self.use_linear_patch_embed = use_linear_patch_embed
        self.band_dropout_rate = band_dropout_rate
        self.random_band_dropout = random_band_dropout
        self.band_dropout_modalities = band_dropout_modalities
        self.patch_embed_hidden_sizes = patch_embed_hidden_sizes
        self.post_proj_hidden_sizes = post_proj_hidden_sizes
        # TODO: want to be able to remove certain bands and modalities
        self.per_modality_embeddings = nn.ModuleDict({})

        for modality in self.supported_modality_names:
            self.per_modality_embeddings[modality] = (
                self._get_patch_embedding_module_for_modality(modality)
            )

        # For every patch embedding module we want to create a unique buffer
        # for selecting the correct band indices from the data tensor
        for modality in self.supported_modality_names:
            for idx, bandset_indices in enumerate(
                self.tokenization_config.get_bandset_indices(modality)
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
        # Get bandset indices from tokenization config (may be overridden)
        bandset_indices = self.tokenization_config.get_bandset_indices(modality)

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
                    for idx, channel_set_idxs in enumerate(bandset_indices)
                }
            )
        else:
            return nn.ModuleDict(
                {
                    self._get_embedding_module_name(modality, idx): FlexiPatchEmbed(
                        in_chans=len(channel_set_idxs),
                        embedding_size=self.embedding_size,
                        base_patch_size_at_16=self.max_patch_size,
                        modality_spec=modality_spec,
                        use_linear_patch_embed=self.use_linear_patch_embed,
                        patch_embed_hidden_sizes=self.patch_embed_hidden_sizes,
                        post_proj_hidden_sizes=self.post_proj_hidden_sizes,
                    )
                    for idx, channel_set_idxs in enumerate(bandset_indices)
                }
            )

    def apply_embedding_to_modality(
        self,
        modality: str,
        input_data: MaskedOlmoEarthSample,
        patch_size: int,
    ) -> tuple[Tensor, Tensor]:
        """Apply embedding to a modality."""
        logger.debug(f"applying embedding to modality:{modality}")
        masked_modality_name = input_data.get_masked_modality_name(modality)
        modality_mask = getattr(input_data, masked_modality_name)
        modality_data = getattr(input_data, modality)

        modality_spec = Modality.get(modality)
        num_band_sets = self.tokenization_config.get_num_bandsets(modality)

        modality_tokens, modality_masks = [], []
        for idx in range(num_band_sets):
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

            buffer_name = self._get_buffer_name(modality, idx)
            inp_data = torch.index_select(modality_data, -1, getattr(self, buffer_name))

            # Check if we should apply band dropout for this bandset
            apply_dropout = (
                self.band_dropout_modalities is None
                or modality in self.band_dropout_modalities
            )
            if self.training and apply_dropout and self.band_dropout_rate > 0.0:
                num_bands = inp_data.shape[-1]
                # Only apply band dropout if there are more than 1 band
                if num_bands > 1:
                    if self.random_band_dropout:
                        rate = (
                            torch.rand(1, device=inp_data.device).item()
                            * self.band_dropout_rate
                        )
                    else:
                        rate = self.band_dropout_rate
                    inp_data = self._apply_band_dropout(inp_data, rate)

            embedding_module = self.per_modality_embeddings[modality][
                self._get_embedding_module_name(modality, idx)
            ]
            patchified_data = embedding_module(inp_data, **modality_specific_kwargs)

            modality_tokens.append(patchified_data)
            modality_masks.append(token_mask)
        return torch.stack(modality_tokens, dim=-2), torch.stack(modality_masks, dim=-1)

    @staticmethod
    def _apply_band_dropout(patchified_data: Tensor, rate: float) -> Tensor:
        """Randomly zero out band channels to force cross-spectral learning.

        Args:
            patchified_data: Input tensor with bands in the last dimension.
            rate: Probability of dropping each band (per sample).

        Returns:
            Tensor with randomly zeroed bands, at least 1 band kept per sample.
        """
        num_bands = patchified_data.shape[-1]
        batch_size = patchified_data.shape[0]
        keep_mask = (
            torch.rand(batch_size, num_bands, device=patchified_data.device) >= rate
        )
        # If no bands are kept, randomly select one band to keep
        no_bands_kept = ~keep_mask.any(dim=1)
        if no_bands_kept.any():
            rand_idx = torch.randint(
                num_bands, (no_bands_kept.sum(),), device=keep_mask.device
            )
            keep_mask[no_bands_kept, rand_idx] = True
        # Broadcast: [B, 1, 1, ..., num_bands]
        view_shape = [batch_size] + [1] * (patchified_data.dim() - 2) + [num_bands]
        return patchified_data * keep_mask.view(*view_shape).to(patchified_data.dtype)

    @staticmethod
    def is_any_data_seen_by_encoder(modality_mask: Tensor) -> bool:
        """Check if any data is seen by the encoder."""
        return (MaskValue.ONLINE_ENCODER.value == modality_mask).any()

    def apply_compile(self) -> None:
        """Apply torch.compile to the model."""
        self.compile(dynamic=False, mode="max-autotune-no-cudagraphs", fullgraph=True)

    def forward(
        self,
        input_data: MaskedOlmoEarthSample,
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
        decoder: nn.Module,
        supported_modalities: list[ModalitySpec],
        max_patch_size: int,
        tokenization_config: TokenizationConfig | None = None,
    ):
        """Initialize the patch embeddings.

        Args:
            decoder: Predictor nn module to use on before reconstructor on input
            supported_modalities: Which modalities from Modality this model
                instantiation supports
            max_patch_size: Maximum size of patches
            tokenization_config: Optional config for custom band groupings
        """
        super().__init__()
        self.max_patch_size = max_patch_size
        self.embedding_size = decoder.output_embedding_size
        self.supported_modalities = supported_modalities
        self.tokenization_config = tokenization_config or TokenizationConfig()
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
        # Get bandset indices from tokenization config (may be overridden)
        bandset_indices = self.tokenization_config.get_bandset_indices(modality.name)

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
                    for idx, channel_set_idxs in enumerate(bandset_indices)
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
                    for idx, channel_set_idxs in enumerate(bandset_indices)
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
        bandset_indices = self.tokenization_config.get_bandset_indices(modality)

        # x: Input tensor with shape [b, h, w, (t), b_s, d]
        modality_tokens, modality_masks = [], []
        for idx, channel_set_indices in enumerate(bandset_indices):
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
    tokenization_config: TokenizationConfig | None = None

    def __post_init__(self) -> None:
        """Coerce raw dicts to TokenizationConfig for old checkpoint compatibility."""
        if isinstance(self.tokenization_config, dict):
            self.tokenization_config = TokenizationConfig(**self.tokenization_config)

    def validate(self) -> None:
        """Validate the configuration."""
        if len(self.supported_modalities) == 0:
            raise ValueError("At least one modality must be added!")
        else:
            for modality in self.supported_modalities:
                if modality not in Modality.values():
                    raise ValueError(f"Modality {modality} is not supported")
        if self.tokenization_config is not None:
            self.tokenization_config.validate()

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


class CompositeEncodings(nn.Module):
    """Composite encodings for FlexiVit models."""

    def __init__(
        self,
        embedding_size: int,
        supported_modalities: list[ModalitySpec],
        max_sequence_length: int,
        learnable_channel_embeddings: bool = True,
        random_channel_embeddings: bool = False,
        tokenization_config: TokenizationConfig | None = None,
        position_encoding: str = "absolute",
        spatial_pos_encoding: str | None = None,
    ):
        """Initialize the composite encodings.

        Args:
            embedding_size: Size of token embeddings
            supported_modalities: Which modalities from Modality this model
                instantiation supports
            max_sequence_length: Maximum sequence length
            learnable_channel_embeddings: Whether to use learnable channel embeddings
            random_channel_embeddings: Initialize channel embeddings randomly (zeros if False)
            tokenization_config: Optional config for custom band groupings
            position_encoding: Position encoding mode; one of the
                ``PositionEncoding`` values.
            spatial_pos_encoding: Deprecated alias for ``position_encoding``.
        """
        super().__init__()
        position_encoding = resolve_position_encoding(
            position_encoding, spatial_pos_encoding
        )
        if position_encoding not in PositionEncoding.values():
            raise ValueError(
                f"position_encoding must be one of {PositionEncoding.values()}, "
                f"got {position_encoding}"
            )
        self.embedding_size = embedding_size
        self.supported_modalities = supported_modalities
        self.supported_modality_names = [
            modality.name for modality in supported_modalities
        ]
        self.tokenization_config = tokenization_config or TokenizationConfig()
        self.position_encoding = position_encoding
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
                num_bandsets = self.tokenization_config.get_num_bandsets(modality.name)
                shape = (num_bandsets, self.embedding_dim_per_embedding_type)
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
                num_bandsets = self.tokenization_config.get_num_bandsets(modality.name)
                shape = (num_bandsets, self.embedding_dim_per_embedding_type)
                if random_channel_embeddings:
                    channel_embeddings = nn.Parameter(torch.rand(shape), **args)
                else:
                    channel_embeddings = nn.Parameter(torch.zeros(shape), **args)
                self.per_modality_channel_embeddings[modality.name] = channel_embeddings

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if getattr(m, "_skip_custom_init", False):
            return
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
        actual_bandsets = modality_tokens.shape[-2]

        # Channel embeddings
        if use_modality_encodings:
            channel_embed = self.per_modality_channel_embeddings[modality.name]
            if channel_embed.shape[0] != actual_bandsets:
                raise ValueError(
                    f"Channel embeddings for {modality.name} expect "
                    f"{channel_embed.shape[0]} bandsets but tokens have "
                    f"{actual_bandsets}. Ensure tokenization_config is "
                    "consistently passed to the encoder/decoder and masking strategy."
                )
            channel_embed = repeat(
                channel_embed, f"b_s d -> {ein_string}", **ein_dict
            ).to(device)
            modality_embed[..., :n] += channel_embed

        if modality.is_multitemporal and use_temporal_encodings:
            # Slot-index temporal encoding (additive). Skipped when 3D RoPE
            # handles temporal position rotationally inside attention.
            if not PositionEncoding.is_3d_rope(self.position_encoding):
                time_embed = repeat(
                    self.pos_embed[:t], f"t d -> {ein_string}", **ein_dict
                )
                modality_embed[..., n : n * 2] += time_embed.to(device)

            # Month encodings stay additive in all modes (calendar/seasonal
            # signal is orthogonal to slot-index).
            assert timestamps is not None
            months = timestamps[:, :, 1]
            month_embed = self.month_embed(months)
            month_embed = repeat(month_embed, f"b t d -> {ein_string}", **ein_dict)
            modality_embed[..., n * 2 : n * 3] += month_embed.to(device)
        if modality.is_spatial and self.position_encoding == PositionEncoding.ABSOLUTE:
            # Spatial encodings
            assert input_res is not None
            assert patch_size is not None
            gsd_ratio = self.calculate_gsd_ratio(input_res, patch_size)
            spatial_embed = get_2d_sincos_pos_encoding_with_resolution(
                grid_size=(h, w),
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


class FlexiVitBase(nn.Module):
    """FlexiVitBase is a base class for FlexiVit models."""

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
        tokenization_config: TokenizationConfig | None = None,
        position_encoding: str = "absolute",
        rope_base: float = 10000.0,
        rope_coordinate_scale: float = 1.0,
        rope_mixed_base: float = 10.0,
        temporal_rope_dim_frac: float = 0.25,
        rope_temporal_base: float | None = None,
        rope_temporal_coordinate_scale: float = 1.0,
        spatial_pos_encoding: str | None = None,
    ) -> None:
        """Initialize the FlexiVitBase class."""
        super().__init__()
        position_encoding = resolve_position_encoding(
            position_encoding, spatial_pos_encoding
        )
        validate_position_encoding(
            position_encoding=position_encoding,
            head_dim=embedding_size // num_heads,
            temporal_rope_dim_frac=temporal_rope_dim_frac,
        )
        if rope_base <= 0:
            raise ValueError(f"rope_base must be positive, got {rope_base}")
        if rope_coordinate_scale <= 0:
            raise ValueError(
                f"rope_coordinate_scale must be positive, got {rope_coordinate_scale}"
            )
        if rope_mixed_base <= 0:
            raise ValueError(f"rope_mixed_base must be positive, got {rope_mixed_base}")
        if not 0.0 < temporal_rope_dim_frac < 1.0:
            raise ValueError(
                f"temporal_rope_dim_frac must be in (0, 1), got {temporal_rope_dim_frac}"
            )
        if rope_temporal_base is not None and rope_temporal_base <= 0:
            raise ValueError(
                f"rope_temporal_base must be positive, got {rope_temporal_base}"
            )
        if rope_temporal_coordinate_scale <= 0:
            raise ValueError(
                "rope_temporal_coordinate_scale must be positive, got "
                f"{rope_temporal_coordinate_scale}"
            )

        self.embedding_size = embedding_size
        self.supported_modalities = supported_modalities
        self.supported_modality_names = [x.name for x in supported_modalities]
        logger.info(f"modalities being used by model: {self.supported_modality_names}")

        self.max_sequence_length = max_sequence_length
        self._base_tokenization_config = tokenization_config or TokenizationConfig()

        self.use_flash_attn = use_flash_attn
        self.position_encoding = position_encoding
        self.rope_base = rope_base
        self.rope_coordinate_scale = rope_coordinate_scale
        self.rope_mixed_base = rope_mixed_base
        self.temporal_rope_dim_frac = temporal_rope_dim_frac
        self.rope_temporal_base = rope_temporal_base
        self.rope_temporal_coordinate_scale = rope_temporal_coordinate_scale
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
                    position_encoding=self.position_encoding,
                    rope_base=self.rope_base,
                    rope_mixed_base=self.rope_mixed_base,
                    temporal_rope_dim_frac=self.temporal_rope_dim_frac,
                    rope_temporal_base=self.rope_temporal_base,
                )
                for _ in range(depth)
            ]
        )

        self.composite_encodings = CompositeEncodings(
            embedding_size,
            self.supported_modalities,
            max_sequence_length,
            learnable_channel_embeddings,
            random_channel_embeddings,
            tokenization_config=self._base_tokenization_config,
            position_encoding=self.position_encoding,
        )
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if getattr(m, "_skip_custom_init", False):
            logger.debug(f"Skipping custom init for {m}")
            return
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
            masked_modality_name = MaskedOlmoEarthSample.get_masked_modality_name(
                modality
            )
            x_modality = x[modality]
            x_modality_mask = x[masked_modality_name]
            tokens.append(rearrange(x_modality, "b ... d -> b (...) d"))
            masks.append(rearrange(x_modality_mask, "b ... -> b (...)"))
        tokens = torch.cat(tokens, dim=1)
        masks = torch.cat(masks, dim=1)

        return tokens, masks

    def build_rope_positions(
        self,
        tokens_only_dict: dict[str, Tensor],
        original_masks_dict: dict[str, Tensor],
        patch_size: int,
        input_res: int,
        timestamps: Tensor | None = None,
    ) -> Tensor | None:
        """Build per-token coordinates for RoPE.

        Returns ``[B, N, 2]`` ``(row, col)`` for 2D RoPE modes and
        ``[B, N, 3]`` ``(t, row, col)`` for 3D RoPE modes. ``None`` for any
        non-RoPE encoding (the additive paths consume raw indices, not
        per-token position tensors).

        Under 3D RoPE the temporal coordinate is days-since-2000 derived from
        ``timestamps`` (so models see real calendar deltas, not slot indices),
        scaled by ``self.rope_temporal_coordinate_scale``. Static modalities
        keep ``t=0`` (no temporal anchor).
        """
        if not PositionEncoding.is_rope(self.position_encoding):
            return None
        is_3d = PositionEncoding.is_3d_rope(self.position_encoding)

        available_modalities = return_modalities_from_dict(tokens_only_dict)
        modalities_to_process = get_modalities_to_process(
            available_modalities, self.supported_modality_names
        )
        gsd_ratio = (
            CompositeEncodings.calculate_gsd_ratio(input_res, patch_size)
            * self.rope_coordinate_scale
        )

        # For 3D RoPE, convert timestamps -> days-since-anchor once. Shape
        # (B, T_max). Each multitemporal modality indexes into this with its
        # own slot count (we assume the first T entries align across modalities,
        # matching how additive temporal encodings already work).
        days_per_timestep: Tensor | None = None
        if is_3d:
            if timestamps is None:
                raise ValueError(
                    "3D RoPE requires timestamps to build the temporal "
                    "coordinate, but none were provided. The temporal axis is "
                    "calendar days since the anchor year and cannot be derived "
                    "from slot indices; pass timestamps on the input sample."
                )
            days_per_timestep = timestamps_to_days(timestamps).to(torch.float32) * (
                self.rope_temporal_coordinate_scale
            )

        position_dict = {}
        for modality_name in modalities_to_process:
            tokens = tokens_only_dict[modality_name]
            modality = Modality.get(modality_name)
            if is_3d:
                positions = self._build_3d_rope_positions_for_modality(
                    modality_name=modality_name,
                    modality=modality,
                    tokens=tokens,
                    gsd_ratio=gsd_ratio,
                    days_per_timestep=days_per_timestep,
                )
            else:
                positions = self._build_2d_rope_positions_for_modality(
                    modality_name=modality_name,
                    modality=modality,
                    tokens=tokens,
                    gsd_ratio=gsd_ratio,
                )
            position_dict[modality_name] = positions

        position_dict.update(original_masks_dict)
        positions, _ = self.collapse_and_combine_hwtc(position_dict)
        return positions

    def build_spatial_token_mask(
        self,
        tokens_only_dict: dict[str, Tensor],
        original_masks_dict: dict[str, Tensor],
    ) -> Tensor:
        """Per-token spatial flag in the same collapsed order as positions.

        ``True`` where a token belongs to a spatial modality (matching the order of
        :meth:`build_rope_positions`).

        Tokens of non-spatial modalities have no meaningful ``(row, col)`` (they sit
        at the coordinate origin), so windowed attention treats them as *global*:
        this flag marks which tokens are spatial so the rest can be exempted.
        """
        flag_dict: dict[str, Tensor] = {}
        available_modalities = return_modalities_from_dict(tokens_only_dict)
        modalities_to_process = get_modalities_to_process(
            available_modalities, self.supported_modality_names
        )
        for modality_name in modalities_to_process:
            tokens = tokens_only_dict[modality_name]
            is_spatial = float(Modality.get(modality_name).is_spatial)
            flag_dict[modality_name] = tokens.new_full(
                (*tokens.shape[:-1], 1), is_spatial
            )
        flag_dict.update(original_masks_dict)
        flags, _ = self.collapse_and_combine_hwtc(flag_dict)
        return flags.squeeze(-1) > 0.5

    def _patch_grid_hw(self, tokens_only_dict: dict[str, Tensor]) -> tuple[int, int]:
        """Spatial patch grid ``(h, w)`` of the (finest) spatial modality.

        Used by the dynamic register bottleneck to size + place its grid to match the
        patches. All spatial modalities share the GSD-scaled coordinate frame, so the
        max over them gives the finest grid (and the largest coordinate extent).
        """
        available_modalities = return_modalities_from_dict(tokens_only_dict)
        modalities_to_process = get_modalities_to_process(
            available_modalities, self.supported_modality_names
        )
        h_max = w_max = 0
        for modality_name in modalities_to_process:
            if not Modality.get(modality_name).is_spatial:
                continue
            h, w = tokens_only_dict[modality_name].shape[1:3]
            h_max, w_max = max(h_max, h), max(w_max, w)
        if h_max == 0 or w_max == 0:
            raise ValueError(
                "dynamic register bottleneck requires at least one spatial modality"
            )
        return (h_max, w_max)

    @staticmethod
    def _zero_rope_positions(tokens: Tensor, coord_dim: int) -> Tensor:
        """Create zero RoPE coordinates matching token layout."""
        return torch.zeros(
            (*tokens.shape[:-1], coord_dim), dtype=torch.float32, device=tokens.device
        )

    @staticmethod
    def _spatial_grid(
        modality_name: str,
        tokens: Tensor,
        gsd_ratio: float,
    ) -> tuple[int, Tensor, Tensor]:
        """Build row/col patch coordinates for a spatial modality."""
        if tokens.ndim not in (5, 6):
            raise ValueError(
                f"Expected spatial tokens for {modality_name} to have 5 "
                f"or 6 dimensions, got {tokens.shape}"
            )
        batch_size, height, width = tokens.shape[:3]
        grid_row = torch.arange(height, device=tokens.device, dtype=torch.float32)
        grid_col = torch.arange(width, device=tokens.device, dtype=torch.float32)
        return batch_size, grid_row * gsd_ratio, grid_col * gsd_ratio

    def _build_2d_rope_positions_for_modality(
        self,
        modality_name: str,
        modality: ModalitySpec,
        tokens: Tensor,
        gsd_ratio: float,
    ) -> Tensor:
        """Build ``(row, col)`` RoPE coordinates for one modality."""
        if not modality.is_spatial:
            return self._zero_rope_positions(tokens, coord_dim=2)

        batch_size, grid_row, grid_col = self._spatial_grid(
            modality_name, tokens, gsd_ratio
        )
        row_g, col_g = torch.meshgrid(grid_row, grid_col, indexing="ij")
        grid = torch.stack([row_g, col_g], dim=-1)

        if tokens.ndim == 5:
            bandsets = tokens.shape[3]
            return repeat(grid, "h w p -> b h w b_s p", b=batch_size, b_s=bandsets)

        timesteps, bandsets = tokens.shape[3], tokens.shape[4]
        return repeat(
            grid,
            "h w p -> b h w t b_s p",
            b=batch_size,
            t=timesteps,
            b_s=bandsets,
        )

    def _build_3d_rope_positions_for_modality(
        self,
        modality_name: str,
        modality: ModalitySpec,
        tokens: Tensor,
        gsd_ratio: float,
        days_per_timestep: Tensor,
    ) -> Tensor:
        """Build ``(t, row, col)`` RoPE coordinates for one modality."""
        positions = self._zero_rope_positions(tokens, coord_dim=3)
        if tokens.ndim == 3:
            # (b, b_s, d): static modality. All coordinates stay zero.
            return positions
        if tokens.ndim == 4:
            # (b, t, b_s, d): temporal-only modality.
            batch_size, timesteps, bandsets, _ = tokens.shape
            t_values = self._select_t_values(
                days_per_timestep, timesteps, device=tokens.device
            )
            positions[..., 0] = repeat(t_values, "b t -> b t b_s", b_s=bandsets)
            return positions

        batch_size, grid_row, grid_col = self._spatial_grid(
            modality_name, tokens, gsd_ratio
        )
        row_g, col_g = torch.meshgrid(grid_row, grid_col, indexing="ij")

        if tokens.ndim == 5:
            # (b, h, w, b_s, d): spatial-only modality.
            bandsets = tokens.shape[3]
            positions[..., 1] = repeat(
                row_g, "h w -> b h w b_s", b=batch_size, b_s=bandsets
            )
            positions[..., 2] = repeat(
                col_g, "h w -> b h w b_s", b=batch_size, b_s=bandsets
            )
            return positions

        if tokens.ndim == 6:
            # (b, h, w, t, b_s, d): full spatiotemporal modality.
            timesteps, bandsets = tokens.shape[3], tokens.shape[4]
            t_values = self._select_t_values(
                days_per_timestep, timesteps, device=tokens.device
            )
            positions[..., 0] = repeat(
                t_values,
                "b t -> b h w t b_s",
                h=tokens.shape[1],
                w=tokens.shape[2],
                b_s=bandsets,
            )
            positions[..., 1] = repeat(
                row_g, "h w -> b h w t b_s", b=batch_size, t=timesteps, b_s=bandsets
            )
            positions[..., 2] = repeat(
                col_g, "h w -> b h w t b_s", b=batch_size, t=timesteps, b_s=bandsets
            )
            return positions

        raise ValueError(
            f"Unsupported tokens shape for {modality_name}: {tokens.shape}"
        )

    @staticmethod
    def _select_t_values(
        days_per_timestep: Tensor,
        num_timesteps: int,
        device: torch.device,
    ) -> Tensor:
        """Pick the first ``num_timesteps`` days for each sample."""
        if days_per_timestep.shape[1] < num_timesteps:
            raise ValueError(
                f"timestamps has {days_per_timestep.shape[1]} slots but modality "
                f"requires {num_timesteps}"
            )
        return days_per_timestep[:, :num_timesteps].to(device)

    @staticmethod
    def split_x_y_positions(
        positions: Tensor,
        indices: Tensor,
        max_length_of_decoded_tokens: Tensor,
        max_length_of_unmasked_tokens: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Split positions using the same sorted order as predictor tokens."""
        positions = positions.gather(1, indices[:, :, None].expand_as(positions))
        positions_to_decode = positions[:, :max_length_of_decoded_tokens]
        unmasked_positions = positions[:, -max_length_of_unmasked_tokens:]
        return positions_to_decode, unmasked_positions

    def add_register_positions(self, positions: Tensor) -> Tensor:
        """Prepend zero coordinates for register tokens."""
        batch_size = positions.shape[0]
        register_positions = positions.new_zeros(
            batch_size,
            self.num_register_tokens,
            positions.shape[-1],
        )
        return torch.cat([register_positions, positions], dim=1)

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


class SpatialRegisterBottleneck(nn.Module):
    """A Perceiver-style spatial register bottleneck.

    A grid of learned latent tokens cross-attention *reads* the encoded (visible) patch
    tokens, then a small *latent transformer* self-attends over the grid. The grid is
    the model's compressed, spatially-anchored representation: the decoder reads only
    this grid, and frozen evals probe it. Register coordinates are placed in the same
    GSD-scaled frame as the patches, so 2D RoPE relative offsets are meaningful.

    The read/process schedule is set by ``interleave``: legacy (all reads, then all
    self-attention) or interleaved (``[read -> self-attend]`` per layer, so the latents
    re-query the input after each refinement -- the Perceiver/DETR/Flamingo pattern).

    Two parameterizations, selected by ``register_grid``:

    - **Legacy / fixed grid** (``register_grid=(n_h, n_w)``): distinct per-cell learned
      latents on a fixed grid whose count is decoupled from the patch count. Kept for
      backwards-compatible loading of checkpoints trained with this module.
    - **Dynamic / single latent** (``register_grid=None``): a *single* learned latent is
      cloned across a grid that matches the input patch grid at forward time. RS imagery
      is translation-invariant, so every spatial query starts from the same content;
      spatial identity comes entirely from 2D RoPE on the per-cell positions. This
      enforces a translation-invariant prior and removes the grid size as a baked
      hyperparameter (it follows the input). Precedents: Perceiver IO dense-output
      queries (shared vector + per-position encoding), the MAE mask token, and Slot
      Attention's shared slot distribution.
    """

    def __init__(
        self,
        encoder_embedding_size: int,
        register_dim: int,
        register_grid: tuple[int, int] | None,
        num_heads: int,
        mlp_ratio: float,
        read_depth: int,
        latent_transformer_depth: int,
        use_2d_rope: bool,
        rope_base: float = 10000.0,
        qk_norm: bool = False,
        interleave: bool = False,
        read_layers: list[int] | None = None,
        per_depth_read_proj: bool = False,
        learned_read_weighting: bool = False,
        fused_read: str | None = None,
        latent_self_attn: bool = True,
    ) -> None:
        """Initialize the spatial register bottleneck.

        Args:
            encoder_embedding_size: Dimension of the encoded patch tokens (the read's K/V source).
            register_dim: Dimension of the register grid (the bottleneck width, typically < encoder dim).
            register_grid: ``(n_h, n_w)`` for a fixed grid of distinct per-cell latents, or
                ``None`` for the dynamic single-latent mode (grid matches the patch grid at
                forward time; requires ``use_2d_rope``).
            num_heads: Number of attention heads for the read + latent transformer blocks.
            mlp_ratio: MLP ratio for the blocks.
            read_depth: Number of cross-attention read blocks (legacy mode only; ignored
                when ``interleave=True``).
            latent_transformer_depth: Number of self-attention blocks over the register grid.
                In ``interleave`` mode this also sets the number of (read -> self-attend)
                layers (one read paired with each self-attention block).
            use_2d_rope: Whether to apply 2D RoPE (requires per-token positions at call time).
            rope_base: RoPE frequency base.
            qk_norm: Whether to apply QK normalization in attention.
            interleave: If True, interleave cross-attention reads with latent self-attention
                (Perceiver/DETR/Flamingo style: ``[read -> self] x latent_transformer_depth``)
                so the latents re-query the input after each refinement, instead of reading
                once up front. If False (default, backwards compatible), do all ``read_depth``
                reads first, then all ``latent_transformer_depth`` self-attention blocks.
            read_layers: If set, enables *multi-depth* reads: the bottleneck reads from a
                different encoder depth at each ``[read -> self-attend]`` step instead of
                re-reading the final layer. ``read_layers`` is the (1-indexed, ascending)
                list of encoder block depths that supply the K/V at each step, so there is
                one read + one latent block per entry. This recovers modality-unique
                information that the final layer drops (Lee et al., CVPR 2026, "Beyond
                What's Shared"). The encoder passes one K/V tensor per layer at forward
                time. When set it forces the interleaved schedule and overrides
                ``read_depth`` / ``latent_transformer_depth``; when None (default) behaviour
                is unchanged (all reads share the final-layer K/V).
            per_depth_read_proj: If True, give each read block its own ``input_norm``
                LayerNorm *and* ``kv_proj`` down-projection instead of a single shared
                pair. In multi-depth mode each read draws from a different encoder depth,
                which have different per-channel statistics and semantics, so a shared
                affine (γ, β) and a shared projection are a poor fit across all of them.
                In interleaved single-source mode every read re-queries the same final
                layer, so per-block projections instead let successive reads extract
                different views through their own lens. Requires more than one read block
                (ignored otherwise). False (default) keeps the shared norm + projection
                (backwards compatible).
            learned_read_weighting: If True, give each read block a learnable scalar gate
                on its residual contribution to the latent: ``z = z + g_d * (read_d(z) -
                z)``. With multi-depth reads this lets the model weight how much each
                encoder depth contributes (ELMo-style scalar mixing / LayerScale per read),
                e.g. down-weighting the early mid-level reads that dilute the pretext-aligned
                final-layer read. Gates initialise to 1.0, so the module is a strict no-op
                at init (reproduces the ungated behaviour) and existing checkpoints are
                unaffected; the learned gates are exposed as ``read_gates`` for logging.
                False (default) keeps the ungated reads.
            fused_read: If set, the multi-depth K/V sources are combined into ONE fused
                source (RAEv2 "multi-layer sum" style) instead of one read block per
                depth: the read/latent schedule reverts to the single-source rules
                (``interleave`` / ``read_depth`` / ``latent_transformer_depth``), and
                every read consumes the fused source -- so the bottleneck architecture
                matches a final-layer-only model exactly, with only the K/V source
                differing. Two combinations:

                - ``"uniform"``: each depth is standardized by a parameter-free
                  LayerNorm and the depths are averaged, then fed through the standard
                  shared ``input_norm``/``kv_proj``. Training cannot re-weight the
                  combination, so mid-depth features are preserved even where the
                  pretext loss would discard them.
                - ``"learned"``: each depth gets its own LayerNorm + projection to
                  ``register_dim`` and the projected contributions are averaged
                  (replacing the shared ``input_norm``/``kv_proj``). The projections
                  can learn to re-weight or suppress depths.

                Per-depth contribution norms are stashed on ``last_read_source_norms``
                for logging (collapse toward the final layer in the learned arm is the
                signal to watch). Requires ``read_layers``; incompatible with
                ``per_depth_read_proj`` (the fusion replaces per-depth projections).
                None (default) keeps one read block per depth.
            latent_self_attn: If True (default), self-attention "latent" blocks run over
                the register grid -- interleaved after each read, or after all reads in the
                legacy schedule. If False, those blocks are dropped entirely: the registers
                are produced by the cross-attention read(s) alone, with no
                register-to-register mixing. The read blocks (and their count) are
                unchanged, so this cleanly isolates the latent self-attention's
                contribution. Backwards compatible (default True).
        """
        super().__init__()
        self.register_dim = register_dim
        self.use_2d_rope = use_2d_rope
        # Multi-depth reads force the interleaved schedule (one read + one latent block per
        # source layer); the read/latent counts are then set by ``read_layers``. With
        # ``fused_read`` the per-depth sources are combined into one K/V source instead, so
        # the schedule reverts to the single-source rules.
        self.multi_depth = read_layers is not None
        self.read_layers = list(read_layers) if read_layers is not None else None
        if fused_read is not None:
            if fused_read not in ("uniform", "learned"):
                raise ValueError(
                    f"fused_read must be 'uniform' or 'learned', got {fused_read!r}"
                )
            if not self.multi_depth:
                raise ValueError("fused_read requires read_layers (the depths to fuse)")
            if per_depth_read_proj:
                raise ValueError(
                    "fused_read replaces the per-depth read projections; it cannot be "
                    "combined with per_depth_read_proj"
                )
        self.fused_read = fused_read
        # Per-depth contribution norms from the most recent fused read, for logging.
        self.last_read_source_norms: Tensor | None = None
        self.interleave = interleave or (self.multi_depth and fused_read is None)
        self.dynamic_grid = register_grid is None
        if register_grid is None:
            if not use_2d_rope:
                # With a single cloned latent the cells are identical at init and stay
                # symmetric without a per-cell positional signal; RoPE is what breaks it.
                raise ValueError(
                    "SpatialRegisterBottleneck dynamic (single-latent) mode requires "
                    "use_2d_rope=True to differentiate grid cells."
                )
            # Grid count + positions are resolved per-forward from the patch grid; the
            # last-used grid is exposed on ``register_grid`` for eval/supervision reshapes.
            self.register_grid: tuple[int, int] | None = None
            self.num_registers: int | None = None
            # A single learned latent, cloned across every grid cell (see class docstring).
            self.register = nn.Parameter(torch.empty(1, register_dim))
            nn.init.trunc_normal_(self.register, std=0.02)
        else:
            self.register_grid = register_grid
            self.num_registers = register_grid[0] * register_grid[1]
            # Distinct per-cell latent vectors (NOT zero-init): because RoPE is *relative*,
            # zero-init registers would give no locality at init; distinct content + fixed
            # grid coordinates are what give each register its spatial identity.
            self.registers = nn.Parameter(torch.empty(self.num_registers, register_dim))
            nn.init.trunc_normal_(self.registers, std=0.02)
        # The read + latent transformer run on small unpacked [B, N, D] tensors with an
        # attention mask, so they use the SDPA path (use_flash_attn=False) regardless of
        # the encoder's flash setting.
        # Multi-depth: one read + one latent per source layer.
        # Interleave: one read per self-attention block, so the read count matches
        # latent_transformer_depth.
        # Legacy: read_depth reads up front, then latent_transformer_depth self-attentions.
        if self.multi_depth and self.fused_read is None:
            assert self.read_layers is not None
            num_read_blocks = len(self.read_layers)
            num_latent_blocks = len(self.read_layers)
        else:
            num_read_blocks = (
                latent_transformer_depth if self.interleave else read_depth
            )
            num_latent_blocks = latent_transformer_depth
        # Optionally drop the latent self-attention entirely (cross-attention reads only).
        # The read count is unchanged; only the register-to-register self-attention blocks
        # are removed, isolating the latent transformer's contribution.
        self.latent_self_attn = latent_self_attn
        if not latent_self_attn:
            num_latent_blocks = 0
        # Per-depth read front-end: give every read block its own input norm + K/V
        # down-projection instead of a single shared pair. Only meaningful with >1 read
        # block. Multi-depth: each block draws from a different encoder depth (distinct
        # per-channel statistics), so a shared affine + projection is a poor fit.
        # Interleaved single-source: each block re-queries the SAME final-layer tokens
        # through its own projection, so successive reads can extract different views
        # instead of being forced through one shared lens.
        # The ``multi_depth`` clause preserves the original gate exactly (multi-depth runs
        # always got per-depth projections when requested); ``num_read_blocks > 1`` extends
        # it to interleaved single-source reads. So this is a strict superset of the old
        # behaviour -- existing checkpoints build the identical parameter set.
        self.per_depth_read_proj = per_depth_read_proj and (
            self.multi_depth or num_read_blocks > 1
        )
        # Down-project the patch K/V source to the (smaller) register dim. The existing
        # Attention ties q/k/v to a single dim, so the read happens entirely at register_dim.
        if self.fused_read == "learned":
            # One norm + projection per fused source depth; their mean IS the fused K/V
            # source (already at register_dim), so no shared input_norm/kv_proj pair.
            assert self.read_layers is not None
            self.fused_norms = nn.ModuleList(
                [nn.LayerNorm(encoder_embedding_size) for _ in self.read_layers]
            )
            self.fused_projs = nn.ModuleList(
                [
                    nn.Linear(encoder_embedding_size, register_dim)
                    for _ in self.read_layers
                ]
            )
        elif self.per_depth_read_proj:
            # One norm + projection per read block.
            self.input_norms = nn.ModuleList(
                [nn.LayerNorm(encoder_embedding_size) for _ in range(num_read_blocks)]
            )
            self.kv_projs = nn.ModuleList(
                [
                    nn.Linear(encoder_embedding_size, register_dim)
                    for _ in range(num_read_blocks)
                ]
            )
        else:
            if self.fused_read == "uniform":
                # Parameter-free per-depth standardization before the uniform average;
                # no learnable affine, so training cannot re-weight the combination.
                self.fused_norm = nn.LayerNorm(
                    encoder_embedding_size, elementwise_affine=False
                )
            self.input_norm = nn.LayerNorm(encoder_embedding_size)
            self.kv_proj = nn.Linear(encoder_embedding_size, register_dim)
        self.read_blocks = nn.ModuleList(
            [
                Block(
                    register_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    qk_norm=qk_norm,
                    cross_attn=True,
                    use_flash_attn=False,
                    position_encoding=(
                        PositionEncoding.AXIAL_2D_ROPE
                        if use_2d_rope
                        else PositionEncoding.ABSOLUTE
                    ),
                    rope_base=rope_base,
                )
                for _ in range(num_read_blocks)
            ]
        )
        self.latent_blocks = nn.ModuleList(
            [
                Block(
                    register_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    qk_norm=qk_norm,
                    cross_attn=False,
                    use_flash_attn=False,
                    position_encoding=(
                        PositionEncoding.AXIAL_2D_ROPE
                        if use_2d_rope
                        else PositionEncoding.ABSOLUTE
                    ),
                    rope_base=rope_base,
                )
                for _ in range(num_latent_blocks)
            ]
        )
        # Learnable per-read residual gates (one scalar per read block). Init to 1.0 so the
        # gated update ``z + g*(read(z) - z)`` equals the ungated ``read(z)`` at init, making
        # this a no-op until the gates move (and leaving existing checkpoints unchanged).
        self.learned_read_weighting = learned_read_weighting
        if learned_read_weighting:
            self.read_gates = nn.Parameter(torch.ones(num_read_blocks))
        self.norm = nn.LayerNorm(register_dim)

    def build_register_positions(
        self, patch_positions: Tensor, register_grid: tuple[int, int]
    ) -> Tensor:
        """Place the register grid evenly across the patch extent (GSD-scaled frame).

        Args:
            patch_positions: ``[B, N, 2]`` GSD-scaled ``(row, col)`` patch coordinates.
            register_grid: ``(n_h, n_w)`` grid to lay down (matches the patch grid in
                dynamic mode, so the register coords coincide with the patch coords).

        Returns:
            ``[B, n_h * n_w, 2]`` register coordinates spanning ``[0, max_patch_coord]``.
        """
        n_h, n_w = register_grid
        device = patch_positions.device
        # Patch coords are >= 0 (non-spatial tokens sit at 0), so amax gives the extent.
        max_pos = patch_positions.amax(dim=1)  # [B, 2]
        lin_h = torch.linspace(0.0, 1.0, n_h, device=device)
        lin_w = torch.linspace(0.0, 1.0, n_w, device=device)
        grid_h, grid_w = torch.meshgrid(lin_h, lin_w, indexing="ij")
        grid = torch.stack([grid_h, grid_w], dim=-1).reshape(
            -1, 2
        )  # [n_reg, 2] in [0, 1]
        return grid.unsqueeze(0) * max_pos.unsqueeze(1)  # [B, n_reg, 2]

    def _fuse_read_sources(
        self, patch_tokens: list[Tensor], visible_mask: Tensor | None
    ) -> Tensor:
        """Mean-combine the per-depth K/V sources into one fused source.

        ``uniform``: standardize each depth with the parameter-free ``fused_norm`` and
        average, then run the standard shared ``input_norm`` + ``kv_proj``. ``learned``:
        project each depth through its own ``fused_norms[i]`` + ``fused_projs[i]`` and
        average the projected contributions (already at register_dim).

        Stashes the mean per-depth contribution norm (over visible tokens) on
        ``last_read_source_norms`` -- in the learned arm these drifting apart (e.g.
        collapsing onto the final layer) is the signal the run exists to measure; in the
        uniform arm they are ~constant by construction and just confirm uniformity.
        """
        if self.fused_read == "learned":
            contribs = [
                proj(norm(t))
                for norm, proj, t in zip(
                    self.fused_norms, self.fused_projs, patch_tokens
                )
            ]
            kv = torch.stack(contribs).mean(dim=0)
        else:
            contribs = [self.fused_norm(t) for t in patch_tokens]
            kv = self.kv_proj(self.input_norm(torch.stack(contribs).mean(dim=0)))
        with torch.no_grad():
            per_depth = []
            for contrib in contribs:
                token_norms = contrib.detach().float().norm(dim=-1)  # [B, N]
                if visible_mask is not None:
                    mask = visible_mask.bool()
                    per_depth.append(
                        (token_norms * mask).sum() / mask.sum().clamp(min=1)
                    )
                else:
                    per_depth.append(token_norms.mean())
            self.last_read_source_norms = torch.stack(per_depth)
        return kv

    def forward(
        self,
        patch_tokens: Tensor | list[Tensor],
        patch_positions: Tensor | None,
        visible_mask: Tensor | None,
        spatial_grid: tuple[int, int] | None = None,
        window_half_extent: float | None = None,
        patch_is_global: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """Read the (visible) patch tokens into the register grid.

        Args:
            patch_tokens: Encoded tokens ``[B, N, encoder_embedding_size]``. In multi-depth
                mode (``read_layers`` set) this is instead a list of one such tensor per
                read block, giving the K/V source at each successive encoder depth.
            patch_positions: GSD-scaled ``[B, N, 2]`` coords (None if not using RoPE).
            visible_mask: Bool ``[B, N]``, True where a token is a valid key
                (``MaskValue.ONLINE_ENCODER``). None means attend to all tokens.
            spatial_grid: ``(n_h, n_w)`` patch grid; required in dynamic mode (the single
                latent is cloned to this many cells), ignored in fixed-grid mode.
            window_half_extent: If set, restrict the read (register query vs patch key) and
                the latent self-attention (register vs register) to a sliding window of this
                half-width, in the same coordinate units as the positions. None -> the read
                and latent transformer use full attention (backwards compatible).
            patch_is_global: Optional ``[B, N]`` bool, True where a patch key is non-spatial
                and should be readable from every register regardless of the window. Used
                only when ``window_half_extent`` is set.

        Returns:
            registers: ``[B, num_registers, register_dim]``
            register_positions: ``[B, num_registers, 2]`` or None
        """
        # Down-project the K/V source(s) to register_dim. Multi-depth gets one source per
        # read block; otherwise a single source is reused by every read. With
        # per_depth_read_proj each read block has its own norm + projection (so even the
        # single-source case is projected once per block); otherwise they share one pair.
        if self.multi_depth:
            if not isinstance(patch_tokens, list):
                raise ValueError(
                    "multi-depth register bottleneck expects a list of per-layer "
                    "patch_tokens (one per read block)"
                )
            assert self.read_layers is not None
            if len(patch_tokens) != len(self.read_layers):
                raise ValueError(
                    f"expected {len(self.read_layers)} K/V sources (one per read layer), "
                    f"got {len(patch_tokens)}"
                )
            if self.fused_read is not None:
                # All reads consume the single fused source (RAEv2 multi-layer-sum style).
                kv = self._fuse_read_sources(patch_tokens, visible_mask)
                kv_per_read = [kv] * len(self.read_blocks)
            elif self.per_depth_read_proj:
                kv_per_read = [
                    proj(norm(t))
                    for norm, proj, t in zip(
                        self.input_norms, self.kv_projs, patch_tokens
                    )
                ]
            else:
                kv_per_read = [self.kv_proj(self.input_norm(t)) for t in patch_tokens]
            reference_tokens = patch_tokens[0]
        else:
            if isinstance(patch_tokens, list):
                raise ValueError(
                    "single-source register bottleneck expects a tensor, not a list"
                )
            if self.per_depth_read_proj:
                # Each read block re-projects the same final-layer source through its own
                # norm + projection (interleaved single-source reads).
                kv_per_read = [
                    proj(norm(patch_tokens))
                    for norm, proj in zip(self.input_norms, self.kv_projs)
                ]
            else:
                kv = self.kv_proj(self.input_norm(patch_tokens))
                kv_per_read = [kv] * len(self.read_blocks)
            reference_tokens = patch_tokens
        batch_size = reference_tokens.shape[0]
        if self.dynamic_grid:
            if spatial_grid is None:
                raise ValueError(
                    "dynamic register bottleneck requires a spatial_grid (the patch grid)"
                )
            register_grid = spatial_grid
            # Expose the grid actually used so eval/supervision can reshape the registers.
            self.register_grid = register_grid
            num_registers = register_grid[0] * register_grid[1]
            # Clone the single learned latent across the batch and all grid cells; RoPE on
            # the per-cell register_positions is what differentiates them.
            registers = (
                self.register.unsqueeze(0)
                .expand(batch_size, num_registers, -1)
                .contiguous()
            )
        else:
            assert self.register_grid is not None  # set in __init__ for fixed-grid mode
            register_grid = self.register_grid
            registers = (
                self.registers.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
            )
        register_positions = None
        if self.use_2d_rope:
            if patch_positions is None:
                raise ValueError(
                    "patch_positions are required for the RoPE register bottleneck"
                )
            register_positions = self.build_register_positions(
                patch_positions, register_grid
            )
        # Read mask: by default just the [B, N] key-visibility mask. With windowing the
        # read instead restricts each register to a spatial window of the patch tokens
        # (non-spatial patches stay globally readable), AND-ed with visibility; the latent
        # transformer windows register-over-register. Both windowed masks are built lazily
        # per query-chunk inside attention (via WindowSpec) to bound memory.
        read_attn_mask: Tensor | None = (
            visible_mask.bool() if visible_mask is not None else None
        )
        read_window_spec: WindowSpec | None = None
        latent_window_spec: WindowSpec | None = None
        if window_half_extent is not None and self.use_2d_rope:
            assert register_positions is not None and patch_positions is not None
            read_attn_mask = None  # validity carried by read_window_spec.key_valid
            read_window_spec = WindowSpec(
                q_positions=register_positions,
                k_positions=patch_positions,
                half_extent=window_half_extent,
                k_is_global=patch_is_global,
                key_valid=visible_mask.bool() if visible_mask is not None else None,
            )
            latent_window_spec = WindowSpec(
                q_positions=register_positions,
                k_positions=register_positions,
                half_extent=window_half_extent,
            )

        def read(registers: Tensor, i: int, blk: nn.Module, kv: Tensor) -> Tensor:
            out = blk(
                x=registers,
                y=kv,
                attn_mask=read_attn_mask,
                rope_positions=register_positions,
                rope_positions_y=patch_positions,
                window_spec=read_window_spec,
            )
            if self.learned_read_weighting:
                # Gate the read's residual contribution; g=1 reproduces the ungated read.
                return registers + self.read_gates[i] * (out - registers)
            return out

        if self.interleave:
            # [read -> self-attend] per layer: the latents re-query the input after each
            # refinement (Perceiver/DETR/Flamingo style). In multi-depth mode each read
            # draws its K/V from a successively deeper encoder layer; otherwise every read
            # re-queries the same (final-layer) source.
            for i, (read_blk, kv) in enumerate(zip(self.read_blocks, kv_per_read)):
                registers = read(registers, i, read_blk, kv)
                # latent_blocks is empty when latent self-attention is disabled; otherwise
                # it has one block per read (built above), so index by the read position.
                if self.latent_blocks:
                    registers = self.latent_blocks[i](
                        x=registers,
                        rope_positions=register_positions,
                        window_spec=latent_window_spec,
                    )
        else:
            # Legacy: all reads first, then the latent transformer.
            for i, (read_blk, kv) in enumerate(zip(self.read_blocks, kv_per_read)):
                registers = read(registers, i, read_blk, kv)
            for latent_blk in self.latent_blocks:
                registers = latent_blk(
                    x=registers,
                    rope_positions=register_positions,
                    window_spec=latent_window_spec,
                )
        return self.norm(registers), register_positions


class Encoder(FlexiVitBase):
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
        num_register_tokens: int = 0,
        learnable_channel_embeddings: bool = True,
        random_channel_embeddings: bool = False,
        num_projection_layers: int = 1,
        aggregate_then_project: bool = True,
        use_flash_attn: bool = False,
        frozen_patch_embeddings: bool = False,
        qk_norm: bool = False,
        log_token_norm_stats: bool = False,
        output_embedding_size: int | None = None,
        tokenization_config: TokenizationConfig | None = None,
        use_linear_patch_embed: bool = True,
        band_dropout_rate: float = 0.0,
        random_band_dropout: bool = False,
        band_dropout_modalities: list[str] | None = None,
        patch_embed_hidden_sizes: list[int] | None = None,
        post_proj_hidden_sizes: list[int] | None = None,
        position_encoding: str = "absolute",
        rope_base: float = 10000.0,
        rope_coordinate_scale: float = 1.0,
        rope_mixed_base: float = 10.0,
        temporal_rope_dim_frac: float = 0.25,
        rope_temporal_base: float | None = None,
        rope_temporal_coordinate_scale: float = 1.0,
        spatial_pos_encoding: str | None = None,
        attn_window_size: int | None = None,
        use_register_bottleneck: bool = False,
        register_grid_size: int | None = 0,
        register_dim: int | None = None,
        register_read_depth: int = 1,
        register_latent_depth: int = 2,
        register_num_heads: int | None = None,
        register_interleave: bool = False,
        register_read_layers: list[int] | None = None,
        register_per_depth_read_proj: bool = False,
        register_learned_read_weighting: bool = False,
        register_fused_read: str | None = None,
        register_latent_self_attn: bool = True,
        register_contrastive_source: str = "registers",
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
            num_register_tokens: Number of register tokens to use
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
            log_token_norm_stats: Whether to log the token norm stats
            output_embedding_size: If set, project tokens to this size after attention
            tokenization_config: Optional config for custom band groupings
            use_linear_patch_embed: If True, use nn.Linear for patch projection (faster).
                Set False to load checkpoints trained before this flag existed (Conv2d weights).
            band_dropout_rate: Probability of dropping each band channel during training.
            random_band_dropout: If True, sample dropout rate from Uniform(0, band_dropout_rate).
            band_dropout_modalities: If provided, only apply band dropout to these
                modalities. If None, apply to all modalities. Default: None.
            patch_embed_hidden_sizes: Optional list of hidden layer widths for a
                per-pixel MLP applied BEFORE patchification in the spatial patch
                projection. If None or empty, the projection is a single nn.Linear
                over the flattened patch (current behavior). Otherwise, each pixel's
                ``in_chans`` channel vector is mapped via
                Linear(in_chans, h[0]) -> ReLU -> ... -> Linear(h[-2], h[-1]) -> ReLU
                (weights shared across all pixels), and the resulting H x W x h[-1]
                feature map is patchified and projected to embedding_size.
            post_proj_hidden_sizes: Optional list of hidden layer widths for an MLP
                applied AFTER the patch projection. Each entry adds a
                ReLU -> Linear(prev, h) layer, applied before the norm.
            position_encoding: Position encoding mode; one of the
                ``PositionEncoding`` values.
            rope_base: Frequency base for axial RoPE.
            rope_coordinate_scale: Multiplier applied to runtime GSD-scaled RoPE coordinates.
            rope_mixed_base: Frequency base used to initialize learnable
                RoPE-Mixed frequencies.
            temporal_rope_dim_frac: Fraction of head_dim allocated to the
                temporal axis in axial 3D RoPE.
            rope_temporal_base: Optional separate frequency base for the
                temporal axis in axial 3D RoPE. ``None`` reuses ``rope_base``.
            rope_temporal_coordinate_scale: Multiplier applied to days-since-2000
                temporal RoPE coordinates (default 1.0 = raw days). E.g. set to
                1/30 for months.
            spatial_pos_encoding: Deprecated alias for ``position_encoding``.
            attn_window_size: If set, restrict every attention block (encoder self-attention,
                register read, register latent self-attention) to a square sliding window of
                this side length (in patch cells) centred on each query. Requires
                ``spatial_pos_encoding="rope"`` and ``use_flash_attn=False``. When the input
                patch grid is no larger than the window in both dims, full attention is used.
                None (default) -> full attention.
            use_register_bottleneck: If True, add a Perceiver-style spatial register
                bottleneck that reads the encoded patch tokens into a fixed register grid.
            register_grid_size: Side length of the (square) register grid; the grid has
                ``register_grid_size ** 2`` distinct per-cell registers, independent of the
                patch grid size. If ``0`` (the dynamic sentinel; legacy ``None`` is also
                accepted), use the dynamic single-latent mode: one shared latent cloned
                across a grid that matches the input patch grid at forward time (requires
                ``spatial_pos_encoding="rope"``).
            register_dim: Width of the register grid (the bottleneck dim). Defaults to
                ``embedding_size // 2`` when None.
            register_read_depth: Number of cross-attention read blocks.
            register_latent_depth: Number of latent-transformer self-attention blocks
                over the register grid.
            register_num_heads: Number of attention heads in the bottleneck blocks.
                Defaults to ``num_heads`` when None.
            register_interleave: If True, interleave the cross-attention reads with the
                latent self-attention (``[read -> self] x register_latent_depth``) so the
                registers re-query the input after each refinement, instead of reading once
                up front. Defaults to False (legacy schedule, backwards compatible).
            register_read_layers: If set, the register bottleneck reads from these (1-indexed,
                ascending) encoder block depths -- one ``[read -> self-attend]`` step per
                entry, each reading the patch tokens at that depth -- instead of re-reading
                the final layer. Recovers modality-unique information that the final layer
                drops. Forces the interleaved schedule and overrides ``register_read_depth``
                / ``register_latent_depth``. Defaults to None (final-layer read only,
                backwards compatible).
            register_per_depth_read_proj: If True, give each read block its own input
                LayerNorm and K/V down-projection instead of one shared pair. Helps
                multi-depth reads (different encoder depths have different statistics) and
                interleaved single-source reads (each block gets its own lens on the final
                layer). Requires more than one read block. Defaults to False (shared norm +
                projection, backwards compatible).
            register_learned_read_weighting: If True, give each read block a learnable scalar
                gate on its residual contribution (``z + g_d * (read_d(z) - z)``), so the
                model can weight how much each (multi-depth) read contributes. Gates init to
                1.0 (no-op at init); exposed as ``register_bottleneck.read_gates``. Defaults
                to False.
            register_fused_read: If set (``"uniform"`` or ``"learned"``), fuse the
                multi-depth K/V sources (``register_read_layers``) into ONE source read on
                the standard single-source schedule (RAEv2 multi-layer-sum style), instead
                of one read block per depth. ``"uniform"`` standardizes and averages the
                depths with no learnable combination weights; ``"learned"`` gives each
                depth its own norm + projection (mean-combined), which can re-weight
                depths. Per-depth contribution norms are exposed as
                ``register_bottleneck.last_read_source_norms`` for logging. Requires
                ``register_read_layers``; incompatible with
                ``register_per_depth_read_proj``. Defaults to None (one read per depth).
            register_latent_self_attn: If False, drop the bottleneck's latent
                self-attention blocks entirely (cross-attention reads only, no
                register-to-register mixing); the read count is unchanged. Defaults to True
                (keep the latent transformer, backwards compatible).
            register_contrastive_source: Where the contrastive (project-and-aggregate)
                head reads from when the bottleneck is active: ``"registers"`` (default,
                project from the register latents at ``register_dim``) or
                ``"encoder_tokens"`` (project from the encoder's patch-token output at the
                final embedding size, as before the bottleneck existed). Ignored when the
                bottleneck is off (always reads encoder tokens).
        """
        self.tokenization_config = tokenization_config or TokenizationConfig()
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
            tokenization_config=self.tokenization_config,
            position_encoding=position_encoding,
            spatial_pos_encoding=spatial_pos_encoding,
            rope_base=rope_base,
            rope_coordinate_scale=rope_coordinate_scale,
            rope_mixed_base=rope_mixed_base,
            temporal_rope_dim_frac=temporal_rope_dim_frac,
            rope_temporal_base=rope_temporal_base,
            rope_temporal_coordinate_scale=rope_temporal_coordinate_scale,
        )
        self.num_register_tokens = num_register_tokens
        self.has_register_tokens = num_register_tokens > 0
        self.attn_window_size = attn_window_size
        self.log_token_norm_stats = log_token_norm_stats
        if self.has_register_tokens:
            self.register_tokens = nn.Parameter(
                torch.zeros(num_register_tokens, embedding_size)
            )
        self.min_patch_size = min_patch_size
        self.max_patch_size = max_patch_size
        self.embedding_size = embedding_size
        self.use_linear_patch_embed = use_linear_patch_embed
        # Configured rate; remains inactive until ``enable_band_dropout`` is called.
        # Default is disabled so fine-tuning never applies band dropout unless the
        # caller (e.g. pretraining online encoder) explicitly enables it.
        self.band_dropout_rate = band_dropout_rate
        self.random_band_dropout = random_band_dropout
        self.band_dropout_modalities = band_dropout_modalities
        self.patch_embed_hidden_sizes = patch_embed_hidden_sizes
        self.post_proj_hidden_sizes = post_proj_hidden_sizes
        self.patch_embeddings = MultiModalPatchEmbeddings(
            self.supported_modality_names,
            self.max_patch_size,
            self.embedding_size,
            tokenization_config=self.tokenization_config,
            use_linear_patch_embed=self.use_linear_patch_embed,
            band_dropout_rate=0.0,
            random_band_dropout=self.random_band_dropout,
            band_dropout_modalities=self.band_dropout_modalities,
            patch_embed_hidden_sizes=self.patch_embed_hidden_sizes,
            post_proj_hidden_sizes=self.post_proj_hidden_sizes,
        )
        self.output_embedding_size = output_embedding_size
        # If output_embedding_size is set, project tokens to that size after attention
        self.embedding_projector: ProjectAndAggregate | None = None
        if output_embedding_size is not None:
            self.embedding_projector = ProjectAndAggregate(
                embedding_size=self.embedding_size,
                num_layers=1,
                output_embedding_size=output_embedding_size,
                only_project=True,
            )
            final_embedding_size = output_embedding_size
        else:
            final_embedding_size = self.embedding_size
        self.norm = nn.LayerNorm(self.embedding_size)

        self.use_register_bottleneck = use_register_bottleneck
        self.register_bottleneck: SpatialRegisterBottleneck | None = None
        if use_register_bottleneck:
            if register_read_layers is not None:
                if sorted(set(register_read_layers)) != list(register_read_layers):
                    raise ValueError(
                        "register_read_layers must be strictly ascending and unique, got "
                        f"{register_read_layers}"
                    )
                if not all(1 <= layer <= depth for layer in register_read_layers):
                    raise ValueError(
                        f"register_read_layers must lie in [1, depth={depth}], got "
                        f"{register_read_layers}"
                    )
            resolved_register_dim = (
                register_dim if register_dim is not None else embedding_size // 2
            )
            resolved_register_heads = (
                register_num_heads if register_num_heads is not None else num_heads
            )
            self.register_dim = resolved_register_dim
            self.register_bottleneck = SpatialRegisterBottleneck(
                encoder_embedding_size=embedding_size,
                register_dim=resolved_register_dim,
                register_grid=(
                    # 0 (or legacy None) -> dynamic single-latent grid; >0 -> fixed grid.
                    None
                    if register_grid_size is None or register_grid_size <= 0
                    else (register_grid_size, register_grid_size)
                ),
                num_heads=resolved_register_heads,
                mlp_ratio=mlp_ratio,
                read_depth=register_read_depth,
                latent_transformer_depth=register_latent_depth,
                use_2d_rope=PositionEncoding.is_2d_rope(self.position_encoding),
                rope_base=rope_base,
                qk_norm=qk_norm,
                interleave=register_interleave,
                read_layers=register_read_layers,
                per_depth_read_proj=register_per_depth_read_proj,
                learned_read_weighting=register_learned_read_weighting,
                fused_read=register_fused_read,
                latent_self_attn=register_latent_self_attn,
            )

        if register_contrastive_source not in ("registers", "encoder_tokens"):
            raise ValueError(
                "register_contrastive_source must be 'registers' or 'encoder_tokens', "
                f"got {register_contrastive_source!r}"
            )
        # Whether the contrastive head projects from the register latents (default) or from
        # the encoder patch-token output (backwards-compatible pre-bottleneck behaviour).
        self.contrastive_from_registers = (
            self.register_bottleneck is not None
            and register_contrastive_source == "registers"
        )
        # When projecting from the register tokens the head operates at the bottleneck's
        # register_dim; otherwise it reads the encoder's final-embedding-size patch tokens.
        project_aggregate_embedding_size = (
            self.register_dim
            if self.contrastive_from_registers
            else final_embedding_size
        )
        self.project_and_aggregate = ProjectAndAggregate(
            embedding_size=project_aggregate_embedding_size,
            num_layers=num_projection_layers,
            aggregate_then_project=aggregate_then_project,
        )

        self.apply(self._init_weights)

        if frozen_patch_embeddings:
            for p in self.patch_embeddings.parameters():
                p.requires_grad = False
        if self.has_register_tokens:
            self._init_register_tokens()

    def enable_band_dropout(self) -> None:
        """Enable band dropout using the configured rate.

        Band dropout is disabled by default so it never activates during
        fine-tuning. Call this only on the online encoder during pretraining.
        """
        self.patch_embeddings.band_dropout_rate = self.band_dropout_rate

    def _init_register_tokens(self) -> None:
        """Initialize the register tokens."""
        nn.init.xavier_uniform_(self.register_tokens)

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

    def add_register_tokens_and_masks(
        self,
        tokens: Tensor,
        attn_mask: Tensor | None,
        processed_register_tokens: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """Concatenate register tokens to the tokens."""
        batch_size = tokens.shape[0]
        # Expand register tokens to match batch size: [num_register_tokens, embedding_size] -> [batch_size, num_register_tokens, embedding_size]
        if processed_register_tokens is None:
            reg_tokens = self.register_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            reg_tokens = processed_register_tokens
        # Concatenate register tokens at the beginning: [batch_size, seq_len, embedding_size] -> [batch_size, num_register_tokens + seq_len, embedding_size]
        tokens = torch.cat([reg_tokens, tokens], dim=1)
        if attn_mask is not None:
            # Create mask for register tokens (all True - they should participate in attention)
            reg_mask = torch.ones(
                batch_size,
                self.num_register_tokens,
                dtype=attn_mask.dtype,
                device=attn_mask.device,
            )
            attn_mask = torch.cat([reg_mask, attn_mask], dim=1)
        else:
            reg_mask = None
        return tokens, attn_mask

    def pop_register_tokens(self, tokens: Tensor) -> tuple[Tensor, Tensor]:
        """Pop the register tokens from the tokens."""
        register_tokens = tokens[:, : self.num_register_tokens, :]
        tokens = tokens[:, self.num_register_tokens :, :]
        return tokens, register_tokens

    def get_token_norm_stats(
        self, tokens: Tensor, register_tokens: Tensor
    ) -> dict[str, float]:
        """Get the token norm stats."""
        # Compute norms for register tokens: [batch_size, num_register_tokens]
        register_tokens_norms = torch.norm(register_tokens, dim=2)
        reg_norms_flat = register_tokens_norms.flatten()
        reg_stats = {
            "register_mean": reg_norms_flat.mean().item(),
            "register_min": reg_norms_flat.min().item(),
            "register_max": reg_norms_flat.max().item(),
        }

        # Compute norms for non-register tokens: [batch_size, seq_len]
        nonreg_tokens_norms = torch.norm(tokens, dim=2)
        nonreg_norms_flat = nonreg_tokens_norms.flatten()
        percentiles = [25.0, 75.0, 90.0, 95.0, 99.0]
        nonreg_percentiles = torch.quantile(
            nonreg_norms_flat.float(),
            torch.tensor(
                [p / 100.0 for p in percentiles], device=nonreg_norms_flat.device
            ),
        ).tolist()
        nonreg_stats = {
            "nonregister_mean": nonreg_norms_flat.mean().item(),
            "nonregister_min": nonreg_norms_flat.min().item(),
            "nonregister_max": nonreg_norms_flat.max().item(),
            "nonregister_std": nonreg_norms_flat.std().item(),
            "nonregister_25th": nonreg_percentiles[0],
            "nonregister_75th": nonreg_percentiles[1],
            "nonregister_90th": nonreg_percentiles[2],
            "nonregister_95th": nonreg_percentiles[3],
            "nonregister_99th": nonreg_percentiles[4],
        }

        token_norm_stats = {**reg_stats, **nonreg_stats}
        return token_norm_stats

    def _maybe_remove_masked_tokens(
        self,
        tokens: Tensor,
        mask: Tensor,
        fast_pass: bool,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Remove masked tokens from the tokens and masks."""
        if fast_pass and not self.use_flash_attn:
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
    ) -> tuple[dict[str, Tensor], dict[str, Any] | None, dict[str, Any] | None]:
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
        positions = self.build_rope_positions(
            tokens_only_dict,
            original_masks_dict,
            patch_size,
            input_res,
            timestamps=timestamps,
        )
        # Full (pre-masking) positions in collapsed order, kept for the register
        # bottleneck read so registers attend over the encoded *visible* patch tokens
        # using their original coordinates (`positions` below is reduced/packed in place).
        register_kv_positions = positions

        # Windowed (local) spatial attention setup. Active only when the patch grid is
        # larger than the window in some dim; otherwise we leave the fast (full-attention)
        # path untouched. `patch_spatial_flag` is in the full (pre-masking) collapsed order
        # for the register read; `encoder_spatial_flag` is reduced alongside `positions`.
        window_half_extent: float | None = None
        patch_spatial_flag: Tensor | None = None
        encoder_spatial_flag: Tensor | None = None
        window_spec: WindowSpec | None = None
        if self.attn_window_size is not None:
            if not PositionEncoding.is_2d_rope(self.position_encoding):
                raise ValueError(
                    "attn_window_size requires a 2D RoPE position_encoding "
                    '(e.g. "rope" or "rope_mixed")'
                )
            grid_h, grid_w = self._patch_grid_hw(tokens_only_dict)
            if grid_h > self.attn_window_size or grid_w > self.attn_window_size:
                gsd_ratio = (
                    CompositeEncodings.calculate_gsd_ratio(input_res, patch_size)
                    * self.rope_coordinate_scale
                )
                window_half_extent = (self.attn_window_size / 2.0) * gsd_ratio
                patch_spatial_flag = self.build_spatial_token_mask(
                    tokens_only_dict, original_masks_dict
                )
                encoder_spatial_flag = patch_spatial_flag

        tokens_dict.update(original_masks_dict)

        tokens, mask = self.collapse_and_combine_hwtc(tokens_dict)

        tokens, indices, new_mask, seq_lengths, max_seqlen, bool_mask = (
            self._maybe_remove_masked_tokens(tokens, mask, fast_pass)
        )
        if positions is not None and bool_mask is not None:
            positions, _, _, _, _ = self.remove_masked_tokens(positions, bool_mask)
            if encoder_spatial_flag is not None:
                reduced_flag, _, _, _, _ = self.remove_masked_tokens(
                    encoder_spatial_flag[..., None].float(), bool_mask
                )
                encoder_spatial_flag = reduced_flag.squeeze(-1) > 0.5

        if exit_ids_seq is not None:
            exit_ids_seq, _, _, _, _ = self.remove_masked_tokens(
                exit_ids_seq, bool_mask
            )
            # still linear projections
            exited_tokens, _, _, _, _ = self.remove_masked_tokens(
                exited_tokens, bool_mask
            )

        # Pack x tokens
        if self.use_flash_attn:
            cu_seqlens = get_cumulative_sequence_lengths(seq_lengths)
            og_shape = tokens.shape
            tokens = self.pack_tokens(tokens, new_mask)
            if positions is not None:
                positions = self.pack_tokens(positions, new_mask)
        else:
            cu_seqlens = None

        attn_mask = self._maybe_get_attn_mask(
            new_mask,
            fast_pass=fast_pass,
        )

        if self.has_register_tokens:
            tokens, attn_mask = self.add_register_tokens_and_masks(tokens, attn_mask)
            if positions is not None:
                positions = self.add_register_positions(positions)

        if window_half_extent is not None:
            # Now that positions include the prepended global register tokens, assemble the
            # window ingredients. The per-block mask is built lazily, one query-chunk at a
            # time inside attention (see WindowSpec / Attention._windowed_sdpa), so peak
            # memory is bounded by the chunk rather than by N*N -- without this the full
            # unmasked eval sequence (no MAE masking) would OOM a dense [B, N, N] mask. The
            # spec carries key validity, so it replaces the [B, N] key mask for every block
            # (train and eval alike). Register tokens are global (non-spatial); padded keys
            # are excluded via the true padding mask (`new_mask`), independent of fast_pass.
            assert encoder_spatial_flag is not None
            if self.has_register_tokens:
                reg_flag = encoder_spatial_flag.new_zeros(
                    encoder_spatial_flag.shape[0], self.num_register_tokens
                )
                encoder_spatial_flag = torch.cat(
                    [reg_flag, encoder_spatial_flag], dim=1
                )
            key_valid: Tensor | None = None
            if new_mask is not None:
                key_valid = new_mask.bool()
                if self.has_register_tokens:
                    reg_valid = key_valid.new_ones(
                        key_valid.shape[0], self.num_register_tokens
                    )
                    key_valid = torch.cat([reg_valid, key_valid], dim=1)
            is_global = ~encoder_spatial_flag
            window_spec = WindowSpec(
                q_positions=positions,
                k_positions=positions,
                half_extent=window_half_extent,
                q_is_global=is_global,
                k_is_global=is_global,
                key_valid=key_valid,
            )
            attn_mask = None  # validity carried by window_spec.key_valid

        # Multi-depth register reads: stash the (raw, in-loop) patch tokens at the
        # configured 1-indexed depths so the bottleneck can read each one. The patch stack
        # is independent of the registers (registers never write back), so caching here is
        # equivalent to truly interleaving the reads into the block loop.
        multi_depth_read_layers: set[int] = set()
        if (
            self.register_bottleneck is not None
            and self.register_bottleneck.multi_depth
        ):
            assert self.register_bottleneck.read_layers is not None
            multi_depth_read_layers = set(self.register_bottleneck.read_layers)
        cached_read_tokens: dict[int, Tensor] = {}

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
                rope_positions=positions,
                window_spec=window_spec,
            )
            # Stash this depth's output for the multi-depth register read (1-indexed).
            if (i_blk + 1) in multi_depth_read_layers:
                cached_read_tokens[i_blk + 1] = tokens

        if self.has_register_tokens:
            tokens, register_tokens = self.pop_register_tokens(tokens)
            token_norm_stats = (
                self.get_token_norm_stats(tokens, register_tokens)
                if self.log_token_norm_stats
                else None
            )
        else:
            token_norm_stats = None

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
        # we apply the norm before we add the removed tokens,
        # so that the norm is only computed against "real" tokens
        tokens = self.norm(tokens)
        # we don't care about the mask returned by add_removed_tokens, since we will
        # just use the original, unclipped mask here
        tokens = self._maybe_add_removed_tokens(tokens, indices, new_mask, fast_pass)

        # Perceiver-style read: a fixed register grid reads the encoded visible patch
        # tokens (bool_mask restricts the read to ONLINE_ENCODER keys), followed by the
        # latent transformer inside the bottleneck module.
        register_output = None
        if self.register_bottleneck is not None:
            # Dynamic mode clones the single latent to match the patch grid; fixed mode
            # ignores spatial_grid and uses its own learned grid.
            spatial_grid = (
                self._patch_grid_hw(tokens_only_dict)
                if self.register_bottleneck.dynamic_grid
                else None
            )
            if self.register_bottleneck.multi_depth:
                assert self.register_bottleneck.read_layers is not None

                def _finalize_read_tokens(raw: Tensor) -> Tensor:
                    # Bring a cached, in-loop block output into the shape the bottleneck
                    # reads: drop register tokens, unpack (flash), and re-add masked tokens
                    # (the read masks them out). No norm here -- the bottleneck applies its
                    # own input_norm to every K/V source, so an encoder norm would be a
                    # redundant double-norm (and would mismatch across read depths).
                    t = raw
                    if self.has_register_tokens:
                        t, _ = self.pop_register_tokens(t)
                    if self.use_flash_attn:
                        t = self.unpack_tokens(t, new_mask, og_shape)
                    return self._maybe_add_removed_tokens(
                        t, indices, new_mask, fast_pass
                    )

                # One K/V source per read layer, each finalized from its cached (pre-norm)
                # block output so all depths are normalized identically by the bottleneck's
                # input_norm. Every read layer is cached above, including the final depth
                # when it is a read layer.
                patch_tokens_arg: Tensor | list[Tensor] = [
                    _finalize_read_tokens(cached_read_tokens[depth])
                    for depth in self.register_bottleneck.read_layers
                ]
            else:
                patch_tokens_arg = tokens
            registers, register_positions = self.register_bottleneck(
                patch_tokens=patch_tokens_arg,
                patch_positions=register_kv_positions,
                visible_mask=bool_mask,
                spatial_grid=spatial_grid,
                window_half_extent=window_half_extent,
                # `patch_spatial_flag` is in the full (pre-masking) order matching
                # `register_kv_positions`; non-spatial patches read globally.
                patch_is_global=(
                    ~patch_spatial_flag if patch_spatial_flag is not None else None
                ),
            )
            register_output = {
                "registers": registers,
                "register_positions": register_positions,
            }

        tokens_per_modality_dict = self.split_and_expand_per_modality(
            tokens, modalities_to_dims_dict
        )
        # merge original masks and the processed tokens
        tokens_per_modality_dict.update(original_masks_dict)
        return tokens_per_modality_dict, token_norm_stats, register_output

    def forward(
        self,
        x: MaskedOlmoEarthSample,
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

        patchified_tokens_and_masks = self.patch_embeddings.forward(x, patch_size)

        register_output: dict[str, Any] | None = None
        token_norm_stats: dict[str, Any] | None = None
        if token_exit_cfg is None or any(
            [exit_depth > 0 for exit_depth in token_exit_cfg.values()]
        ):
            patchified_tokens_and_masks, token_norm_stats, register_output = (
                self.apply_attn(
                    x=patchified_tokens_and_masks,
                    timestamps=x.timestamps,
                    patch_size=patch_size,
                    input_res=input_res,
                    token_exit_cfg=token_exit_cfg,
                    fast_pass=fast_pass,
                )
            )
        else:
            token_norm_stats = {}
        output = TokensAndMasks(**patchified_tokens_and_masks)

        # Project to output_embedding_size if configured
        if self.embedding_projector is not None:
            output = self.embedding_projector(output)

        output_dict: dict[str, Any] = {
            "tokens_and_masks": output,
        }
        if token_norm_stats:
            output_dict["token_norm_stats"] = token_norm_stats

        if register_output is not None:
            output_dict["registers"] = register_output["registers"]
            output_dict["register_positions"] = register_output["register_positions"]

        if not fast_pass:
            if self.contrastive_from_registers:
                # The contrastive projection reads the register tokens (only) and is sized
                # to register_dim. Registers are produced whenever attention runs (the
                # standard, token_exit_cfg=None pass); the only path that skips them is the
                # all-zero-exit target pass, which discards project_aggregated anyway.
                if register_output is not None:
                    output_dict["project_aggregated"] = self.project_and_aggregate(
                        register_output["registers"]
                    )
            else:
                # No bottleneck, or register_contrastive_source="encoder_tokens": project
                # from the encoder's patch-token output (masked-mean pooled), as before the
                # bottleneck existed.
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


class PredictorBase(FlexiVitBase):
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
        tokenization_config: TokenizationConfig | None = None,
        position_encoding: str = "absolute",
        rope_base: float = 10000.0,
        rope_coordinate_scale: float = 1.0,
        rope_mixed_base: float = 10.0,
        temporal_rope_dim_frac: float = 0.25,
        rope_temporal_base: float | None = None,
        rope_temporal_coordinate_scale: float = 1.0,
        spatial_pos_encoding: str | None = None,
        use_register_bottleneck: bool = False,
        register_dim: int | None = None,
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
            tokenization_config: Optional config for custom band groupings
            position_encoding: Position encoding mode; one of the
                ``PositionEncoding`` values.
            rope_base: Frequency base for axial RoPE.
            rope_coordinate_scale: Multiplier applied to runtime GSD-scaled RoPE coordinates.
            rope_mixed_base: Frequency base used to initialize learnable
                RoPE-Mixed frequencies.
            temporal_rope_dim_frac: Fraction of head_dim allocated to the
                temporal axis in axial 3D RoPE.
            rope_temporal_base: Optional separate frequency base for the
                temporal axis in axial 3D RoPE. ``None`` reuses ``rope_base``.
            rope_temporal_coordinate_scale: Multiplier applied to days-since-2000
                temporal RoPE coordinates (default 1.0 = raw days). E.g. set to
                1/30 for months.
            spatial_pos_encoding: Deprecated alias for ``position_encoding``.
            use_register_bottleneck: If True, the decoder cross-attends to the encoder
                register grid instead of the visible patch tokens.
            register_dim: Width of the register grid; required when use_register_bottleneck.
        """
        self.tokenization_config = tokenization_config or TokenizationConfig()
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
            tokenization_config=self.tokenization_config,
            position_encoding=position_encoding,
            spatial_pos_encoding=spatial_pos_encoding,
            rope_base=rope_base,
            rope_coordinate_scale=rope_coordinate_scale,
            rope_mixed_base=rope_mixed_base,
            temporal_rope_dim_frac=temporal_rope_dim_frac,
            rope_temporal_base=rope_temporal_base,
            rope_temporal_coordinate_scale=rope_temporal_coordinate_scale,
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

        self.use_register_bottleneck = use_register_bottleneck
        self.register_to_decoder_embed: nn.Linear | None = None
        if use_register_bottleneck:
            if register_dim is None:
                raise ValueError(
                    "register_dim is required when use_register_bottleneck is True"
                )
            self.register_to_decoder_embed = nn.Linear(
                register_dim, decoder_embedding_size, bias=True
            )

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
        registers: Tensor | None = None,
        register_positions: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Apply attention to the tokens."""
        tokens_only_dict, original_masks_dict, modalities_to_dims_dict = (
            self.split_tokens_masks_and_dims(x)
        )
        tokens_dict = self.composite_encodings(
            tokens_only_dict, timestamps, patch_size, input_res
        )
        positions = self.build_rope_positions(
            tokens_only_dict,
            original_masks_dict,
            patch_size,
            input_res,
            timestamps=timestamps,
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
        if positions is not None:
            positions_to_decode, unmasked_positions = self.split_x_y_positions(
                positions,
                indices,
                max_length_of_tokens_to_decode,
                max_length_of_unmasked_tokens,
            )
        else:
            positions_to_decode = None
            unmasked_positions = None
        # Pack x tokens
        if self.use_flash_attn:
            og_shape_tokens_to_decode = tokens_to_decode.shape
            tokens_to_decode = self.pack_tokens(
                tokens_to_decode, tokens_to_decode_mask.bool()
            )
            if positions_to_decode is not None:
                positions_to_decode = self.pack_tokens(
                    positions_to_decode, tokens_to_decode_mask.bool()
                )
            og_shape_unmasked_tokens = unmasked_tokens.shape
            unmasked_tokens = self.pack_tokens(
                unmasked_tokens, unmasked_tokens_mask.bool()
            )
            if unmasked_positions is not None:
                unmasked_positions = self.pack_tokens(
                    unmasked_positions, unmasked_tokens_mask.bool()
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

        # Decoder context: either the visible patch tokens (default) or, for the register
        # bottleneck, the encoder register grid (projected to the decoder dim). The decode
        # queries are mask tokens at masked-patch coords; they attend only to this context.
        if registers is not None:
            if self.register_to_decoder_embed is None:
                raise ValueError(
                    "Predictor received registers but was built without "
                    "use_register_bottleneck=True"
                )
            context = self.register_to_decoder_embed(registers)
            context_positions = register_positions
            num_registers = context.shape[1]
            if self.use_flash_attn:
                register_bool = torch.ones(
                    context.shape[0],
                    num_registers,
                    dtype=torch.bool,
                    device=context.device,
                )
                context = self.pack_tokens(context, register_bool)
                if context_positions is not None:
                    context_positions = self.pack_tokens(
                        context_positions, register_bool
                    )
                cu_seqlens_context = get_cumulative_sequence_lengths(
                    torch.full(
                        (register_bool.shape[0],),
                        num_registers,
                        dtype=torch.int32,
                        device=context.device,
                    )
                )
            else:
                cu_seqlens_context = None
            max_length_of_context = num_registers
            context_attn_mask = None
        else:
            context = unmasked_tokens
            context_positions = unmasked_positions
            cu_seqlens_context = cu_seqlens_unmasked_tokens
            max_length_of_context = max_length_of_unmasked_tokens
            context_attn_mask = (
                unmasked_tokens_mask.bool() if not self.use_flash_attn else None
            )

        for blk in self.blocks:
            # note that we are not taking the inverse of the mask, since split_x_y gives us
            # true values for values we want to take part in attention
            tokens_to_decode = blk(
                x=tokens_to_decode,
                y=context,
                attn_mask=context_attn_mask,
                cu_seqlens_q=cu_seqlens_tokens_to_decode,
                cu_seqlens_k=cu_seqlens_context,
                max_seqlen_q=max_length_of_tokens_to_decode,
                max_seqlen_k=max_length_of_context,
                rope_positions=positions_to_decode,
                rope_positions_y=context_positions,
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
        registers: Tensor | None = None,
        register_positions: Tensor | None = None,
    ) -> TokensAndMasks:
        """Generate predictions from encoded token representations.

        Args:
            x: TokensAndMasks containing the encoded tokens to make predictions from
            timestamps: Timestamps of the tokens
            patch_size: Patch size of the tokens
            input_res: Input resolution of the tokens
            registers: Optional encoder register grid ``[B, n_reg, register_dim]``. When
                provided (register bottleneck), the decoder cross-attends to it instead of
                the visible patch tokens.
            register_positions: Optional ``[B, n_reg, 2]`` register coordinates for RoPE.

        Returns:
            TokensAndMasks containing the predicted tokens and their masks
        """
        decoder_emedded_dict = x.as_dict()
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
            decoder_emedded_dict,
            timestamps,
            patch_size,
            input_res,
            registers=registers,
            register_positions=register_positions,
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

            num_band_sets = self.tokenization_config.get_num_bandsets(modality)
            for idx in range(num_band_sets):
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
    num_register_tokens: int = 0
    learnable_channel_embeddings: bool = True
    random_channel_embeddings: bool = False
    num_projection_layers: int = 1
    aggregate_then_project: bool = True
    use_flash_attn: bool = False
    frozen_patch_embeddings: bool = False
    qk_norm: bool = False
    log_token_norm_stats: bool = False
    output_embedding_size: int | None = None
    tokenization_config: TokenizationConfig | None = None
    use_linear_patch_embed: bool = True
    band_dropout_rate: float = 0.0
    random_band_dropout: bool = False
    band_dropout_modalities: list[str] | None = None
    patch_embed_hidden_sizes: list[int] | None = None
    post_proj_hidden_sizes: list[int] | None = None
    position_encoding: str = "absolute"
    rope_base: float = 10000.0
    rope_coordinate_scale: float = 1.0
    rope_mixed_base: float = 10.0
    temporal_rope_dim_frac: float = 0.25
    rope_temporal_base: float | None = None
    rope_temporal_coordinate_scale: float = 1.0
    # Deprecated alias for ``position_encoding``. Kept as a field (not dropped)
    # so old checkpoint configs deserialized via Config.from_dict still carry it
    # through to __post_init__ for reconciliation.
    spatial_pos_encoding: str | None = None
    # Windowed (local) spatial attention: each token attends only to tokens within a
    # square window of side ``attn_window_size`` patch cells centred on it (sliding
    # window), applied to encoder self-attention and the register read + latent
    # self-attention. None -> full attention (backwards compatible). When the input
    # patch grid is no larger than the window in both dims, full attention is used.
    # Requires a 2D RoPE position_encoding and is incompatible with use_flash_attn.
    attn_window_size: int | None = None
    # Perceiver-style spatial register bottleneck (sweepable).
    use_register_bottleneck: bool = False
    # >0 -> fixed grid of distinct per-cell latents; 0 -> dynamic single cloned latent
    # whose grid matches the patch grid at forward time (requires rope). 0 (not None) is
    # the dynamic sentinel so it survives serialization: ``as_config_dict`` drops None
    # values, which silently turned dynamic-grid checkpoints back into fixed grids on
    # reload. Legacy None is coerced to 0 in ``__post_init__``.
    register_grid_size: int = 0
    register_dim: int | None = None
    register_read_depth: int = 1
    register_latent_depth: int = 2
    register_num_heads: int | None = None
    # Interleave reads with latent self-attention ([read -> self] x register_latent_depth)
    # instead of reading once up front. False -> legacy schedule (backwards compatible).
    register_interleave: bool = False
    # Multi-depth reads: 1-indexed, ascending encoder depths the bottleneck reads from
    # (one [read -> self-attend] step per entry). Forces the interleaved schedule and
    # overrides register_read_depth / register_latent_depth. None -> final-layer read only.
    register_read_layers: list[int] | None = None
    # Give each read block its own input norm + K/V down-projection instead of sharing one
    # pair (multi-depth: per-depth stats; interleave: a distinct lens per re-read). Needs
    # >1 read block. False -> shared (backwards compatible).
    register_per_depth_read_proj: bool = False
    # Learnable per-read residual gate (z + g_d * (read_d(z) - z)), letting the model weight
    # how much each (multi-depth) read contributes. Gates init to 1.0 (no-op at init).
    # False -> ungated reads (backwards compatible).
    register_learned_read_weighting: bool = False
    # Fuse the multi-depth K/V sources into ONE source read on the standard single-source
    # schedule (RAEv2 multi-layer-sum style), instead of one read block per depth.
    # "uniform": parameter-free standardize-and-average (training cannot re-weight the
    # combination, so mid-depth features survive even where the pretext loss would discard
    # them); "learned": per-depth norm + projection, mean-combined (can re-weight depths;
    # per-depth contribution norms are logged to watch for collapse onto the final layer).
    # Requires register_read_layers; incompatible with register_per_depth_read_proj.
    # None -> one read per depth (backwards compatible).
    register_fused_read: str | None = None
    # Where the contrastive head reads when the bottleneck is on: "registers" (default,
    # project from the register latents) or "encoder_tokens" (project from the encoder
    # patch-token output, as before the bottleneck existed). Ignored when bottleneck off.
    register_contrastive_source: str = "registers"
    # If False, drop the register bottleneck's latent self-attention blocks entirely
    # (cross-attention reads only, no register-to-register mixing). The read count is
    # unchanged. Default True keeps the latent transformer (backwards compatible).
    register_latent_self_attn: bool = True

    def __post_init__(self) -> None:
        """Coerce raw dicts to TokenizationConfig for old checkpoint compatibility."""
        if isinstance(self.tokenization_config, dict):
            self.tokenization_config = TokenizationConfig(**self.tokenization_config)
        self.position_encoding = resolve_position_encoding(
            self.position_encoding, self.spatial_pos_encoding
        )
        self.spatial_pos_encoding = None

    def validate(self) -> None:
        """Validate the configuration."""
        if len(self.supported_modalities) == 0:
            raise ValueError("At least one modality must be added!")
        else:
            for modality in self.supported_modalities:
                if modality not in Modality.values():
                    raise ValueError(f"Modality {modality} is not supported")
        if self.band_dropout_modalities is not None:
            unknown = set(self.band_dropout_modalities) - set(
                self.supported_modality_names
            )
            if unknown:
                raise ValueError(
                    f"band_dropout_modalities contains modalities not in "
                    f"supported_modality_names: {unknown}"
                )
        if self.tokenization_config is not None:
            self.tokenization_config.validate()
        if self.position_encoding not in PositionEncoding.values():
            raise ValueError(
                f"position_encoding must be one of {PositionEncoding.values()}, "
                f"got {self.position_encoding}"
            )
        if self.rope_base <= 0:
            raise ValueError(f"rope_base must be positive, got {self.rope_base}")
        if self.rope_coordinate_scale <= 0:
            raise ValueError(
                f"rope_coordinate_scale must be positive, got {self.rope_coordinate_scale}"
            )
        if self.attn_window_size is not None:
            if self.attn_window_size <= 0:
                raise ValueError(
                    f"attn_window_size must be positive, got {self.attn_window_size}"
                )
            if not PositionEncoding.is_2d_rope(self.position_encoding):
                raise ValueError(
                    "attn_window_size requires a 2D RoPE position_encoding (the window "
                    "is computed from per-token RoPE coordinates)"
                )
            if self.use_flash_attn:
                raise ValueError(
                    "attn_window_size is incompatible with use_flash_attn (the flash "
                    "varlen path cannot express a 2D spatial mask); set "
                    "use_flash_attn=False"
                )
        if self.use_register_bottleneck:
            # Legacy None sentinel -> 0 (dynamic single-latent grid).
            if self.register_grid_size is None:
                self.register_grid_size = 0
            if self.register_grid_size < 0:
                raise ValueError(
                    f"register_grid_size must be >= 0 (0 = dynamic single-latent grid), "
                    f"got {self.register_grid_size}"
                )
            if self.register_grid_size == 0 and not PositionEncoding.is_2d_rope(
                self.position_encoding
            ):
                raise ValueError(
                    "register_grid_size=0 (dynamic single-latent bottleneck) requires "
                    "a 2D RoPE position_encoding"
                )
            register_dim = (
                self.register_dim
                if self.register_dim is not None
                else self.embedding_size // 2
            )
            register_heads = (
                self.register_num_heads
                if self.register_num_heads is not None
                else self.num_heads
            )
            if register_dim % register_heads != 0:
                raise ValueError(
                    f"register_dim ({register_dim}) must be divisible by "
                    f"register_num_heads ({register_heads})"
                )
            if (
                PositionEncoding.is_2d_rope(self.position_encoding)
                and (register_dim // register_heads) % 4 != 0
            ):
                raise ValueError(
                    "2D RoPE requires register head_dim divisible by 4, got "
                    f"{register_dim // register_heads}"
                )
            if self.register_read_layers is not None:
                if sorted(set(self.register_read_layers)) != list(
                    self.register_read_layers
                ):
                    raise ValueError(
                        "register_read_layers must be strictly ascending and unique, got "
                        f"{self.register_read_layers}"
                    )
                if not all(
                    1 <= layer <= self.depth for layer in self.register_read_layers
                ):
                    raise ValueError(
                        f"register_read_layers must lie in [1, depth={self.depth}], got "
                        f"{self.register_read_layers}"
                    )
            if self.register_fused_read is not None:
                if self.register_fused_read not in ("uniform", "learned"):
                    raise ValueError(
                        "register_fused_read must be 'uniform' or 'learned', got "
                        f"{self.register_fused_read!r}"
                    )
                if self.register_read_layers is None:
                    raise ValueError(
                        "register_fused_read requires register_read_layers (the depths "
                        "to fuse)"
                    )
                if self.register_per_depth_read_proj:
                    raise ValueError(
                        "register_fused_read is incompatible with "
                        "register_per_depth_read_proj (the fusion replaces the per-depth "
                        "read projections)"
                    )
        elif self.register_read_layers is not None:
            raise ValueError(
                "register_read_layers requires use_register_bottleneck=True"
            )
        elif self.register_fused_read is not None:
            raise ValueError(
                "register_fused_read requires use_register_bottleneck=True"
            )
        if self.register_contrastive_source not in ("registers", "encoder_tokens"):
            raise ValueError(
                "register_contrastive_source must be 'registers' or 'encoder_tokens', "
                f"got {self.register_contrastive_source!r}"
            )
        if self.rope_mixed_base <= 0:
            raise ValueError(
                f"rope_mixed_base must be positive, got {self.rope_mixed_base}"
            )
        if not 0.0 < self.temporal_rope_dim_frac < 1.0:
            raise ValueError(
                f"temporal_rope_dim_frac must be in (0, 1), got "
                f"{self.temporal_rope_dim_frac}"
            )
        if self.rope_temporal_base is not None and self.rope_temporal_base <= 0:
            raise ValueError(
                f"rope_temporal_base must be positive, got {self.rope_temporal_base}"
            )
        if self.rope_temporal_coordinate_scale <= 0:
            raise ValueError(
                "rope_temporal_coordinate_scale must be positive, got "
                f"{self.rope_temporal_coordinate_scale}"
            )
        validate_position_encoding(
            position_encoding=self.position_encoding,
            head_dim=self.embedding_size // self.num_heads,
            temporal_rope_dim_frac=self.temporal_rope_dim_frac,
        )

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
        # exclude_none drops register_grid_size when None, but None is meaningful here
        # (dynamic single-latent bottleneck), so pass it through explicitly.
        kwargs["register_grid_size"] = self.register_grid_size
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
    tokenization_config: TokenizationConfig | None = None
    position_encoding: str = "absolute"
    rope_base: float = 10000.0
    rope_coordinate_scale: float = 1.0
    rope_mixed_base: float = 10.0
    temporal_rope_dim_frac: float = 0.25
    rope_temporal_base: float | None = None
    rope_temporal_coordinate_scale: float = 1.0
    # Deprecated alias for ``position_encoding``. Kept as a field (not dropped)
    # so old checkpoint configs deserialized via Config.from_dict still carry it
    # through to __post_init__ for reconciliation.
    spatial_pos_encoding: str | None = None
    # Perceiver-style register bottleneck: when True the decoder cross-attends to the
    # encoder register grid (of width register_dim) instead of the visible patch tokens.
    use_register_bottleneck: bool = False
    register_dim: int | None = None

    def __post_init__(self) -> None:
        """Coerce raw dicts to TokenizationConfig for old checkpoint compatibility."""
        if isinstance(self.tokenization_config, dict):
            self.tokenization_config = TokenizationConfig(**self.tokenization_config)
        self.position_encoding = resolve_position_encoding(
            self.position_encoding, self.spatial_pos_encoding
        )
        self.spatial_pos_encoding = None

    def validate(self) -> None:
        """Validate the configuration."""
        if len(self.supported_modalities) == 0:
            raise ValueError("At least one modality must be added!")
        else:
            for modality in self.supported_modalities:
                if modality not in Modality.values():
                    raise ValueError(f"Modality {modality} is not supported")
        if self.use_register_bottleneck and self.register_dim is None:
            raise ValueError(
                "register_dim must be set when use_register_bottleneck is True"
            )
        if self.tokenization_config is not None:
            self.tokenization_config.validate()
        if self.position_encoding not in PositionEncoding.values():
            raise ValueError(
                f"position_encoding must be one of {PositionEncoding.values()}, "
                f"got {self.position_encoding}"
            )
        if self.rope_base <= 0:
            raise ValueError(f"rope_base must be positive, got {self.rope_base}")
        if self.rope_coordinate_scale <= 0:
            raise ValueError(
                f"rope_coordinate_scale must be positive, got {self.rope_coordinate_scale}"
            )
        if self.rope_mixed_base <= 0:
            raise ValueError(
                f"rope_mixed_base must be positive, got {self.rope_mixed_base}"
            )
        if not 0.0 < self.temporal_rope_dim_frac < 1.0:
            raise ValueError(
                f"temporal_rope_dim_frac must be in (0, 1), got "
                f"{self.temporal_rope_dim_frac}"
            )
        if self.rope_temporal_base is not None and self.rope_temporal_base <= 0:
            raise ValueError(
                f"rope_temporal_base must be positive, got {self.rope_temporal_base}"
            )
        if self.rope_temporal_coordinate_scale <= 0:
            raise ValueError(
                "rope_temporal_coordinate_scale must be positive, got "
                f"{self.rope_temporal_coordinate_scale}"
            )
        validate_position_encoding(
            position_encoding=self.position_encoding,
            head_dim=self.decoder_embedding_size // self.num_heads,
            temporal_rope_dim_frac=self.temporal_rope_dim_frac,
        )

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
