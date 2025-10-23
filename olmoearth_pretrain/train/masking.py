"""Masking module."""

import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, NamedTuple

import numpy as np
import torch
from class_registry import ClassRegistry
from einops import rearrange, repeat
from olmo_core.config import Config

from olmoearth_pretrain.data.constants import MISSING_VALUE, Modality, ModalitySpec
from olmoearth_pretrain.data.dataset import OlmoEarthSample
from olmoearth_pretrain.decorators import experimental
from olmoearth_pretrain.types import ArrayTensor

logger = logging.getLogger(__name__)

# all bandset indices should be tuples of (modality, bandset_idx) sow e can create a power set of these combinations from it
ALL_BANDSET_IDXS: list[tuple[str, int]] = []
for modality in Modality.values():
    for bandset_idx in range(modality.num_band_sets):
        ALL_BANDSET_IDXS.append((modality.name, bandset_idx))


class MaskValue(Enum):
    """Masks can take 4 possible values.

    ONLINE_ENCODER: The token is seen by the online encoder
    TARGET_ENCODER_ONLY: The token is seen by the target encoder only
    DECODER: The token is seen by the decoder only
    MISSING: The token is missing
    """

    ONLINE_ENCODER = 0
    TARGET_ENCODER_ONLY = 1
    DECODER = 2
    MISSING = 3


class MaskedOlmoEarthSample(NamedTuple):
    """A masked sample of the data from the OlmoEarth Pretrain dataset.

    We always require sentinel2 data.
    This is a namedtuple that contains the data for a single sample from the OlmoEarth Pretrain dataset.
    latlon and timestamps are the same for all modalities.
    For each modality. we have an ArrayTensor named by modality, and a mask for each modality named by modality_mask.
    we also have a mask for the latlon called latlon_mask
    """

    timestamps: (
        ArrayTensor  # [B, T, D=3], where D=[day, month, year] (months are zero indexed)
    )
    sentinel2_l2a: ArrayTensor | None = None
    sentinel2_l2a_mask: ArrayTensor | None = None
    sentinel1: ArrayTensor | None = None
    sentinel1_mask: ArrayTensor | None = None
    worldcover: ArrayTensor | None = None
    worldcover_mask: ArrayTensor | None = None
    latlon: ArrayTensor | None = None  # [B, 2]
    latlon_mask: ArrayTensor | None = None
    openstreetmap_raster: ArrayTensor | None = None
    openstreetmap_raster_mask: ArrayTensor | None = None
    srtm: ArrayTensor | None = None
    srtm_mask: ArrayTensor | None = None
    landsat: ArrayTensor | None = None
    landsat_mask: ArrayTensor | None = None
    naip: ArrayTensor | None = None
    naip_mask: ArrayTensor | None = None
    naip_10: ArrayTensor | None = None
    naip_10_mask: ArrayTensor | None = None
    gse: ArrayTensor | None = None
    gse_mask: ArrayTensor | None = None
    cdl: ArrayTensor | None = None
    cdl_mask: ArrayTensor | None = None
    worldpop: ArrayTensor | None = None
    worldpop_mask: ArrayTensor | None = None
    worldcereal: ArrayTensor | None = None
    worldcereal_mask: ArrayTensor | None = None
    wri_canopy_height_map: ArrayTensor | None = None
    wri_canopy_height_map_mask: ArrayTensor | None = None
    era5_10: ArrayTensor | None = None
    era5_10_mask: ArrayTensor | None = None

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

    def unmask(self) -> "MaskedOlmoEarthSample":
        """Return an unmasked MaskedOlmoEarthSample.

        All mask values are MaskValue.ONLINE_ENCODER except for MaskValue.MISSING,
        which remain MISSING.
        """
        return_dict: dict[str, ArrayTensor] = {}
        for key, val in self.as_dict().items():
            if val is None:
                continue
            if key.endswith("mask"):
                # 1s where it is missing, 0 elsewhere
                all_but_missing = val == MaskValue.MISSING
                return_dict[key] = val * all_but_missing
            else:
                return_dict[key] = val
        return MaskedOlmoEarthSample(**return_dict)

    @property
    def modalities(self) -> list[str]:
        """Get the present modalities in this instance of MaskedOlmoEarthSample."""
        return [
            field
            for field in self._fields
            if not field.endswith("_mask")
            and field != "timestamps"
            and getattr(self, field) is not None
        ]

    @staticmethod
    def get_masked_modality_name(modality: str) -> str:
        """Get the masked modality name."""
        return f"{modality}_mask"

    @staticmethod
    def get_unmasked_modality_name(modality_mask_name: str) -> str:
        """Get the unmasked modality name."""
        return modality_mask_name.replace("_mask", "")

    # TODO: add unit test because this does modlaity based checking
    @classmethod
    def from_olmoearthsample(
        cls,
        sample: OlmoEarthSample,
    ) -> "MaskedOlmoEarthSample":
        """Transforms a OlmoEarthSample into a MaskedOlmoEarthSample.

        This function assumes modalities are uniformly missing.
        """
        masked_sample_dict = {}
        for key, t in sample.as_dict(ignore_nones=False).items():
            if key == "timestamps":
                # lets assume timestamps is not None
                masked_sample_dict[key] = t
            else:
                if t is None:
                    masked_sample_dict[key] = None
                    masked_sample_dict[
                        MaskedOlmoEarthSample.get_masked_modality_name(key)
                    ] = None
                else:
                    masked_sample_dict[key] = t
                    masked_sample_dict[
                        MaskedOlmoEarthSample.get_masked_modality_name(key)
                    ] = (
                        torch.ones(sample.shape(key, mask=False))
                        * MaskValue.ONLINE_ENCODER.value
                    )

        return MaskedOlmoEarthSample(**masked_sample_dict)

    @classmethod
    def from_dict(cls, dict: dict[str, Any]) -> "MaskedOlmoEarthSample":
        """Create a MaskedOlmoEarthSample from a dictionary, creating empty tensors for missing modalities.

        Args:
            dict: Dictionary representation of the MaskedOlmoEarthSample.
        """
        return cls(**dict)


class MaskingStrategy:
    """Abstract base class for masking strategies.

    Be sure to implement apply_mask in subclasses.
    """

    @property
    def name(self) -> str:
        """Return the name of the masking strategy."""
        return self.__class__.__name__.replace("MaskingStrategy", "").lower()

    @property
    def encode_ratio(self) -> float:
        """Return the encode ratio."""
        if not hasattr(self, "_encode_ratio"):
            raise AttributeError("Encode ratio not set")
        return self._encode_ratio

    @property
    def decode_ratio(self) -> float:
        """Return the decode ratio."""
        if not hasattr(self, "_decode_ratio"):
            raise AttributeError("Decode ratio not set")
        return self._decode_ratio

    def apply_mask(
        self, batch: OlmoEarthSample, patch_size: int | None = None, **kwargs: Any
    ) -> MaskedOlmoEarthSample:
        """Apply masking to the input data.

        Args:
            batch: Input data of type OlmoEarthSample
            patch_size: Optional patch size for spatial masking strategies
            **kwargs: Additional arguments for maskings
        """
        raise NotImplementedError("Subclasses must implement this method")

    def get_missing_mask(
        self, instance: torch.Tensor, modality: ModalitySpec, mask: torch.Tensor
    ) -> torch.Tensor:
        """Get the missing mask for the input data."""
        missing_mask = mask.new_zeros(mask.shape, dtype=torch.bool)
        for i, band_set_indices in enumerate(modality.bandsets_as_indices()):
            instance_band_set = instance[..., band_set_indices]
            missing_mask_band_set = instance_band_set == MISSING_VALUE
            missing_mask_band_set_any = missing_mask_band_set.any(dim=-1)
            # If any band in the band set is missing, set the whole band set to missing
            missing_mask[..., i] = missing_mask_band_set_any
        return missing_mask

    def fill_mask_with_missing_values(
        self, instance: torch.Tensor, mask: torch.Tensor, modality: ModalitySpec
    ) -> torch.Tensor:
        """Apply a missing mask to the input data."""
        missing_mask = self.get_missing_mask(instance, modality, mask)
        # If we are changing the mask, we need to clone it as it may be a view of a masked used by different modalities
        if missing_mask.any():
            output_mask = mask.clone()
            output_mask[missing_mask] = MaskValue.MISSING.value
        else:
            output_mask = mask
        return output_mask

    def _create_random_mask(
        self,
        modality: ModalitySpec,
        shape: torch.Size,
        patch_size_at_16: int,
        device: torch.device | None = None,
        encode_ratio: float | None = None,
        decode_ratio: float | None = None,
    ) -> ArrayTensor:
        mask_shape = list(shape)
        mask_shape[-1] = modality.num_band_sets
        if modality.is_spatial:
            patch_size = patch_size_at_16 * modality.image_tile_size_factor
            mask_shape[1] //= patch_size
            mask_shape[2] //= patch_size

        if modality.is_spatial or modality.is_multitemporal:
            b = shape[0]
            num_tokens = math.prod(mask_shape[1:])
        else:
            num_tokens = math.prod(mask_shape[:-1])

        if encode_ratio is None:
            encode_ratio = self.encode_ratio
        if decode_ratio is None:
            decode_ratio = self.decode_ratio

        encode_tokens = int(num_tokens * encode_ratio)
        decode_tokens = int(num_tokens * decode_ratio)
        target_tokens = int(num_tokens - (encode_tokens + decode_tokens))
        flat_mask_tokens = torch.cat(
            [
                torch.full(
                    (encode_tokens,), MaskValue.ONLINE_ENCODER.value, device=device
                ),
                torch.full((decode_tokens,), MaskValue.DECODER.value, device=device),
                torch.full(
                    (target_tokens,), MaskValue.TARGET_ENCODER_ONLY.value, device=device
                ),
            ]
        )

        if modality.is_spatial or modality.is_multitemporal:
            masks = [
                flat_mask_tokens[torch.randperm(num_tokens, device=device)]
                for i in range(b)
            ]
            flat_mask_tokens = torch.stack(masks)
        else:
            flat_mask_tokens = flat_mask_tokens[
                torch.randperm(num_tokens, device=device)
            ]

        mask = flat_mask_tokens.view(*mask_shape)
        if modality.is_spatial:
            mask = repeat(
                mask, "b h w ... -> b (h hp) (w wp) ...", hp=patch_size, wp=patch_size
            )

        return mask

    def _random_fill_unmasked(
        self,
        mask: torch.Tensor,
        modality: ModalitySpec,
        patch_size_at_16: int,
        encode_ratio: float | None = None,
        decode_ratio: float | None = None,
    ) -> ArrayTensor:
        """This function assumes B=1."""
        assert mask.shape[0] == 1, (
            f"_random_fill_unmasked does not support B != 1, got input shape {mask.shape}"
        )
        device = mask.device
        if modality.is_spatial:
            patch_size = patch_size_at_16 * modality.image_tile_size_factor
            # the first two dimensions are spatial; lets turn them
            # from h, w to p_h, p_w
            mask = mask[:, 0::patch_size, 0::patch_size]

        original_shape = mask.shape
        # this only works because we assume B = 1
        flat_mask = mask.flatten()  # N tokens
        not_missing_tokens = flat_mask != MaskValue.MISSING.value
        num_not_missing_tokens = sum(not_missing_tokens)

        if encode_ratio is None:
            encode_ratio = self.encode_ratio
        if decode_ratio is None:
            decode_ratio = self.decode_ratio

        if num_not_missing_tokens == 1:
            encode_tokens = 1
            decode_tokens = 0
        else:
            encode_tokens = int(num_not_missing_tokens * encode_ratio)
            decode_tokens = int(num_not_missing_tokens * decode_ratio)

        target_tokens = int(num_not_missing_tokens - (encode_tokens + decode_tokens))
        flat_mask_tokens = torch.cat(
            [
                torch.full(
                    (encode_tokens,), MaskValue.ONLINE_ENCODER.value, device=device
                ),
                torch.full((decode_tokens,), MaskValue.DECODER.value, device=device),
                torch.full(
                    (target_tokens,), MaskValue.TARGET_ENCODER_ONLY.value, device=device
                ),
            ]
        )

        flat_mask_tokens = flat_mask_tokens[
            torch.randperm(num_not_missing_tokens, device=device)
        ]
        flat_mask[not_missing_tokens] = flat_mask_tokens
        mask = flat_mask.view(*original_shape)
        if modality.is_spatial:
            mask = repeat(
                mask, "b h w ... -> b (h hp) (w wp) ...", hp=patch_size, wp=patch_size
            )

        return mask


MASKING_STRATEGY_REGISTRY = ClassRegistry[MaskingStrategy]()


@MASKING_STRATEGY_REGISTRY.register("time")
class TimeMaskingStrategy(MaskingStrategy):
    """Time structured random masking of the input data."""

    def __init__(
        self,
        encode_ratio: float = 0.5,
        decode_ratio: float = 0.5,
    ) -> None:
        """Initialize the masking strategy."""
        self._encode_ratio = encode_ratio
        self._decode_ratio = decode_ratio

    def _create_temporal_mask(
        self,
        shape: torch.Size,
        timesteps_with_at_least_one_modality: torch.Tensor,
        device: torch.device | None = None,
    ) -> ArrayTensor:
        b = shape[0]
        t = shape[-2]
        # timesteps withat least one modality are the only ones we can put as either encoder and decoder randomly pick from those instead
        # can we relax the all sample contraint here as we are doing per sample stuff anyways
        present_t = timesteps_with_at_least_one_modality.shape[0]  # across all samples
        assert present_t >= 3
        logger.info(f"Present timesteps: {present_t}")
        encode_times = max(int(self.encode_ratio * present_t), 1)
        decode_times = max(int(self.decode_ratio * present_t), 1)
        target_times = present_t - encode_times - decode_times
        logger.info(
            f"Encode times: {encode_times}, Decode times: {decode_times}, Target times: {target_times}"
        )
        # Create mask values only for the encodable timesteps
        encodable_mask_values = torch.cat(
            [
                torch.full(
                    (encode_times,), MaskValue.ONLINE_ENCODER.value, device=device
                ),
                torch.full((decode_times,), MaskValue.DECODER.value, device=device),
                torch.full(
                    (target_times,), MaskValue.TARGET_ENCODER_ONLY.value, device=device
                ),
            ]
        )

        # Create masks for each sample in the batch
        masks = [
            torch.full(
                (t,), MaskValue.TARGET_ENCODER_ONLY.value, device=device
            ).index_put_(
                (timesteps_with_at_least_one_modality,),
                encodable_mask_values[torch.randperm(present_t, device=device)],
            )
            for _ in range(b)
        ]

        mask = torch.stack(masks)
        return mask

    def apply_mask(
        self, batch: OlmoEarthSample, patch_size: int | None = None, **kwargs: Any
    ) -> MaskedOlmoEarthSample:
        """Apply random masking to the input data.

        Masking happens temporally, with whole time steps having the same mask. Non-temporal data is randomly masked.

        Args:
            batch: Input data of type OlmoEarthSample
            patch_size: patch size applied to sample
            **kwargs: Additional arguments for maskings

        Returns:
            MaskedOlmoEarthSample containing the masked data and mask
        """
        if patch_size is None:
            raise ValueError("patch_size must be provided for time masking")
        output_dict: dict[str, ArrayTensor | None] = {}
        temporal_mask = None
        timesteps_with_at_least_one_modality = (
            batch.timesteps_with_at_least_one_modality
        )
        num_valid_timesteps = timesteps_with_at_least_one_modality.shape[0]
        for modality_name in batch.modalities:
            instance = getattr(batch, modality_name)
            if instance is None:
                # set instance and mask to None
                output_dict[modality_name] = None
                output_dict[
                    MaskedOlmoEarthSample.get_masked_modality_name(modality_name)
                ] = None
            else:
                if modality_name == "timestamps":
                    output_dict[modality_name] = instance
                    continue

                if isinstance(instance, torch.Tensor):
                    device: torch.device | None = instance.device
                else:
                    device = None

                modality = Modality.get(modality_name)
                shape = instance.shape
                if not modality.is_multitemporal or num_valid_timesteps < 3:
                    mask = self._create_random_mask(modality, shape, patch_size, device)
                else:
                    if temporal_mask is None:
                        # if there are timesteps that we wouldn't want to pick we should call a seprate mask creation function
                        logger.info(
                            f"Creating temporal mask for modality {modality.name}"
                        )
                        temporal_mask = self._create_temporal_mask(
                            shape, timesteps_with_at_least_one_modality, device
                        )
                    b_s = modality.num_band_sets
                    b, h, w = list(shape[:-2]) + [1] * (3 - len(shape[:-2]))
                    # Repeat shares a view of the temporal masks so if we don't clone future changes may propogate across modalities
                    mask = repeat(
                        temporal_mask, "b t -> b h w t b_s", h=h, w=w, b_s=b_s
                    )
                    mask = mask.view(*shape[:-1], b_s).clone()
                # After setting up encoder and decoder masks, fill in missing values

                mask = self.fill_mask_with_missing_values(instance, mask, modality)
                output_dict[modality_name] = instance
                output_dict[
                    MaskedOlmoEarthSample.get_masked_modality_name(modality_name)
                ] = mask
        return MaskedOlmoEarthSample(**output_dict)


@experimental(
    "This masking strategy is experimental and may not work with all combinations of modalities"
)
@MASKING_STRATEGY_REGISTRY.register("space")
class SpaceMaskingStrategy(MaskingStrategy):
    """Spatially structured random masking of the input data."""

    def __init__(
        self,
        encode_ratio: float = 0.5,
        decode_ratio: float = 0.5,
    ) -> None:
        """Initialize the masking strategy."""
        self._encode_ratio = encode_ratio
        self._decode_ratio = decode_ratio

    def _create_patch_spatial_mask(
        self,
        modality: ModalitySpec,
        shape: torch.Size,
        patch_size_at_16: int,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Create a h_p x w_p spatial mask.

        Here, h_p and w_p are the number of patches along height and width dimension
        respectively.

        The mask computed here is modality-agnostic, but we still expect a specific
        modality to be passed since it will be used to compute h_p/w_p. The mask will
        then need to be resized using _resize_spatial_mask_for_modality to the
        modality's patch size.

        Args:
            modality: the modality we are using to compute h_p/w_p.
            shape: the shape of the image for that modality.
            patch_size_at_16: the patch size measured in 10 m/pixel pixels.
            device: the device to use.
        """
        if not modality.is_spatial:
            raise ValueError("Non-spatial modality {modality}")

        b, h, w = shape[:3]

        patch_size = patch_size_at_16 * modality.image_tile_size_factor
        assert (h % patch_size == 0) and (w % patch_size == 0)
        h_p = h // patch_size
        w_p = w // patch_size

        patches = h_p * w_p
        encode_patches = int(self.encode_ratio * patches)
        decode_patches = int(self.decode_ratio * patches)
        target_patches = patches - encode_patches - decode_patches

        flat_mask = torch.cat(
            [
                torch.full(
                    (encode_patches,), MaskValue.ONLINE_ENCODER.value, device=device
                ),
                torch.full((decode_patches,), MaskValue.DECODER.value, device=device),
                torch.full(
                    (target_patches,),
                    MaskValue.TARGET_ENCODER_ONLY.value,
                    device=device,
                ),
            ]
        )

        masks = [flat_mask[torch.randperm(patches, device=device)] for i in range(b)]
        random_batch_mask = torch.stack(masks)
        return rearrange(random_batch_mask, "b (h w) -> b h w", h=h_p, w=w_p)

    def _resize_spatial_mask_for_modality(
        self,
        patch_mask: torch.Tensor,
        modality: ModalitySpec,
        patch_size_at_16: int,
    ) -> ArrayTensor:
        """Resize the mask computed by _create_patch_spatial_mask for the given modality.

        Args:
            patch_mask: the mask computed by _create_patch_spatial_mask.
            modality: the modality to compute the mask for.
            patch_size_at_16: the patch size measured in 10 m/pixel pixels.
        """
        if not modality.is_spatial:
            raise ValueError("Non-spatial modality {modality}")

        patch_size = patch_size_at_16 * modality.image_tile_size_factor
        mask = repeat(
            patch_mask, "b h w -> b (h hps) (w wps)", hps=patch_size, wps=patch_size
        )
        return mask

    def apply_mask(
        self, batch: OlmoEarthSample, patch_size: int | None = None, **kwargs: Any
    ) -> MaskedOlmoEarthSample:
        """Apply random masking to the input data.

        Masking happens in patchified form, with whole patches having the same mask. Non-spatial data is randomly masked.

        Args:
            batch: Input data of type OlmoEarthSample
            patch_size: patch size applied to sample, at an image_tile_size_factor == 16
            **kwargs: Additional arguments for maskings

        Returns:
            MaskedOlmoEarthSample containing the masked data and mask
        """
        if patch_size is None:
            raise ValueError("patch_size must be provided for space masking")
        output_dict: dict[str, ArrayTensor | None] = {}
        patch_spatial_mask = None
        # Same spatial mask for all modalities
        for modality_name in batch.modalities:
            instance = getattr(batch, modality_name)
            if instance is None:
                # set instance and mask to None
                output_dict[modality_name] = None
                output_dict[
                    MaskedOlmoEarthSample.get_masked_modality_name(modality_name)
                ] = None
                continue

            if modality_name == "timestamps":
                output_dict[modality_name] = instance
                continue

            if isinstance(instance, torch.Tensor):
                device: torch.device | None = instance.device
            else:
                device = None

            modality = Modality.get(modality_name)
            shape = instance.shape
            if not modality.is_spatial:
                logger.warning(
                    f"Modality {modality.name} is not spatial, random masking strategy will be applied"
                )
                mask = self._create_random_mask(modality, shape, patch_size, device)
            else:
                if patch_spatial_mask is None:
                    logger.info(f"Creating spatial mask for modality {modality.name}")
                    patch_spatial_mask = self._create_patch_spatial_mask(
                        modality, shape, patch_size, device
                    )
                resized_spatial_mask = self._resize_spatial_mask_for_modality(
                    patch_spatial_mask, modality, patch_size
                )

                if resized_spatial_mask.shape[0:3] != shape[0:3]:
                    raise ValueError(
                        f"Mismached shapes for {modality.name}: "
                        f"computed mask {mask.shape} but image shape is {shape}"
                    )

                if len(shape) == 5:
                    t = shape[-2]
                else:
                    t = 1
                b_s = modality.num_band_sets
                # Mask is a view of the spatial mask, so changes to mask will change spatial_mask
                mask = repeat(resized_spatial_mask, "... -> ... t b_s", t=t, b_s=b_s)
                mask = mask.view(*shape[:-1], b_s).clone()
            mask = self.fill_mask_with_missing_values(instance, mask, modality)

            # Keep data as is
            output_dict[modality_name] = instance
            output_dict[
                MaskedOlmoEarthSample.get_masked_modality_name(modality_name)
            ] = mask
        return MaskedOlmoEarthSample(**output_dict)


@MASKING_STRATEGY_REGISTRY.register("modality")
class ModalityMaskingStrategy(MaskingStrategy):
    """Modality structured random masking of the input data."""

    def __init__(
        self,
        encode_ratio: float = 0.5,
        decode_ratio: float = 0.5,
    ) -> None:
        """Initialize the masking strategy."""
        self._encode_ratio = encode_ratio
        self._decode_ratio = decode_ratio

    def apply_mask(
        self, batch: OlmoEarthSample, patch_size: int | None = None, **kwargs: Any
    ) -> MaskedOlmoEarthSample:
        """Randomly mask out modalities in the input data.

        Entire modalities (per instance) are assigned the same mask.

        Args:
            batch: Input data of type OlmoEarthSample
            patch_size: Optional patch size for spatial masking strategies
            **kwargs: Additional arguments for maskings

        Returns:
            MaskedOlmoEarthSample containing the masked data and mask
        """
        output_dict: dict[str, ArrayTensor | None] = {"timestamps": batch.timestamps}
        present_modalities = [b for b in batch.modalities if b != "timestamps"]

        num_present_modalities = len(present_modalities)
        encode_modalities = max(1, int(self.encode_ratio * num_present_modalities))
        decode_modalities = max(1, int(self.decode_ratio * num_present_modalities))
        target_modalities = (
            num_present_modalities - encode_modalities - decode_modalities
        )

        # TODO get device for this
        band_mask_per_instance = torch.cat(
            [
                torch.full((encode_modalities,), MaskValue.ONLINE_ENCODER.value),
                torch.full((decode_modalities,), MaskValue.DECODER.value),
                torch.full((target_modalities,), MaskValue.TARGET_ENCODER_ONLY.value),
            ]
        )
        batch_masks = [
            band_mask_per_instance[torch.randperm(num_present_modalities)]
            for i in range(batch.batch_size)
        ]
        random_batch_mask = torch.stack(batch_masks)
        for idx, modality_name in enumerate(present_modalities):
            instance = getattr(batch, modality_name)
            output_dict[modality_name] = instance
            modality = Modality.get(modality_name)

            if isinstance(instance, torch.Tensor):
                device: torch.device | None = instance.device
            else:
                device = None

            modality_mask = torch.tensor(random_batch_mask[:, idx], device=device)
            shape = instance.shape
            b_s = modality.num_band_sets
            b, h, w, t = list(shape[:-1]) + [1] * (4 - len(shape[:-1]))
            mask = repeat(modality_mask, "b -> b h w t b_s", h=h, w=w, b_s=b_s, t=t)
            # Ensure we don't do index_put_ on expanded tensors is deprecated.
            mask = mask.view(*shape[:-1], b_s).contiguous()
            mask = self.fill_mask_with_missing_values(instance, mask, modality)
            output_dict[
                MaskedOlmoEarthSample.get_masked_modality_name(modality_name)
            ] = mask

        return MaskedOlmoEarthSample(**output_dict)


@MASKING_STRATEGY_REGISTRY.register("space_time")
class SpaceTimeMaskingStrategy(MaskingStrategy):
    """Randomly select space or time masking and apply it to the input data."""

    def __init__(
        self,
        encode_ratio: float = 0.5,
        decode_ratio: float = 0.5,
    ) -> None:
        """Initialize the masking strategy."""
        self._encode_ratio = encode_ratio
        self._decode_ratio = decode_ratio
        self.generator = np.random.default_rng(0)

        self.space_strategy = SpaceMaskingStrategy(encode_ratio, decode_ratio)
        self.time_strategy = TimeMaskingStrategy(encode_ratio, decode_ratio)

    def apply_mask(
        self, batch: OlmoEarthSample, patch_size: int | None = None, **kwargs: Any
    ) -> MaskedOlmoEarthSample:
        """Apply space or time masking to the input data."""
        has_enough_timesteps = batch.valid_time >= 3
        # I need a timestamp mask

        if not has_enough_timesteps:
            logger.debug(f"Valid time: {batch.valid_time}, Time: {batch.time}")
        if (self.generator.random() < 0.5) or (not has_enough_timesteps):
            logger.info("Applying space masking")
            return self.space_strategy.apply_mask(batch, patch_size, **kwargs)
        else:
            logger.info("Applying time masking")
            return self.time_strategy.apply_mask(batch, patch_size, **kwargs)


@MASKING_STRATEGY_REGISTRY.register("random_space")
class RandomSpaceMaskingStrategy(MaskingStrategy):
    """Randomly select space or random masking."""

    def __init__(
        self,
        encode_ratio: float = 0.5,
        decode_ratio: float = 0.5,
    ) -> None:
        """Initialize the masking strategy."""
        self._encode_ratio = encode_ratio
        self._decode_ratio = decode_ratio
        self.generator = np.random.default_rng(0)

        self.random_strategy = RandomMaskingStrategy(encode_ratio, decode_ratio)
        self.space_strategy = SpaceMaskingStrategy(encode_ratio, decode_ratio)

    def apply_mask(
        self, batch: OlmoEarthSample, patch_size: int | None = None, **kwargs: Any
    ) -> MaskedOlmoEarthSample:
        """Apply space or time masking to the input data."""
        if self.generator.random() < 0.5:
            logger.info("Applying space masking")
            return self.space_strategy.apply_mask(batch, patch_size, **kwargs)
        else:
            logger.info("Applying random masking")
            return self.random_strategy.apply_mask(batch, patch_size, **kwargs)


@MASKING_STRATEGY_REGISTRY.register("modality_space_time")
class ModalitySpaceTimeMaskingStrategy(MaskingStrategy):
    """Randomly select modality, space or time masking and apply it to the input data."""

    def __init__(
        self,
        encode_ratio: float = 0.5,
        decode_ratio: float = 0.5,
    ) -> None:
        """Initialize the masking strategy."""
        self._encode_ratio = encode_ratio
        self._decode_ratio = decode_ratio
        self.generator = np.random.default_rng(0)

        self.space_strategy = SpaceMaskingStrategy(encode_ratio, decode_ratio)
        self.time_strategy = TimeMaskingStrategy(encode_ratio, decode_ratio)
        self.modality_strategy = ModalityMaskingStrategy(encode_ratio, decode_ratio)

    def apply_mask(
        self, batch: OlmoEarthSample, patch_size: int | None = None, **kwargs: Any
    ) -> MaskedOlmoEarthSample:
        """Apply band or space or time masking to the input data."""
        has_enough_timesteps = batch.valid_time >= 3
        has_enough_modalities = (len(batch.as_dict()) - 1) >= 2

        possible_strategies: list[MaskingStrategy] = [self.space_strategy]
        if has_enough_timesteps:
            possible_strategies.append(self.time_strategy)
        if has_enough_modalities:
            possible_strategies.append(self.modality_strategy)

        selected_strategy: MaskingStrategy = self.generator.choice(possible_strategies)
        if not isinstance(selected_strategy, ModalityMaskingStrategy):
            kwargs["patch_size"] = patch_size

        return selected_strategy.apply_mask(batch, **kwargs)


class ModalityCrossMaskingStrategy(MaskingStrategy):
    """Cross-modality masking strategy that separates bandsets for encoding and decoding.

    This strategy wraps a base masking strategy and adds cross-modality logic on top. It selects
    which bandsets (modality, bandset_idx pairs) should be used for encoding vs decoding, enabling
    the model to learn cross-modality relationships.

    Algorithm Overview:
    1. Apply the base masking strategy (e.g., space/time masking)
    2. Identify which modalities/bandsets are present in each sample
    3. For each sample, select which bandsets to encode vs decode based on modality count:
       - 1 modality: encode only (no cross-modality decoding possible)
       - 2 modalities: encode first, decode second (simple cross-modality)
       - 3+ modalities: randomly select subset to encode, rest to decode
    4. Apply bandset-level masking rules:
       - Bandsets not selected for encoding: suppress ONLINE_ENCODER tokens
       - Bandsets not selected for decoding: suppress DECODER tokens
    5. Handle edge cases where samples end up with no encoded/decoded tokens
       (rare, typically only in S2-only ablations with small spatial dimensions)
    """

    def __init__(
        self,
        strategy: MaskingStrategy,
        encode_ratio: float = 0.5,
        decode_ratio: float = 0.5,
        allow_encoding_decoding_same_bandset: bool = False,
        min_encoded_bandsets: int | None = None,
        max_encoded_bandsets: int | None = None,
        min_decoded_bandsets: int | None = None,
        max_decoded_bandsets: int | None = None,
        only_decode_modalities: list[str] = [],
    ) -> None:
        """Initialize the masking strategy.

        Args:
            strategy: The base masking strategy to apply before cross-modality masking.
            encode_ratio: Ratio of tokens to encode (default: 0.5). Used by the base strategy.
            decode_ratio: Ratio of tokens to decode (default: 0.5). Used by the base strategy.
            allow_encoding_decoding_same_bandset: If True, allows the same bandset to be both
                encoded and decoded. If False (default), encoded and decoded bandsets are disjoint.
            min_encoded_bandsets: Minimum number of bandsets to encode per sample. If None (default),
                encodes all available bandsets when there are 3+ modalities, or 1 bandset when there are 2 modalities.
            max_encoded_bandsets: Maximum number of bandsets to encode per sample. If None (default),
                encodes all available bandsets.
            min_decoded_bandsets: Minimum number of bandsets to decode per sample. Only used when
                allow_encoding_decoding_same_bandset=True. If None (default), uses 1.
            max_decoded_bandsets: Maximum number of bandsets to decode per sample. Only used when
                allow_encoding_decoding_same_bandset=True. If None (default), uses all available bandsets.
            only_decode_modalities: List of modality names that should only be used for decoding,
                never for encoding. Empty list by default (all modalities can be encoded).
        """
        self._encode_ratio = encode_ratio
        self._decode_ratio = decode_ratio
        self.strategy = strategy
        self.allow_encoding_decoding_same_bandset = allow_encoding_decoding_same_bandset
        if min_encoded_bandsets is None:
            assert max_encoded_bandsets is None, (
                "max_encoded_bandsets must be set if min_encoded_bandsets is set"
            )
        else:
            assert min_encoded_bandsets > 1, (
                "min_encoded_bandsets must be greater than 1 so that we don't only  \
                encode a modality that is randomly masked on batch dimension ie latlon"
            )
        self.min_encoded_bandsets = min_encoded_bandsets
        self.max_encoded_bandsets = max_encoded_bandsets
        self.min_decoded_bandsets = min_decoded_bandsets
        self.max_decoded_bandsets = max_decoded_bandsets
        self.only_decode_modalities = only_decode_modalities

    def get_sample_present_modalities_bandsets(
        self, batch: MaskedOlmoEarthSample
    ) -> list[list[tuple[str, int]]]:
        """Get the modalities that are present for each sample."""
        masked_sample_dict = batch.as_dict(return_none=False)
        batch_size = batch.timestamps.shape[0]
        present_modalities_bandsets: list[list[tuple[str, int]]] = [
            [] for _ in range(batch_size)
        ]
        for modality in batch.modalities:
            if modality == "timestamps":
                continue
            modality_mask_name = MaskedOlmoEarthSample.get_masked_modality_name(
                modality
            )
            modality_mask = masked_sample_dict[modality_mask_name]
            missing_values_mask = modality_mask == MaskValue.MISSING.value
            # Find the samples where the modality is completely missing
            is_modality_completely_missing_for_samples = torch.all(
                missing_values_mask.view(batch_size, -1), dim=1
            )
            is_modality_present_for_samples = (
                ~is_modality_completely_missing_for_samples
            )
            num_bandsets = modality_mask.shape[-1]

            present_sample_indices = torch.where(is_modality_present_for_samples)[0]
            for sample_idx in present_sample_indices:
                sample_idx = sample_idx.item()
                for bandset_idx in range(num_bandsets):
                    # check if that modality bandset has any encoded tokens if it has no encoded tokens it is not present
                    is_any_tokens_encoded_for_sample = (
                        torch.sum(
                            modality_mask[sample_idx, ..., bandset_idx]
                            == MaskValue.ONLINE_ENCODER.value
                        )
                        > 0
                    )
                    # only say something is present if it has any encoded tokens
                    # A little hacky but basically means that we leave the bandset untouched for encoding and decoding
                    if (
                        not is_any_tokens_encoded_for_sample
                        and modality not in self.only_decode_modalities
                    ):
                        continue
                    present_modalities_bandsets[sample_idx].append(
                        (modality, bandset_idx)
                    )
        return present_modalities_bandsets

    def select_encoded_decoded_bandsets(
        self, present_modalities_bandsets: list[list[tuple[str, int]]]
    ) -> list[tuple[set[tuple[str, int]], set[tuple[str, int]]]]:
        """Select the encoded and decoded bandsets for each sample.

        Routes each sample to the appropriate handler based on the number of present modalities.
        Returns a list of (encoded_bandsets, decoded_bandsets) tuples, one per sample.
        """
        encoded_decoded_bandsets: list[
            tuple[set[tuple[str, int]], set[tuple[str, int]]]
        ] = []
        for sample_idx in range(len(present_modalities_bandsets)):
            present = present_modalities_bandsets[sample_idx]
            encoded, decoded = self._select_bandsets_for_sample(present)
            encoded_decoded_bandsets.append((encoded, decoded))
        return encoded_decoded_bandsets

    def _select_bandsets_for_sample(
        self, present: list[tuple[str, int]]
    ) -> tuple[set[tuple[str, int]], set[tuple[str, int]]]:
        """Select encoded and decoded bandsets for a single sample based on modality count."""
        if len(present) == 1:
            return self._select_bandsets_single_modality(present)
        elif len(present) == 2:
            return self._select_bandsets_two_modalities(present)
        else:
            return self._select_bandsets_multiple_modalities(present)

    def _select_bandsets_single_modality(
        self, present: list[tuple[str, int]]
    ) -> tuple[set[tuple[str, int]], set[tuple[str, int]]]:
        """Handle single modality case: encode only, no cross-modality decoding possible."""
        encoded_bandset_idxs = set(present)
        decoded_bandset_idxs = set()
        return encoded_bandset_idxs, decoded_bandset_idxs

    def _select_bandsets_two_modalities(
        self, present: list[tuple[str, int]]
    ) -> tuple[set[tuple[str, int]], set[tuple[str, int]]]:
        """Handle two modality case: encode first, decode second."""
        encoded_bandset_idxs = set([present[0]])
        decoded_bandset_idxs = set([present[1]])
        return encoded_bandset_idxs, decoded_bandset_idxs

    def _select_bandsets_multiple_modalities(
        self, present: list[tuple[str, int]]
    ) -> tuple[set[tuple[str, int]], set[tuple[str, int]]]:
        """Handle 3+ modality case: randomly select subset to encode, rest to decode."""
        num_present = len(present)
        encodable = self._filter_encodable_bandsets(present)
        num_encode = self._compute_num_bandsets_to_encode(encodable, num_present)
        encoded = self._randomly_select_encoded_bandsets(encodable, num_encode)
        num_decode = self._compute_num_bandsets_to_decode(num_present)
        decoded = self._randomly_select_decoded_bandsets(present, encoded, num_decode)
        return encoded, decoded

    def _filter_encodable_bandsets(
        self, present: list[tuple[str, int]]
    ) -> list[tuple[str, int]]:
        """Filter out bandsets that are restricted to decoding only."""
        return [
            modality_bandset
            for modality_bandset in present
            if modality_bandset[0] not in self.only_decode_modalities
        ]

    def _compute_num_bandsets_to_encode(
        self, encodable: list[tuple[str, int]], num_present: int
    ) -> int:
        """Compute how many bandsets to encode based on configuration."""
        num_encodable = len(encodable)
        # Determine upper limit for encoding
        upper_limit = num_encodable
        if not self.allow_encoding_decoding_same_bandset:
            # Need to reserve at least one for decoding
            upper_limit -= 1

        # Apply configured max, if any
        if self.max_encoded_bandsets is None:
            max_encoded_bandsets = upper_limit
        else:
            max_encoded_bandsets = min(self.max_encoded_bandsets, upper_limit)

        # Apply configured min, if any
        if self.min_encoded_bandsets is None:
            min_encoded_bandsets = num_encodable
        else:
            min_encoded_bandsets = min(self.min_encoded_bandsets, num_encodable)

        # Ensure min <= max
        min_encoded_bandsets = min(min_encoded_bandsets, max_encoded_bandsets)

        # Randomly sample within the range
        return np.random.randint(min_encoded_bandsets, max_encoded_bandsets + 1)

    def _randomly_select_encoded_bandsets(
        self, encodable: list[tuple[str, int]], num_encode: int
    ) -> set[tuple[str, int]]:
        """Randomly select which bandsets to encode."""
        encoded_idxs = np.random.choice(
            len(encodable), size=num_encode, replace=False
        )
        return set([encodable[i] for i in encoded_idxs])

    def _compute_num_bandsets_to_decode(self, num_present: int) -> int:
        """Compute how many bandsets to decode based on configuration."""
        min_decoded = min(self.min_decoded_bandsets or 1, num_present)
        max_decoded = min(self.max_decoded_bandsets or num_present, num_present)
        return min_decoded, max_decoded

    def _randomly_select_decoded_bandsets(
        self,
        present: list[tuple[str, int]],
        encoded: set[tuple[str, int]],
        num_decode_tuple: tuple[int, int],
    ) -> set[tuple[str, int]]:
        """Randomly select which bandsets to decode."""
        min_decoded, max_decoded = num_decode_tuple

        if self.allow_encoding_decoding_same_bandset:
            # Can select from all present bandsets
            num_decoded_bandsets = np.random.randint(min_decoded, max_decoded + 1)
            decoded_idxs = np.random.choice(
                len(present), size=num_decoded_bandsets, replace=False
            )
            return set([present[i] for i in decoded_idxs])
        else:
            # Can only select from bandsets not used for encoding
            # When disjoint, we decode ALL bandsets not used for encoding
            available = list(set(present) - encoded)
            num_available = len(available)
            min_decoded = min(min_decoded, num_available)
            max_decoded = min(max_decoded, num_available)
            # Select all available decoded bandsets
            num_decoded_bandsets = num_available
            decoded_idxs = np.random.choice(
                len(available), size=num_decoded_bandsets, replace=False
            )
            return set([available[i] for i in decoded_idxs])

    def overide_strategy_mask(self, modality_spec: ModalitySpec) -> bool:
        """Overide the mask for a modality depending on the strategy being modality cross masked.

        e.g in time masking, static in time data is randomly masked but we want that data to be either used to predict temporally masked data or
        predicted from temporal data.
        """
        return False

    def apply_bandset_mask_rules(
        self,
        masked_batch: MaskedOlmoEarthSample,
        encoded_decoded_bandsets: list[
            tuple[set[tuple[str, int]], set[tuple[str, int]]]
        ],
        present_modalities_bandsets: list[list[tuple[str, int]]],
        patch_size: int,
    ) -> MaskedOlmoEarthSample:
        """Compute masks for each band set based on the encode and decode selections.

        The encoded and decoded bandsets are typically computed by the select_encoded_decoded_bandsets method.

        This method applies the bandset-level masking rules:
        - Bandsets not selected for encoding: ONLINE_ENCODER → TARGET_ENCODER_ONLY
        - Bandsets not selected for decoding: DECODER → TARGET_ENCODER_ONLY
        - Special modalities (e.g., static in time/space): force all to ONLINE_ENCODER or DECODER

        Args:
            masked_batch: The masked batch to apply the mask to.
            encoded_decoded_bandsets: The encoded and decoded bandsets for each sample.
            present_modalities_bandsets: The present modalities and bandsets for each sample.
            patch_size: The patch size being applied

        Returns:
            The masked batch with the masks applied.
        """
        masked_batch_dict = masked_batch.as_dict(return_none=False)
        num_encoded: None | torch.Tensor = None
        num_decoded: None | torch.Tensor = None

        # Apply bandset rules for each modality
        for modality in masked_batch.modalities:
            if modality == "timestamps":
                continue
            out_mask, enc_count, dec_count = self._apply_bandset_rules_to_modality(
                modality,
                masked_batch,
                masked_batch_dict,
                encoded_decoded_bandsets,
                present_modalities_bandsets,
            )
            masked_modality_name = MaskedOlmoEarthSample.get_masked_modality_name(
                modality
            )
            masked_batch_dict[masked_modality_name] = out_mask
            # Tracking these to ensure no tokens are accumulated.
            num_encoded = self._accumulate_token_counts(num_encoded, enc_count)
            num_decoded = self._accumulate_token_counts(num_decoded, dec_count)

        # Handle edge cases where samples have no encoded or decoded tokens (only happens for some ablations)
        self._handle_no_tokens_edge_cases(
            masked_batch_dict, num_encoded, num_decoded, patch_size
        )

        masked_batch = MaskedOlmoEarthSample(**masked_batch_dict)
        return masked_batch

    def _apply_bandset_rules_to_modality(
        self,
        modality: str,
        masked_batch: MaskedOlmoEarthSample,
        masked_batch_dict: dict[str, Any],
        encoded_decoded_bandsets: list[
            tuple[set[tuple[str, int]], set[tuple[str, int]]]
        ],
        present_modalities_bandsets: list[list[tuple[str, int]]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply encode/decode bandset rules to all samples for this modality.

        Returns:
            Tuple of (out_modality_mask, encoded_count, decoded_count)
        """
        masked_modality_name = MaskedOlmoEarthSample.get_masked_modality_name(modality)
        modality_spec = Modality.get(modality)
        modality_mask = masked_batch_dict[masked_modality_name]
        # Clone to avoid aliasing errors (seen with variable patch sizes 1-12)
        out_modality_mask = modality_mask.clone()
        num_bandsets = modality_mask.shape[-1]
        batch_size = masked_batch.timestamps.shape[0]

        # Apply rules for each sample
        for sample_idx in range(batch_size):
            self._apply_bandset_mask_for_sample(
                sample_idx,
                modality,
                modality_spec,
                modality_mask,
                out_modality_mask,
                num_bandsets,
                encoded_decoded_bandsets[sample_idx],
                present_modalities_bandsets[sample_idx],
            )

        # Count how many tokens are encoded/decoded for this modality
        encoded_count, decoded_count = self._count_modality_tokens(out_modality_mask)
        return out_modality_mask, encoded_count, decoded_count

    def _apply_bandset_mask_for_sample(
        self,
        sample_idx: int,
        modality: str,
        modality_spec: ModalitySpec,
        modality_mask: torch.Tensor,
        out_modality_mask: torch.Tensor,
        num_bandsets: int,
        encoded_decoded: tuple[set[tuple[str, int]], set[tuple[str, int]]],
        present_bandsets: list[tuple[str, int]],
    ) -> None:
        """Apply bandset mask rules for a single sample."""
        encoded_bandset_idxs, decoded_bandset_idxs = encoded_decoded
        available_modalities = [mb[0] for mb in present_bandsets]

        if modality not in available_modalities:
            logger.debug(f"Modality {modality} not present for sample {sample_idx}")
            return

        # Apply rules for each bandset
        for bandset_idx in range(num_bandsets):
            self._apply_single_bandset_mask(
                sample_idx,
                bandset_idx,
                modality,
                modality_spec,
                modality_mask,
                out_modality_mask,
                encoded_bandset_idxs,
                decoded_bandset_idxs,
            )

    def _apply_single_bandset_mask(
        self,
        sample_idx: int,
        bandset_idx: int,
        modality: str,
        modality_spec: ModalitySpec,
        modality_mask: torch.Tensor,
        out_modality_mask: torch.Tensor,
        encoded_bandset_idxs: set[tuple[str, int]],
        decoded_bandset_idxs: set[tuple[str, int]],
    ) -> None:
        """Apply mask rules for a single bandset within a sample."""
        is_encoded = (modality, bandset_idx) in encoded_bandset_idxs
        is_decoded = (modality, bandset_idx) in decoded_bandset_idxs

        # Handle special modalities that need override (e.g., static in time/space)
        if self.overide_strategy_mask(modality_spec):
            self._force_override_mask(
                sample_idx,
                bandset_idx,
                is_encoded,
                is_decoded,
                modality,
                modality_mask,
                out_modality_mask,
            )
            return

        # Suppress ONLINE_ENCODER tokens for bandsets not selected for encoding
        if not is_encoded:
            self._suppress_unencoded_bandset(
                sample_idx, bandset_idx, modality_mask, out_modality_mask
            )
            return

        # Suppress DECODER tokens for bandsets not selected for decoding
        if not is_decoded:
            self._suppress_undecoded_bandset(
                sample_idx, bandset_idx, modality_mask, out_modality_mask
            )

    def _force_override_mask(
        self,
        sample_idx: int,
        bandset_idx: int,
        is_encoded: bool,
        is_decoded: bool,
        modality: str,
        modality_mask: torch.Tensor,
        out_modality_mask: torch.Tensor,
    ) -> None:
        """Force all non-missing tokens to ONLINE_ENCODER or DECODER for special modalities.

        Some modalities don't follow the base strategy structure (e.g., static in space
        is randomly masked in space masking). We force them to all ONLINE_ENCODER or DECODER
        to maintain the cross-modality structure.
        """
        if is_encoded:
            forced_mask_value = MaskValue.ONLINE_ENCODER.value
        elif is_decoded:
            forced_mask_value = MaskValue.DECODER.value
        else:
            return

        logger.debug(
            f"Setting {modality} bandset {bandset_idx} to {forced_mask_value}"
        )
        not_missing_mask = (
            modality_mask[sample_idx, ..., bandset_idx] != MaskValue.MISSING.value
        )
        out_modality_mask[sample_idx, ..., bandset_idx] = torch.where(
            not_missing_mask,
            forced_mask_value,
            modality_mask[sample_idx, ..., bandset_idx],
        )

    def _suppress_unencoded_bandset(
        self,
        sample_idx: int,
        bandset_idx: int,
        modality_mask: torch.Tensor,
        out_modality_mask: torch.Tensor,
    ) -> None:
        """Suppress all ONLINE_ENCODER tokens for bandsets not selected for encoding.

        Converts ONLINE_ENCODER → TARGET_ENCODER_ONLY so these tokens are only
        seen by the target encoder, not the online encoder.
        """
        online_encoder_mask = (
            modality_mask[sample_idx, ..., bandset_idx]
            == MaskValue.ONLINE_ENCODER.value
        )
        out_modality_mask[sample_idx, ..., bandset_idx] = torch.where(
            online_encoder_mask.clone(),
            MaskValue.TARGET_ENCODER_ONLY.value,
            modality_mask[sample_idx, ..., bandset_idx],
        )

    def _suppress_undecoded_bandset(
        self,
        sample_idx: int,
        bandset_idx: int,
        modality_mask: torch.Tensor,
        out_modality_mask: torch.Tensor,
    ) -> None:
        """Suppress all DECODER tokens for bandsets not selected for decoding.

        Converts DECODER → TARGET_ENCODER_ONLY so these tokens are only
        seen by the target encoder, not decoded by the online model.
        """
        decoder_mask = (
            modality_mask[sample_idx, ..., bandset_idx] == MaskValue.DECODER.value
        )
        out_modality_mask[sample_idx, ..., bandset_idx] = torch.where(
            decoder_mask,
            MaskValue.TARGET_ENCODER_ONLY.value,
            modality_mask[sample_idx, ..., bandset_idx],
        )

    def _count_modality_tokens(
        self, modality_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Count encoded and decoded tokens for a modality across all samples."""
        flat_mask = torch.flatten(modality_mask, start_dim=1)
        encoded_count = (flat_mask == MaskValue.ONLINE_ENCODER.value).sum(dim=-1)
        decoded_count = (flat_mask == MaskValue.DECODER.value).sum(dim=-1)
        return encoded_count, decoded_count

    def _accumulate_token_counts(
        self, current: torch.Tensor | None, new_count: torch.Tensor
    ) -> torch.Tensor:
        """Accumulate token counts across modalities."""
        if current is None:
            return new_count
        return current + new_count

    def _handle_no_tokens_edge_cases(
        self,
        masked_batch_dict: dict[str, Any],
        num_encoded: torch.Tensor | None,
        num_decoded: torch.Tensor | None,
        patch_size: int,
    ) -> None:
        """Handle samples that ended up with no encoded or decoded tokens.

        This is a rare edge case that happens in specific scenarios like S2-only
        ablations with small spatial dimensions (h, w). When it occurs, we fall back
        to random masking to ensure each sample has some tokens to encode/decode.

        Note: These loops should be entered very rarely in practice.
        """
        no_encoded_indices = torch.argwhere(num_encoded == 0)
        no_decoded_indices = torch.argwhere(num_decoded == 0)

        # Fix samples with no encoded tokens
        for i in no_encoded_indices:
            self._fix_sample_with_no_tokens(masked_batch_dict, i, patch_size)

        # Fix samples with no decoded tokens
        for i in no_decoded_indices:
            self._fix_sample_with_no_tokens(masked_batch_dict, i, patch_size)

    def _fix_sample_with_no_tokens(
        self, masked_batch_dict: dict[str, Any], sample_idx: torch.Tensor, patch_size: int
    ) -> None:
        """Fix a single sample that has no encoded or decoded tokens using random masking."""
        for key, val in masked_batch_dict.items():
            if not key.endswith("mask"):
                continue

            modality_name = MaskedOlmoEarthSample.get_unmasked_modality_name(key)
            if modality_name in self.only_decode_modalities:
                continue

            modality_mask = val[sample_idx]
            modality_spec = Modality.get(modality_name)
            masked_batch_dict[key][sample_idx] = self._random_fill_unmasked(
                modality_mask, modality_spec, patch_size
            )

    def apply_mask(
        self, batch: OlmoEarthSample, patch_size: int | None = None, **kwargs: Any
    ) -> MaskedOlmoEarthSample:
        """Apply space masking to the input data."""
        if patch_size is None:
            # this is because we use a random-masking proxy in case of
            # no encoded or decoded tokens.
            raise ValueError("patch_size must be provided for cross masking")

        masked_sample = self.strategy.apply_mask(batch, patch_size, **kwargs)
        present_modalities_bandsets = self.get_sample_present_modalities_bandsets(
            masked_sample
        )
        encoded_decoded_bandsets = self.select_encoded_decoded_bandsets(
            present_modalities_bandsets
        )
        masked_sample = self.apply_bandset_mask_rules(
            masked_sample,
            encoded_decoded_bandsets,
            present_modalities_bandsets,
            patch_size,
        )

        return masked_sample


@MASKING_STRATEGY_REGISTRY.register("modality_cross_space")
class ModalityCrossSpaceMaskingStrategy(ModalityCrossMaskingStrategy):
    """Randomly select a modality and apply space masking to it."""

    def __init__(
        self,
        encode_ratio: float = 0.5,
        decode_ratio: float = 0.5,
        allow_encoding_decoding_same_bandset: bool = False,
        min_encoded_bandsets: int = 2,
        max_encoded_bandsets: int | None = None,
        min_decoded_bandsets: int | None = None,
        max_decoded_bandsets: int | None = None,
        only_decode_modalities: list[str] = [],
    ) -> None:
        """Initialize the masking strategy."""
        space_strategy = SpaceMaskingStrategy(encode_ratio, decode_ratio)
        super().__init__(
            strategy=space_strategy,
            encode_ratio=encode_ratio,
            decode_ratio=decode_ratio,
            allow_encoding_decoding_same_bandset=allow_encoding_decoding_same_bandset,
            min_encoded_bandsets=min_encoded_bandsets,
            max_encoded_bandsets=max_encoded_bandsets,
            min_decoded_bandsets=min_decoded_bandsets,
            max_decoded_bandsets=max_decoded_bandsets,
            only_decode_modalities=only_decode_modalities,
        )

    def overide_strategy_mask(self, modality_spec: ModalitySpec) -> bool:
        """Overide the random mask  for the given modality by the encoding and decoding bandsets."""
        # For space masking non spatial data is randomly masked but we want to use the encoding and decoding bandsets
        # to determine the mask for the non spatial data
        return not modality_spec.is_spatial


@experimental(
    "This masking strategy is experimental and may not work with all combinations of modalities"
)
@MASKING_STRATEGY_REGISTRY.register("modality_cross_time")
class ModalityCrossTimeMaskingStrategy(ModalityCrossMaskingStrategy):
    """Randomly select a modality and apply time masking to it."""

    def __init__(
        self,
        encode_ratio: float = 0.5,
        decode_ratio: float = 0.5,
        allow_encoding_decoding_same_bandset: bool = False,
        min_encoded_bandsets: int = 2,
        max_encoded_bandsets: int | None = None,
        min_decoded_bandsets: int | None = None,
        max_decoded_bandsets: int | None = None,
        only_decode_modalities: list[str] = [],
    ) -> None:
        """Initialize the masking strategy."""
        space_strategy = SpaceMaskingStrategy(encode_ratio, decode_ratio)
        super().__init__(
            strategy=space_strategy,
            encode_ratio=encode_ratio,
            decode_ratio=decode_ratio,
            allow_encoding_decoding_same_bandset=allow_encoding_decoding_same_bandset,
            min_encoded_bandsets=min_encoded_bandsets,
            max_encoded_bandsets=max_encoded_bandsets,
            min_decoded_bandsets=min_decoded_bandsets,
            max_decoded_bandsets=max_decoded_bandsets,
            only_decode_modalities=only_decode_modalities,
        )

    def overide_strategy_mask(self, modality_spec: ModalitySpec) -> bool:
        """Overide the random mask  for the given modality by the encoding and decoding bandsets."""
        # For time masking static data is randomly masked but we want to use the encoding and decoding bandsets
        # to determine the mask for the static data
        return not modality_spec.is_spatial


@MASKING_STRATEGY_REGISTRY.register("modality_cross_space_time")
class ModalityCrossSpaceTimeMaskingStrategy(MaskingStrategy):
    """Randomly apply space cross modality masking and time cross modality masking."""

    def __init__(
        self,
        encode_ratio: float = 0.5,
        decode_ratio: float = 0.5,
        allow_encoding_decoding_same_bandset: bool = False,
        min_encoded_bandsets: int = 2,
        max_encoded_bandsets: int | None = None,
        min_decoded_bandsets: int | None = None,
        max_decoded_bandsets: int | None = None,
        only_decode_modalities: list[str] = [],
    ) -> None:
        """Initialize the masking strategy."""
        self._encode_ratio = encode_ratio
        self._decode_ratio = decode_ratio
        self.time_strategy = ModalityCrossTimeMaskingStrategy(
            encode_ratio,
            decode_ratio,
            allow_encoding_decoding_same_bandset=allow_encoding_decoding_same_bandset,
            min_encoded_bandsets=min_encoded_bandsets,
            max_encoded_bandsets=max_encoded_bandsets,
            min_decoded_bandsets=min_decoded_bandsets,
            max_decoded_bandsets=max_decoded_bandsets,
            only_decode_modalities=only_decode_modalities,
        )
        self.space_strategy = ModalityCrossSpaceMaskingStrategy(
            encode_ratio,
            decode_ratio,
            allow_encoding_decoding_same_bandset=allow_encoding_decoding_same_bandset,
            min_encoded_bandsets=min_encoded_bandsets,
            max_encoded_bandsets=max_encoded_bandsets,
            min_decoded_bandsets=min_decoded_bandsets,
            max_decoded_bandsets=max_decoded_bandsets,
            only_decode_modalities=only_decode_modalities,
        )
        self.generator = np.random.default_rng(0)

    def apply_mask(
        self, batch: OlmoEarthSample, patch_size: int | None = None, **kwargs: Any
    ) -> MaskedOlmoEarthSample:
        """Apply space and time cross modality masking to the input data."""
        has_enough_timesteps = batch.valid_time >= 3
        if (self.generator.random() < 0.5) or (not has_enough_timesteps):
            logger.info("Applying space masking")
            return self.space_strategy.apply_mask(batch, patch_size, **kwargs)
        else:
            logger.info("Applying time masking")
            return self.time_strategy.apply_mask(batch, patch_size, **kwargs)


@MASKING_STRATEGY_REGISTRY.register("random")
class RandomMaskingStrategy(MaskingStrategy):
    """Randomly masks the input data."""

    def __init__(
        self,
        encode_ratio: float = 0.5,
        decode_ratio: float = 0.5,
    ) -> None:
        """Initialize the masking strategy."""
        self._encode_ratio = encode_ratio
        self._decode_ratio = decode_ratio

    def apply_mask(
        self, batch: OlmoEarthSample, patch_size: int | None = None, **kwargs: Any
    ) -> MaskedOlmoEarthSample:
        """Apply random masking to the input data.

        All Masking happens in unpatchified form and not grouped across bandsets
        as the modality data is unpatchified and not grouped across bandsets

        The mask created for the space-time varying modality will be different than
        for the static modality.

        For space-time varying data, we will mask out the same ratio of values for
        all the instances in the batch. However, since a static modality might have
        very few tokens in a batch (e.g. 1 for latlons) instead we mask out a certain
        ratios of values across the entire batch.

        Args:
            batch: Input data of type OlmoEarthSample
            patch_size: patch size applied to sample
            **kwargs: Additional arguments for maskings

        Returns:
            MaskedOlmoEarthSample containing the masked data and mask
        """
        if patch_size is None:
            raise ValueError("patch_size must be provided for random masking")
        output_dict: dict[str, ArrayTensor | None] = {}
        for modality_name in batch.modalities:
            instance = getattr(batch, modality_name)
            if instance is None:
                # set instance and mask to None
                output_dict[modality_name] = None
                output_dict[
                    MaskedOlmoEarthSample.get_masked_modality_name(modality_name)
                ] = None
            else:
                if modality_name == "timestamps":
                    output_dict[modality_name] = instance
                    continue

                if isinstance(instance, torch.Tensor):
                    device: torch.device | None = instance.device
                else:
                    device = None
                modality = Modality.get(modality_name)
                mask = self._create_random_mask(
                    modality, instance.shape, patch_size, device
                )
                mask = self.fill_mask_with_missing_values(instance, mask, modality)
                output_dict[modality_name] = instance
                output_dict[
                    MaskedOlmoEarthSample.get_masked_modality_name(modality_name)
                ] = mask
        return MaskedOlmoEarthSample(**output_dict)


@MASKING_STRATEGY_REGISTRY.register("modality_cross_random")
class ModalityCrossRandomMaskingStrategy(ModalityCrossMaskingStrategy):
    """Randomly select a modality and apply random masking to it."""

    def __init__(
        self,
        encode_ratio: float = 0.5,
        decode_ratio: float = 0.5,
        allow_encoding_decoding_same_bandset: bool = False,
        min_encoded_bandsets: int = 2,
        max_encoded_bandsets: int | None = None,
        min_decoded_bandsets: int | None = None,
        max_decoded_bandsets: int | None = None,
        only_decode_modalities: list[str] = [],
    ) -> None:
        """Initialize the masking strategy."""
        random_strategy = RandomMaskingStrategy(encode_ratio, decode_ratio)
        super().__init__(
            strategy=random_strategy,
            encode_ratio=encode_ratio,
            decode_ratio=decode_ratio,
            allow_encoding_decoding_same_bandset=allow_encoding_decoding_same_bandset,
            min_encoded_bandsets=min_encoded_bandsets,
            max_encoded_bandsets=max_encoded_bandsets,
            min_decoded_bandsets=min_decoded_bandsets,
            max_decoded_bandsets=max_decoded_bandsets,
            only_decode_modalities=only_decode_modalities,
        )


@MASKING_STRATEGY_REGISTRY.register("random_increasing")
class RandomIncreasingMaskingStrategy(RandomMaskingStrategy):
    """Gradually increase the masked tokens (reduce encode ratio)."""

    def __init__(
        self,
        initial_encode_ratio: float = 0.5,
        final_encode_ratio: float = 0.1,
        initial_decode_ratio: float = 0.5,
        final_decode_ratio: float = 0.9,
        steps: int = 1000,
    ) -> None:
        """Initialize the masking strategy."""
        super().__init__(initial_encode_ratio, initial_decode_ratio)
        self.initial_encode_ratio = initial_encode_ratio
        self.final_encode_ratio = final_encode_ratio
        self.initial_decode_ratio = initial_decode_ratio
        self.final_decode_ratio = final_decode_ratio
        self.steps = steps
        self.elapsed = 0

    def apply_mask(
        self, batch: OlmoEarthSample, patch_size: int | None = None, **kwargs: Any
    ) -> MaskedOlmoEarthSample:
        """Apply masking while changing the encode and decode ratio over time."""
        self.elapsed += 1
        if self.elapsed >= self.steps:
            self._encode_ratio = self.final_encode_ratio
            self._decode_ratio = self.final_decode_ratio
        else:
            factor = self.elapsed / self.steps
            self._encode_ratio = (
                self.initial_encode_ratio
                + (self.final_encode_ratio - self.initial_encode_ratio) * factor
            )
            self._decode_ratio = (
                self.initial_decode_ratio
                + (self.final_decode_ratio - self.initial_decode_ratio) * factor
            )
        return super().apply_mask(batch, patch_size, **kwargs)


@MASKING_STRATEGY_REGISTRY.register("random_range")
class RandomRangeMaskingStrategy(MaskingStrategy):
    """Randomly masks the input data."""

    def __init__(
        self,
        min_encode_ratio: float = 0.1,
        max_encode_ratio: float = 0.5,
        min_decode_ratio: float | None = None,
        max_decode_ratio: float | None = None,
    ) -> None:
        """Initialize the masking strategy.

        Args:
            min_encode_ratio: lower bound of range to sample encode ratio.
            max_encode_ratio: upper bound of range to sample encode ratio.
            min_decode_ratio: lower bound of range to sample decode ratio. If None, the
                decode ratio is 1 - (sampled encode ratio).
            max_decode_ratio: upper bound of range to sample decode ratio.
        """
        self.min_encode_ratio = min_encode_ratio
        self.max_encode_ratio = max_encode_ratio
        self.min_decode_ratio = min_decode_ratio
        self.max_decode_ratio = max_decode_ratio
        self._encode_ratio = (min_encode_ratio + max_encode_ratio) / 2

        if min_decode_ratio is not None and max_decode_ratio is not None:
            self._decode_ratio = (min_decode_ratio + max_decode_ratio) / 2
        elif min_decode_ratio is not None or max_decode_ratio is not None:
            raise ValueError(
                "min_decode_ratio and max_decode_ratio must be both None or both not None"
            )
        else:
            self._decode_ratio = 1 - self._encode_ratio

        self.generator = np.random.default_rng(0)

    def apply_mask(
        self, batch: OlmoEarthSample, patch_size: int | None = None, **kwargs: Any
    ) -> MaskedOlmoEarthSample:
        """Apply random masking to the input data.

        All Masking happens in unpatchified form and not grouped across bandsets
        as the modality data is unpatchified and not grouped across bandsets

        The mask created for the space-time varying modality will be different than
        for the static modality.

        For space-time varying data, we will mask out the same ratio of values for
        all the instances in the batch. However, since a static modality might have
        very few tokens in a batch (e.g. 1 for latlons) instead we mask out a certain
        ratios of values across the entire batch.

        Args:
            batch: Input data of type OlmoEarthSample
            patch_size: patch size applied to sample
            **kwargs: Additional arguments for maskings

        Returns:
            MaskedOlmoEarthSample containing the masked data and mask
        """
        if patch_size is None:
            raise ValueError("patch_size must be provided for random masking")
        output_dict: dict[str, ArrayTensor | None] = {}
        for modality_name in batch.modalities:
            instance = getattr(batch, modality_name)
            if instance is None:
                # set instance and mask to None
                output_dict[modality_name] = None
                output_dict[
                    MaskedOlmoEarthSample.get_masked_modality_name(modality_name)
                ] = None
            else:
                if modality_name == "timestamps":
                    output_dict[modality_name] = instance
                    continue

                if isinstance(instance, torch.Tensor):
                    device: torch.device | None = instance.device
                else:
                    device = None

                modality = Modality.get(modality_name)

                if modality.is_spatial or modality.is_multitemporal:
                    # Create masks per element so that we can leverage _create_random_mask
                    # while also ensuring each example can have its own encode and decode
                    # ratios.
                    batch_size = instance.shape[0]
                    example_encode_ratios = self.generator.uniform(
                        self.min_encode_ratio, self.max_encode_ratio, (batch_size,)
                    )
                    if self.min_decode_ratio is not None:
                        example_decode_ratios = self.generator.uniform(
                            self.min_decode_ratio, self.max_decode_ratio, (batch_size,)
                        )
                    else:
                        example_decode_ratios = 1 - example_encode_ratios

                    example_masks = []
                    for batch_idx in range(batch_size):
                        example_masks.append(
                            self._create_random_mask(
                                modality,
                                instance[batch_idx : batch_idx + 1].shape,
                                patch_size,
                                device,
                                encode_ratio=example_encode_ratios[batch_idx],
                                decode_ratio=example_decode_ratios[batch_idx],
                            )
                        )
                    mask = torch.cat(example_masks, dim=0)

                else:
                    # For ones that could be single token we just pass the whole batch.
                    mask = self._create_random_mask(
                        modality, instance.shape, patch_size, device
                    )

                mask = self.fill_mask_with_missing_values(instance, mask, modality)
                output_dict[modality_name] = instance
                output_dict[
                    MaskedOlmoEarthSample.get_masked_modality_name(modality_name)
                ] = mask
        return MaskedOlmoEarthSample(**output_dict)


@MASKING_STRATEGY_REGISTRY.register("selectable_modality")
class SelectableModalityMaskingStrategy(MaskingStrategy):
    """Like modality masking but we mask some for decoding and others fully.

    Plus we also apply random masking for the remaining modalities.
    """

    def __init__(
        self,
        decodable_modalities: list[str],
        fully_mask_modalities: list[str],
        max_to_mask: int,
        encode_ratio: float = 0.5,
        decode_ratio: float = 0.5,
    ) -> None:
        """Initialize the masking strategy."""
        self.decodable_modalities = decodable_modalities
        self.fully_mask_modalities = fully_mask_modalities
        self.max_to_mask = max_to_mask
        self._encode_ratio = encode_ratio
        self._decode_ratio = decode_ratio
        self.generator = np.random.default_rng(0)
        self.random_strategy = RandomMaskingStrategy(encode_ratio, decode_ratio)

    def apply_mask(
        self, batch: OlmoEarthSample, patch_size: int | None = None, **kwargs: Any
    ) -> MaskedOlmoEarthSample:
        """Apply random masking, plus mask certain additional modalities."""
        # First apply random masking.
        masked_sample = self.random_strategy.apply_mask(batch, patch_size, **kwargs)

        # Choose additional modalities to mask entirely (either set DECODER or
        # MISSING).
        all_modalities = self.decodable_modalities + self.fully_mask_modalities
        modality_indices = np.arange(len(all_modalities))
        self.generator.shuffle(modality_indices)
        num_to_mask = self.generator.integers(self.max_to_mask + 1)
        cur_mask_modalities = [
            all_modalities[idx] for idx in modality_indices[0:num_to_mask]
        ]

        logger.debug("Decided to mask modalities: %s", cur_mask_modalities)
        for modality in cur_mask_modalities:
            if modality in self.decodable_modalities:
                value = MaskValue.DECODER.value
            else:
                value = MaskValue.MISSING.value
            logger.debug("Filling modality %s mask with %s", modality, value)
            getattr(
                masked_sample, MaskedOlmoEarthSample.get_masked_modality_name(modality)
            )[:] = value

        return masked_sample


@MASKING_STRATEGY_REGISTRY.register("selectable_random_range_modality")
class SelectableRandomRangeModalityMaskingStrategy(MaskingStrategy):
    """Like modality masking but we mask some for decoding and others fully.

    Plus we also apply random range masking for the remaining modalities.
    """

    def __init__(
        self,
        decodable_modalities: list[str],
        fully_mask_modalities: list[str],
        max_to_mask: int,
        min_encode_ratio: float = 0.1,
        max_encode_ratio: float = 0.5,
        min_decode_ratio: float | None = None,
        max_decode_ratio: float | None = None,
    ) -> None:
        """Initialize the masking strategy."""
        self.decodable_modalities = decodable_modalities
        self.fully_mask_modalities = fully_mask_modalities
        self.max_to_mask = max_to_mask
        self.generator = np.random.default_rng(0)
        self.random_strategy = RandomRangeMaskingStrategy(
            min_encode_ratio, max_encode_ratio, min_decode_ratio, max_decode_ratio
        )
        self._encode_ratio = self.random_strategy._encode_ratio
        self._decode_ratio = self.random_strategy._decode_ratio

    def apply_mask(
        self, batch: OlmoEarthSample, patch_size: int | None = None, **kwargs: Any
    ) -> MaskedOlmoEarthSample:
        """Apply random masking, plus mask certain additional modalities."""
        # First apply random range masking.
        masked_sample = self.random_strategy.apply_mask(batch, patch_size, **kwargs)

        # Decide how many and which modalities to mask per example.
        all_modalities = self.decodable_modalities + self.fully_mask_modalities
        batch_size = getattr(batch, all_modalities[0]).shape[0]

        for batch_idx in range(batch_size):
            # Choose additional modalities to mask entirely (either set DECODER or
            # MISSING).
            modality_indices = np.arange(len(all_modalities))
            self.generator.shuffle(modality_indices)
            num_to_mask = self.generator.integers(self.max_to_mask + 1)
            cur_mask_modalities = [
                all_modalities[idx] for idx in modality_indices[0:num_to_mask]
            ]

            for modality in cur_mask_modalities:
                if modality in self.decodable_modalities:
                    value = MaskValue.DECODER.value
                else:
                    value = MaskValue.MISSING.value
                getattr(
                    masked_sample,
                    MaskedOlmoEarthSample.get_masked_modality_name(modality),
                )[batch_idx] = value

        return masked_sample


class FixedModalityMaskingStrategy(MaskingStrategy):
    """Abstract class for masking strategies always mask certain modalities on top of another masking strategy."""

    def __init__(
        self,
        strategy: MaskingStrategy,
        decoded_modalities: list[str],
        randomize_missing_modalities: list[str] = [],
        encode_ratio: float = 0.5,
        decode_ratio: float = 0.5,
    ) -> None:
        """Initialize the masking strategy."""
        self._encode_ratio = encode_ratio
        self._decode_ratio = decode_ratio
        self.strategy = strategy
        self.decoded_modalities = decoded_modalities
        self.randomize_missing_modalities = randomize_missing_modalities
        self.generator = np.random.default_rng(0)

    def apply_mask(
        self, batch: OlmoEarthSample, patch_size: int | None = None, **kwargs: Any
    ) -> MaskedOlmoEarthSample:
        """Apply masking to the input data."""
        # Apply other strategy first.
        masked_sample = self.strategy.apply_mask(batch, patch_size, **kwargs)

        # Now mark the decoded_modalities for decoding, similar to SelectableModalityMaskingStrategy.
        for modality in self.decoded_modalities:
            mask = getattr(
                masked_sample, MaskedOlmoEarthSample.get_masked_modality_name(modality)
            )
            if mask is None:
                continue
            mask[:] = MaskValue.DECODER.value

        # Randomly decide whether to mark the randomize_missing_modalities as missing.
        # We do this on a per-instance basis since we want to make sure we don't mark
        # all the modalities for that instance missing.
        if len(self.randomize_missing_modalities) > 0:
            batch_size = getattr(batch, self.randomize_missing_modalities[0]).shape[0]
            for batch_idx in range(batch_size):
                cur_available_modalities = []
                for modality in self.randomize_missing_modalities:
                    mask = getattr(
                        masked_sample,
                        MaskedOlmoEarthSample.get_masked_modality_name(modality),
                    )
                    # We check it is available everywhere since if it is missing in
                    # some patches and we mask a different modality then we might end
                    # up with no data for that spatial patch.
                    is_available = torch.all(mask != MaskValue.MISSING.value)
                    if is_available:
                        cur_available_modalities.append(modality)

                if len(cur_available_modalities) <= 1:
                    continue

                # Pick a subset to actually mask. We leave at least one unmasked.
                modality_indices = np.arange(len(cur_available_modalities))
                self.generator.shuffle(modality_indices)
                num_to_mask = self.generator.integers(len(cur_available_modalities))
                cur_mask_modalities = [
                    cur_available_modalities[idx]
                    for idx in modality_indices[0:num_to_mask]
                ]

                for modality in cur_mask_modalities:
                    getattr(
                        masked_sample,
                        MaskedOlmoEarthSample.get_masked_modality_name(modality),
                    )[batch_idx] = MaskValue.MISSING.value

        return masked_sample


@MASKING_STRATEGY_REGISTRY.register("random_fixed_modality")
class RandomFixedModalityMaskingStrategy(FixedModalityMaskingStrategy):
    """Fixed modality masking + random masking."""

    def __init__(
        self,
        decoded_modalities: list[str],
        randomize_missing_modalities: list[str] = [],
        encode_ratio: float = 0.5,
        decode_ratio: float = 0.5,
    ) -> None:
        """Initialize the masking strategy."""
        super().__init__(
            strategy=RandomMaskingStrategy(encode_ratio, decode_ratio),
            decoded_modalities=decoded_modalities,
            randomize_missing_modalities=randomize_missing_modalities,
            encode_ratio=encode_ratio,
            decode_ratio=decode_ratio,
        )


@dataclass
class MaskingConfig(Config):
    """Configuration for masking strategies.

    Args:
        strategy_config: Masking strategy to use in the format of
        {
            "type": "random", # registry key
            # rest of init kwargs
        }
    """

    strategy_config: dict[str, Any]

    def build(self) -> MaskingStrategy:
        """Build a MaskingStrategy from the config."""
        mask_strategy_key = self.strategy_config.pop("type")
        return MASKING_STRATEGY_REGISTRY.get_class(mask_strategy_key)(
            **self.strategy_config
        )
