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

from helios.data.constants import MISSING_VALUE, Modality, ModalitySpec
from helios.data.dataset import HeliosSample
from helios.types import ArrayTensor

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


class MaskedHeliosSample(NamedTuple):
    """A masked sample of the data from the Helios dataset.

    We always require sentinel2 data.
    This is a namedtuple that contains the data for a single sample from the Helios dataset.
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

    def unmask(self) -> "MaskedHeliosSample":
        """Return an unmasked MaskedHelioSample.

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
        return MaskedHeliosSample(**return_dict)

    @property
    def modalities(self) -> list[str]:
        """Get the present modalities in this instance of MaskedHeliosSample."""
        return [
            field
            for field in self._fields
            if not field.endswith("_mask")
            and field != "timestamps"
            and getattr(self, field) is not None
        ]

    @property
    def height(self) -> int:
        """Get the height of the data."""
        for modality in self.modalities:
            if modality == "timestamps":
                continue
            modality_spec = Modality.get(modality)
            if not modality_spec.is_spatial:
                continue
            x = getattr(self, modality)
            if x is not None:
                if len(x.shape) == 5:
                    return x.shape[1]
                else:
                    # no batch dimension
                    if len(x.shape) != 4:
                        raise ValueError(f"Unexpected shape {x.shape} for {modality}")
                    return x.shape[0]
        raise ValueError("No modality with height or width present")

    @property
    def width(self) -> int:
        """Get the height of the data."""
        for modality in self.modalities:
            if modality == "timestamps":
                continue
            modality_spec = Modality.get(modality)
            if not modality_spec.is_spatial:
                continue
            x = getattr(self, modality)
            if x is not None:
                if len(x.shape) == 5:
                    return x.shape[2]
                else:
                    # no batch dimension
                    if len(x.shape) != 4:
                        raise ValueError(f"Unexpected shape {x.shape} for {modality}")
                    return x.shape[1]
        raise ValueError("No modality with height or width present")

    @property
    def time(self) -> int:
        """Get the number of time steps in the data."""
        return self.timestamps.shape[1]

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
    def from_heliossample(
        cls,
        sample: HeliosSample,
    ) -> "MaskedHeliosSample":
        """Transforms a HelioSample into a MaskedHeliosSample.

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
                        MaskedHeliosSample.get_masked_modality_name(key)
                    ] = None
                else:
                    masked_sample_dict[key] = t
                    masked_sample_dict[
                        MaskedHeliosSample.get_masked_modality_name(key)
                    ] = (
                        torch.ones(sample.shape(key, mask=False))
                        * MaskValue.ONLINE_ENCODER.value
                    )

        return MaskedHeliosSample(**masked_sample_dict)

    @classmethod
    def from_dict(cls, dict: dict[str, Any]) -> "MaskedHeliosSample":
        """Create a MaskedHeliosSample from a dictionary, creating empty tensors for missing modalities.

        Args:
            dict: Dictionary representation of the MaskedHeliosSample.
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
        self, batch: HeliosSample, patch_size: int | None = None, **kwargs: Any
    ) -> MaskedHeliosSample:
        """Apply masking to the input data.

        Args:
            batch: Input data of type HeliosSample
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
        patch_size: int,
        device: torch.device | None = None,
        encode_ratio: float | None = None,
        decode_ratio: float | None = None,
    ) -> ArrayTensor:
        mask_shape = list(shape)
        mask_shape[-1] = modality.num_band_sets
        if modality.is_spatial:
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
        device: torch.device | None = None,
    ) -> ArrayTensor:
        b = shape[0]
        t = shape[-2]
        assert t >= 3
        encode_times = max(int(self.encode_ratio * t), 1)
        decode_times = max(int(self.decode_ratio * t), 1)
        target_times = t - encode_times - decode_times

        flat_mask = torch.cat(
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

        # numpy to for permuted function
        masks = [flat_mask[torch.randperm(t, device=device)] for i in range(b)]
        mask = torch.stack(masks)
        return mask

    def apply_mask(
        self, batch: HeliosSample, patch_size: int | None = None, **kwargs: Any
    ) -> MaskedHeliosSample:
        """Apply random masking to the input data.

        Masking happens temporally, with whole time steps having the same mask. Non-temporal data is randomly masked.

        Args:
            batch: Input data of type HeliosSample
            patch_size: patch size applied to sample
            **kwargs: Additional arguments for maskings

        Returns:
            MaskedHeliosSample containing the masked data and mask
        """
        if patch_size is None:
            raise ValueError("patch_size must be provided for time masking")
        output_dict: dict[str, ArrayTensor | None] = {}
        temporal_mask = None
        for modality_name in batch.modalities:
            instance = getattr(batch, modality_name)
            if instance is None:
                # set instance and mask to None
                output_dict[modality_name] = None
                output_dict[
                    MaskedHeliosSample.get_masked_modality_name(modality_name)
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
                if not modality.is_multitemporal:
                    mask = self._create_random_mask(modality, shape, patch_size, device)
                else:
                    if temporal_mask is None:
                        temporal_mask = self._create_temporal_mask(shape, device)
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
                    MaskedHeliosSample.get_masked_modality_name(modality_name)
                ] = mask
        return MaskedHeliosSample(**output_dict)


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

    def _create_spatial_mask(
        self,
        modality: ModalitySpec,
        shape: torch.Size,
        patch_size: int,
        device: torch.device | None = None,
    ) -> ArrayTensor:
        if not modality.is_spatial:
            raise ValueError("Non-spatial modality {modality}")

        b, h, w = shape[:3]

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
        patch_mask = rearrange(random_batch_mask, "b (h w) -> b h w", h=h_p, w=w_p)

        mask = repeat(
            patch_mask, "b h w -> b (h hp) (w wp)", hp=patch_size, wp=patch_size
        )
        return mask

    def apply_mask(
        self, batch: HeliosSample, patch_size: int | None = None, **kwargs: Any
    ) -> MaskedHeliosSample:
        """Apply random masking to the input data.

        Masking happens in patchified form, with whole patches having the same mask. Non-spatial data is randomly masked.

        Args:
            batch: Input data of type HeliosSample
            patch_size: patch size applied to sample
            **kwargs: Additional arguments for maskings

        Returns:
            MaskedHeliosSample containing the masked data and mask
        """
        if patch_size is None:
            raise ValueError("patch_size must be provided for space masking")
        output_dict: dict[str, ArrayTensor | None] = {}
        spatial_mask = None
        # Same spatial mask for all modalities
        for modality_name in batch.modalities:
            instance = getattr(batch, modality_name)
            if instance is None:
                # set instance and mask to None
                output_dict[modality_name] = None
                output_dict[
                    MaskedHeliosSample.get_masked_modality_name(modality_name)
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
                if spatial_mask is None:
                    logger.info(f"Creating spatial mask for modality {modality.name}")
                    spatial_mask = self._create_spatial_mask(
                        modality, shape, patch_size, device
                    )

                if len(shape) == 5:
                    t = shape[-2]
                else:
                    t = 1
                b_s = modality.num_band_sets
                # Mask is a view of the spatial mask, so changes to mask will change spatial_mask
                mask = repeat(spatial_mask, "... -> ... t b_s", t=t, b_s=b_s)
                mask = mask.view(*shape[:-1], b_s).clone()
            mask = self.fill_mask_with_missing_values(instance, mask, modality)

            # Keep data as is
            output_dict[modality_name] = instance
            output_dict[MaskedHeliosSample.get_masked_modality_name(modality_name)] = (
                mask
            )
        return MaskedHeliosSample(**output_dict)


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
        self, batch: HeliosSample, patch_size: int | None = None, **kwargs: Any
    ) -> MaskedHeliosSample:
        """Randomly mask out modalities in the input data.

        Entire modalities (per instance) are assigned the same mask.

        Args:
            batch: Input data of type HeliosSample
            patch_size: Optional patch size for spatial masking strategies
            **kwargs: Additional arguments for maskings

        Returns:
            MaskedHeliosSample containing the masked data and mask
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
            output_dict[MaskedHeliosSample.get_masked_modality_name(modality_name)] = (
                mask
            )

        return MaskedHeliosSample(**output_dict)


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
        self, batch: HeliosSample, patch_size: int | None = None, **kwargs: Any
    ) -> MaskedHeliosSample:
        """Apply space or time masking to the input data."""
        has_enough_timesteps = batch.valid_time >= 3
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
        self, batch: HeliosSample, patch_size: int | None = None, **kwargs: Any
    ) -> MaskedHeliosSample:
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
        self, batch: HeliosSample, patch_size: int | None = None, **kwargs: Any
    ) -> MaskedHeliosSample:
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
    """Abstract class for masking strategies that select a seperate set of bandsets to encode and decode on top of another masking strategy."""

    def __init__(
        self,
        strategy: MaskingStrategy,
        encode_ratio: float = 0.5,
        decode_ratio: float = 0.5,
        allow_encoding_decoding_same_bandset: bool = False,
    ) -> None:
        """Initialize the masking strategy."""
        self._encode_ratio = encode_ratio
        self._decode_ratio = decode_ratio
        self.strategy = strategy
        self.allow_encoding_decoding_same_bandset = allow_encoding_decoding_same_bandset

    def get_sample_present_modalities_bandsets(
        self, batch: MaskedHeliosSample
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
            modality_mask_name = MaskedHeliosSample.get_masked_modality_name(modality)
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
                    present_modalities_bandsets[sample_idx].append(
                        (modality, bandset_idx)
                    )

        return present_modalities_bandsets

    def select_encoded_decoded_bandsets(
        self, present_modalities_bandsets: list[list[tuple[str, int]]]
    ) -> list[tuple[set[tuple[str, int]], set[tuple[str, int]]]]:
        """Select the encoded and decoded bandsets for each sample."""
        encoded_decoded_bandsets: list[
            tuple[set[tuple[str, int]], set[tuple[str, int]]]
        ] = []
        for sample_idx in range(len(present_modalities_bandsets)):
            present_modalities_bandsets_for_sample = present_modalities_bandsets[
                sample_idx
            ]
            # If there is only one modality, we only encode not decode
            if len(present_modalities_bandsets_for_sample) == 1:
                encoded_bandset_idxs = set(present_modalities_bandsets_for_sample)
                decoded_bandset_idxs = set()
            # If there are two modalities, we encode one and decode the other
            elif len(present_modalities_bandsets_for_sample) == 2:
                encoded_bandset_idxs = set([present_modalities_bandsets_for_sample[0]])
                decoded_bandset_idxs = set([present_modalities_bandsets_for_sample[1]])
            # If there are more than two modalities, we randomly select some to encode and the rest to decode
            else:
                num_encoded_bandsets = np.random.randint(
                    1, len(present_modalities_bandsets_for_sample)
                )
                encoded_idxs = np.random.choice(
                    len(present_modalities_bandsets_for_sample),
                    size=num_encoded_bandsets,
                    replace=False,
                )
                encoded_bandset_idxs = set(
                    [present_modalities_bandsets_for_sample[i] for i in encoded_idxs]
                )
                # If not allow overlapping bandsets, we make encoded and decoded bandsets disjoint
                # Otherwise we allow them to overlap
                if self.allow_encoding_decoding_same_bandset:
                    num_decoded_bandsets = np.random.randint(
                        1, len(present_modalities_bandsets_for_sample)
                    )
                    decoded_idxs = np.random.choice(
                        len(present_modalities_bandsets_for_sample),
                        size=num_decoded_bandsets,
                        replace=False,
                    )
                    decoded_bandset_idxs = set(
                        [
                            present_modalities_bandsets_for_sample[i]
                            for i in decoded_idxs
                        ]
                    )
                else:
                    decoded_bandset_idxs = set(
                        [
                            present_modalities_bandsets_for_sample[i]
                            for i in range(len(present_modalities_bandsets_for_sample))
                            if i not in encoded_idxs
                        ]
                    )

            encoded_decoded_bandsets.append(
                (encoded_bandset_idxs, decoded_bandset_idxs)
            )
        return encoded_decoded_bandsets

    def overide_random_mask_condition(self, modality_spec: ModalitySpec) -> bool:
        """Overide the random mask  for the given modality by the encoding and decoding bandsets."""
        # Defaults to not overiding anything that may be random masked
        return False

    def apply_bandset_mask_rules(
        self,
        masked_batch: MaskedHeliosSample,
        encoded_decoded_bandsets: list[
            tuple[set[tuple[str, int]], set[tuple[str, int]]]
        ],
    ) -> MaskedHeliosSample:
        """Allow encoding of encoded bandsets and decoding of decoded bandsets."""
        masked_batch_dict = masked_batch.as_dict(return_none=False)
        for modality in masked_batch.modalities:
            if modality == "timestamps":
                continue
            masked_modality_name = MaskedHeliosSample.get_masked_modality_name(modality)
            modality_spec = Modality.get(modality)
            modality_mask = masked_batch_dict[masked_modality_name]
            num_bandsets = modality_mask.shape[-1]

            for sample_idx in range(masked_batch.timestamps.shape[0]):
                encoded_bandset_idxs, decoded_bandset_idxs = encoded_decoded_bandsets[
                    sample_idx
                ]

                # Get the modalities that are present for the sample
                available_modalities = set([i[0] for i in encoded_bandset_idxs])
                available_modalities.update([i[0] for i in decoded_bandset_idxs])

                if modality not in available_modalities:
                    logger.debug(
                        f"Modality {modality} not present for sample {sample_idx}"
                    )
                    continue

                for bandset_idx in range(num_bandsets):
                    is_encoded = (modality, bandset_idx) in encoded_bandset_idxs
                    is_decoded = (modality, bandset_idx) in decoded_bandset_idxs

                    if self.overide_random_mask_condition(modality_spec):
                        if is_encoded:
                            forced_mask_value = MaskValue.ONLINE_ENCODER.value
                        elif is_decoded:
                            forced_mask_value = MaskValue.DECODER.value
                        else:
                            continue
                        logger.debug(
                            f"Setting {modality} bandset {bandset_idx} to {forced_mask_value}"
                        )
                        not_missing_mask = (
                            modality_mask[sample_idx, ..., bandset_idx]
                            != MaskValue.MISSING.value
                        )
                        modality_mask[sample_idx, ..., bandset_idx] = torch.where(
                            not_missing_mask,
                            forced_mask_value,
                            modality_mask[sample_idx, ..., bandset_idx],
                        )
                        continue

                    if not is_encoded:
                        # Supress all encoded values for a not encoded bandset
                        online_encoder_mask = (
                            modality_mask[sample_idx, ..., bandset_idx]
                            == MaskValue.ONLINE_ENCODER.value
                        )
                        modality_mask[sample_idx, ..., bandset_idx] = torch.where(
                            online_encoder_mask,
                            MaskValue.TARGET_ENCODER_ONLY.value,
                            modality_mask[sample_idx, ..., bandset_idx],
                        )
                        continue

                    if not is_decoded:
                        decoder_mask = (
                            modality_mask[sample_idx, ..., bandset_idx]
                            == MaskValue.DECODER.value
                        )
                        modality_mask[sample_idx, ..., bandset_idx] = torch.where(
                            decoder_mask,
                            MaskValue.TARGET_ENCODER_ONLY.value,
                            modality_mask[sample_idx, ..., bandset_idx],
                        )

            masked_batch_dict[masked_modality_name] = modality_mask

        masked_batch = MaskedHeliosSample(**masked_batch_dict)
        return masked_batch

    def apply_mask(
        self, batch: HeliosSample, patch_size: int | None = None, **kwargs: Any
    ) -> MaskedHeliosSample:
        """Apply space masking to the input data."""
        masked_sample = self.strategy.apply_mask(batch, patch_size, **kwargs)

        present_modalities_bandsets = self.get_sample_present_modalities_bandsets(
            masked_sample
        )
        encoded_decoded_bandsets = self.select_encoded_decoded_bandsets(
            present_modalities_bandsets
        )

        masked_sample = self.apply_bandset_mask_rules(
            masked_sample, encoded_decoded_bandsets
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
    ) -> None:
        """Initialize the masking strategy."""
        space_strategy = SpaceMaskingStrategy(encode_ratio, decode_ratio)
        super().__init__(
            strategy=space_strategy,
            encode_ratio=encode_ratio,
            decode_ratio=decode_ratio,
            allow_encoding_decoding_same_bandset=allow_encoding_decoding_same_bandset,
        )

    def overide_random_mask_condition(self, modality_spec: ModalitySpec) -> bool:
        """Overide the random mask  for the given modality by the encoding and decoding bandsets."""
        # For space masking non spatial data is randomly masked but we want to use the encoding and decoding bandsets
        # to determine the mask for the non spatial data
        return not modality_spec.is_spatial


@MASKING_STRATEGY_REGISTRY.register("modality_cross_time")
class ModalityCrossTimeMaskingStrategy(ModalityCrossMaskingStrategy):
    """Randomly select a modality and apply time masking to it."""

    def __init__(
        self,
        encode_ratio: float = 0.5,
        decode_ratio: float = 0.5,
        allow_encoding_decoding_same_bandset: bool = False,
    ) -> None:
        """Initialize the masking strategy."""
        space_strategy = SpaceMaskingStrategy(encode_ratio, decode_ratio)
        super().__init__(
            strategy=space_strategy,
            encode_ratio=encode_ratio,
            decode_ratio=decode_ratio,
            allow_encoding_decoding_same_bandset=allow_encoding_decoding_same_bandset,
        )

    def overide_random_mask_condition(self, modality_spec: ModalitySpec) -> bool:
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
    ) -> None:
        """Initialize the masking strategy."""
        self._encode_ratio = encode_ratio
        self._decode_ratio = decode_ratio
        self.time_strategy = ModalityCrossTimeMaskingStrategy(
            encode_ratio,
            decode_ratio,
            allow_encoding_decoding_same_bandset=allow_encoding_decoding_same_bandset,
        )
        self.space_strategy = ModalityCrossSpaceMaskingStrategy(
            encode_ratio,
            decode_ratio,
            allow_encoding_decoding_same_bandset=allow_encoding_decoding_same_bandset,
        )
        self.generator = np.random.default_rng(0)

    def apply_mask(
        self, batch: HeliosSample, patch_size: int | None = None, **kwargs: Any
    ) -> MaskedHeliosSample:
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
        self, batch: HeliosSample, patch_size: int | None = None, **kwargs: Any
    ) -> MaskedHeliosSample:
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
            batch: Input data of type HeliosSample
            patch_size: patch size applied to sample
            **kwargs: Additional arguments for maskings

        Returns:
            MaskedHeliosSample containing the masked data and mask
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
                    MaskedHeliosSample.get_masked_modality_name(modality_name)
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
                    MaskedHeliosSample.get_masked_modality_name(modality_name)
                ] = mask
        return MaskedHeliosSample(**output_dict)


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
        self, batch: HeliosSample, patch_size: int | None = None, **kwargs: Any
    ) -> MaskedHeliosSample:
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
        self, batch: HeliosSample, patch_size: int | None = None, **kwargs: Any
    ) -> MaskedHeliosSample:
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
            batch: Input data of type HeliosSample
            patch_size: patch size applied to sample
            **kwargs: Additional arguments for maskings

        Returns:
            MaskedHeliosSample containing the masked data and mask
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
                    MaskedHeliosSample.get_masked_modality_name(modality_name)
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
                    MaskedHeliosSample.get_masked_modality_name(modality_name)
                ] = mask
        return MaskedHeliosSample(**output_dict)


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
        self, batch: HeliosSample, patch_size: int | None = None, **kwargs: Any
    ) -> MaskedHeliosSample:
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
                masked_sample, MaskedHeliosSample.get_masked_modality_name(modality)
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
        self, batch: HeliosSample, patch_size: int | None = None, **kwargs: Any
    ) -> MaskedHeliosSample:
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
                    masked_sample, MaskedHeliosSample.get_masked_modality_name(modality)
                )[batch_idx] = value

        return masked_sample


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
