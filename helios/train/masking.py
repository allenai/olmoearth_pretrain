"""Masking module."""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, NamedTuple

import numpy as np
import torch
from class_registry import ClassRegistry
from einops import rearrange, repeat
from olmo_core.config import Config

from helios.data.constants import Modality, ModalitySpec
from helios.data.dataset import HeliosSample
from helios.types import ArrayTensor

logger = logging.getLogger(__name__)


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
        height_width_time_modalities = ["sentinel2_l2a", "sentinel1", "worldcover"]
        for modality in height_width_time_modalities:
            x = getattr(self, modality)
            if x is not None:
                return x.shape[1]
        raise ValueError("No modality with height or width present")

    @property
    def width(self) -> int:
        """Get the width of the data."""
        height_width_time_modalities = ["sentinel2_l2a", "sentinel1", "worldcover"]
        for modality in height_width_time_modalities:
            x = getattr(self, modality)
            if x is not None:
                return x.shape[2]
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

    generator: np.random.Generator

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

    def _create_random_mask(
        self,
        modality: ModalitySpec,
        shape: torch.Size,
        patch_size: int,
        device: torch.device | None = None,
    ) -> ArrayTensor:
        mask_shape = list(shape)
        mask_shape[-1] = modality.num_band_sets
        if modality.is_spatial:
            mask_shape[1] //= patch_size
            mask_shape[2] //= patch_size

        if modality.is_spatial or modality.is_multitemporal:
            b = shape[0]
            num_tokens = np.prod(mask_shape[1:])
        else:
            num_tokens = np.prod(mask_shape[:-1])

        encode_tokens = int(num_tokens * self.encode_ratio)
        decode_tokens = int(num_tokens * self.decode_ratio)
        target_tokens = int(num_tokens - (encode_tokens + decode_tokens))

        # we do this as a numpy array to take advantage of
        # numpy's permuted function
        flat_mask_tokens = np.concatenate(
            (
                np.ones(target_tokens, dtype=np.int_)
                * MaskValue.TARGET_ENCODER_ONLY.value,
                np.ones(decode_tokens, dtype=np.int_) * MaskValue.DECODER.value,
                np.ones(encode_tokens, dtype=np.int_) * MaskValue.ONLINE_ENCODER.value,
            )
        )
        if modality.is_spatial or modality.is_multitemporal:
            flat_mask_tokens = repeat(flat_mask_tokens, "t -> b t", b=b)
            flat_mask_tokens = self.generator.permuted(flat_mask_tokens, axis=1)
        else:
            flat_mask_tokens = self.generator.permuted(flat_mask_tokens)

        mask = torch.as_tensor(flat_mask_tokens, device=device)
        mask = mask.view(*mask_shape)
        if modality.is_spatial:
            mask = repeat(
                mask, "b h w ... -> b (h hp) (w wp) ...", hp=patch_size, wp=patch_size
            )
        return mask

    @property
    def encode_ratio(self) -> float:
        """Get the encode ratio."""
        if not hasattr(self, "_encode_ratio"):
            raise AttributeError("Encode ratio not set")
        return self._encode_ratio

    @property
    def decode_ratio(self) -> float:
        """Get the decode ratio."""
        if not hasattr(self, "_decode_ratio"):
            raise AttributeError("Decode ratio not set")
        return self._decode_ratio


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
        self.generator = np.random.default_rng(0)

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

        flat_mask = np.concatenate(
            (
                np.ones(target_times, dtype=np.int_)
                * MaskValue.TARGET_ENCODER_ONLY.value,
                np.ones(decode_times, dtype=np.int_) * MaskValue.DECODER.value,
                np.ones(encode_times, dtype=np.int_) * MaskValue.ONLINE_ENCODER.value,
            )
        )

        # numpy to for permuted function
        mask = repeat(flat_mask, "t -> b t", b=b)
        mask = self.generator.permuted(mask, axis=1)
        mask = torch.as_tensor(mask, device=device)
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
        for modality_name in batch._fields:
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
                    mask = repeat(
                        temporal_mask, "b t -> b h w t b_s", h=h, w=w, b_s=b_s
                    )
                    mask = mask.view(*shape[:-1], b_s)

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
        self.generator = np.random.default_rng(0)

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

        flat_mask = np.concatenate(
            (
                np.ones(target_patches, dtype=np.int_)
                * MaskValue.TARGET_ENCODER_ONLY.value,
                np.ones(decode_patches, dtype=np.int_) * MaskValue.DECODER.value,
                np.ones(encode_patches, dtype=np.int_) * MaskValue.ONLINE_ENCODER.value,
            )
        )

        # numpy to for permuted function
        batch_mask = repeat(flat_mask, "x -> b x", b=b)
        random_batch_mask = self.generator.permuted(batch_mask, axis=1)
        patch_mask = rearrange(random_batch_mask, "b (h w) -> b h w", h=h_p, w=w_p)

        mask = np.repeat(patch_mask, repeats=patch_size, axis=1)
        mask = np.repeat(mask, repeats=patch_size, axis=2)
        mask = torch.as_tensor(mask, device=device)
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
        for modality_name in batch._fields:
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
                if not modality.is_spatial:
                    mask = self._create_random_mask(modality, shape, patch_size, device)
                else:
                    if spatial_mask is None:
                        spatial_mask = self._create_spatial_mask(
                            modality, shape, patch_size, device
                        )
                    if len(shape) == 5:
                        t = shape[-2]
                    else:
                        t = 1
                    b_s = modality.num_band_sets
                    mask = repeat(spatial_mask, "... -> ... t b_s", t=t, b_s=b_s)
                    mask = mask.view(*shape[:-1], b_s)

                output_dict[modality_name] = instance
                output_dict[
                    MaskedHeliosSample.get_masked_modality_name(modality_name)
                ] = mask
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
        self.generator = np.random.default_rng(0)

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
        if patch_size is not None:
            raise ValueError("patch_size must not be provided for modality masking")
        output_dict: dict[str, ArrayTensor | None] = {"timestamps": batch.timestamps}

        present_modalities = list(batch.as_dict(ignore_nones=True).keys())
        present_modalities = [b for b in present_modalities if b != "timestamps"]

        num_present_modalities = len(present_modalities)
        encode_modalities = max(1, int(self.encode_ratio * num_present_modalities))
        decode_modalities = max(1, int(self.decode_ratio * num_present_modalities))
        target_modalities = (
            num_present_modalities - encode_modalities - decode_modalities
        )

        band_mask_per_instance = np.concatenate(
            (
                np.ones(target_modalities, dtype=np.int_)
                * MaskValue.TARGET_ENCODER_ONLY.value,
                np.ones(decode_modalities, dtype=np.int_) * MaskValue.DECODER.value,
                np.ones(encode_modalities, dtype=np.int_)
                * MaskValue.ONLINE_ENCODER.value,
            )
        )
        batch_mask = repeat(band_mask_per_instance, "x -> b x", b=batch.batch_size)
        random_batch_mask = self.generator.permuted(batch_mask, axis=1)
        for idx, modality in enumerate(present_modalities):
            instance = getattr(batch, modality)
            output_dict[modality] = instance

            if isinstance(instance, torch.Tensor):
                device: torch.device | None = instance.device
            else:
                device = None

            modality_mask = torch.tensor(random_batch_mask[:, idx], device=device)
            shape = instance.shape
            b_s = shape[-1]
            b, h, w, t = list(shape[:-1]) + [1] * (4 - len(shape[:-1]))
            mask = repeat(modality_mask, "b -> b h w t b_s", h=h, w=w, b_s=b_s, t=t)
            mask = mask.view(*shape)
            output_dict[MaskedHeliosSample.get_masked_modality_name(modality)] = mask

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
        has_enough_timesteps = batch.time >= 3
        if (self.generator.random() < 0.5) or (not has_enough_timesteps):
            return self.space_strategy.apply_mask(batch, patch_size, **kwargs)
        else:
            return self.time_strategy.apply_mask(batch, patch_size, **kwargs)


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
        has_enough_timesteps = batch.time >= 3
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
        for modality_name in batch._fields:
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

                mask = self._create_random_mask(
                    Modality.get(modality_name), instance.shape, patch_size, device
                )

                output_dict[modality_name] = instance
                output_dict[
                    MaskedHeliosSample.get_masked_modality_name(modality_name)
                ] = mask
        return MaskedHeliosSample(**output_dict)


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
