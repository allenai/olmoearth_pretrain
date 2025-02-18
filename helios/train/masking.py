"""Masking module."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, NamedTuple

import numpy as np
import torch
from class_registry import ClassRegistry
from einops import rearrange, repeat
from olmo_core.config import Config

from helios.data.constants import Modality
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


# SHould be return type of masking strategy
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
    sentinel2: ArrayTensor
    sentinel2_mask: ArrayTensor
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
        if self.sentinel2 is None:
            raise ValueError("Sentinel2 is not present in this sample")
        return self.sentinel2.shape[1]

    @property
    def width(self) -> int:
        """Get the width of the data."""
        if self.sentinel2 is None:
            raise ValueError("Sentinel2 is not present in this sample")
        return self.sentinel2.shape[2]

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


class MaskingStrategy(ABC):
    """Abstract base class for masking strategies."""

    @abstractmethod
    def apply_mask(self, batch: HeliosSample, **kwargs: Any) -> MaskedHeliosSample:
        """Apply masking to the input data.

        Args:
            batch: Input data of type HeliosSample
            **kwargs: Additional arguments for maskings

        Returns:
            Tuple of (masked_data, mask)
        """
        pass


MASKING_STRATEGY_REGISTRY = ClassRegistry[MaskingStrategy]()


# EXAMPLE
@MASKING_STRATEGY_REGISTRY.register("random")
class RandomMaskingStrategy(MaskingStrategy):
    """Randomly masks the input data."""

    def __init__(self) -> None:
        """Create a new RandomMaskingStrategy."""
        # Use fixed seed for reproducibility.
        self.generator = np.random.default_rng(0)

    def _create_mask_per_static_modality(
        self,
        b: int,
        encode_ratio: float,
        decode_ratio: float,
        num_band_sets: int,
        return_tensor_device: torch.device | None = None,
    ) -> ArrayTensor:
        num_tokens_per_instance = int(b * num_band_sets)
        num_encode_tokens = int(num_tokens_per_instance * encode_ratio)
        num_decode_tokens = int(num_tokens_per_instance * decode_ratio)
        num_target_encode_tokens = int(
            num_tokens_per_instance - (num_encode_tokens + num_decode_tokens)
        )

        # we do this as a numpy array to take advantage of
        # numpy's permuted function
        flat_mask_tokens = np.concatenate(
            (
                np.ones(num_target_encode_tokens, dtype=np.int_),
                np.ones(num_decode_tokens, dtype=np.int_) * 2,
                np.zeros(num_encode_tokens, dtype=np.int_),
            )
        )
        flat_mask_tokens = self.generator.permuted(flat_mask_tokens, axis=0)
        static_mask = rearrange(flat_mask_tokens, "(b t) -> b t", b=b, t=num_band_sets)
        if return_tensor_device:
            return torch.as_tensor(static_mask, device=return_tensor_device)
        else:
            return static_mask

    def _create_mask_per_space_time_modality(
        self,
        b: int,
        h: int,
        w: int,
        t: int,
        encode_ratio: float,
        decode_ratio: float,
        num_bands: int,
        return_tensor_device: torch.device | None = None,
    ) -> ArrayTensor:
        num_tokens_per_instance = int(h * w * t * num_bands)
        num_encode_tokens = int(num_tokens_per_instance * encode_ratio)
        num_decode_tokens = int(num_tokens_per_instance * decode_ratio)
        num_target_encode_tokens = int(
            num_tokens_per_instance - (num_encode_tokens + num_decode_tokens)
        )

        # we do this as a numpy array to take advantage of
        # numpy's permuted function
        flat_mask_tokens = np.concatenate(
            (
                np.ones(num_target_encode_tokens, dtype=np.int_),
                np.ones(num_decode_tokens, dtype=np.int_) * 2,
                np.zeros(num_encode_tokens, dtype=np.int_),
            )
        )
        b_flat_tokens = repeat(flat_mask_tokens, "t -> b t", b=b)
        b_flat_tokens = self.generator.permuted(b_flat_tokens, axis=1)
        space_time_mask = rearrange(
            b_flat_tokens,
            "b (h w t c) -> b h w t c",
            h=h,
            w=w,
            t=t,
            c=num_bands,
        )

        if return_tensor_device:
            return torch.as_tensor(space_time_mask, device=return_tensor_device)
        else:
            return space_time_mask

    # TODO: We should be able to do this agnostic of dimensionality
    def _create_mask_per_space_modality(
        self,
        b: int,
        h: int,
        w: int,
        encode_ratio: float,
        decode_ratio: float,
        num_bands: int,
        return_tensor_device: torch.device | None = None,
    ) -> ArrayTensor:
        """Create a mask for a space modality."""
        num_tokens_per_instance = int(h * w * num_bands)
        num_encode_tokens = int(num_tokens_per_instance * encode_ratio)
        num_decode_tokens = int(num_tokens_per_instance * decode_ratio)
        num_target_encode_tokens = int(
            num_tokens_per_instance - (num_encode_tokens + num_decode_tokens)
        )

        # we do this as a numpy array to take advantage of
        # numpy's permuted function
        flat_mask_tokens = np.concatenate(
            (
                np.ones(num_target_encode_tokens, dtype=np.int_),
                np.ones(num_decode_tokens, dtype=np.int_) * 2,
                np.zeros(num_encode_tokens, dtype=np.int_),
            )
        )
        b_flat_tokens = repeat(flat_mask_tokens, "t -> b t", b=b)
        b_flat_tokens = self.generator.permuted(b_flat_tokens, axis=1)
        space_time_mask = rearrange(
            b_flat_tokens,
            "b (h w c) -> b h w c",
            h=h,
            w=w,
            c=num_bands,
        )

        if return_tensor_device:
            return torch.as_tensor(space_time_mask, device=return_tensor_device)
        else:
            return space_time_mask

    def apply_mask(self, batch: HeliosSample, **kwargs: Any) -> MaskedHeliosSample:
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
            **kwargs: Additional arguments for maskings

        Returns:
            MaskedHeliosSample containing the masked data and mask
        """
        # should these not be kwargs but instead be explicitly
        encode_ratio: float = kwargs["encode_ratio"]
        decode_ratio: float = kwargs["decode_ratio"]

        output_dict: dict[str, ArrayTensor | None] = {}
        for modality_name in batch._fields:
            modality = getattr(batch, modality_name)
            if modality is None:
                # set modality and mask to None
                output_dict[modality_name] = None
                output_dict[
                    MaskedHeliosSample.get_masked_modality_name(modality_name)
                ] = None
            else:
                if modality_name == "timestamps":
                    output_dict[modality_name] = modality
                    continue

                if isinstance(modality, torch.Tensor):
                    return_device: torch.device | None = modality.device
                else:
                    return_device = None
                # TODO: Make this decions based on modlaity spec and all the varying dimesnions agnostic
                num_bands = Modality.get(modality_name).num_bands
                if modality.ndim == 4:
                    b, h, w, _ = modality.shape
                    mask = self._create_mask_per_space_modality(
                        b,
                        h,
                        w,
                        encode_ratio,
                        decode_ratio,
                        num_bands,
                        return_device,
                    )
                elif modality.ndim == 5:
                    b, h, w, t, _ = modality.shape
                    mask = self._create_mask_per_space_time_modality(
                        b,
                        h,
                        w,
                        t,
                        encode_ratio,
                        decode_ratio,
                        num_bands,
                        return_device,
                    )
                elif modality.ndim == 2:
                    b = modality.shape[0]
                    mask = self._create_mask_per_static_modality(
                        b,
                        encode_ratio,
                        decode_ratio,
                        num_bands,
                        return_device,
                    )
                else:
                    raise ValueError(f"Unsupported modality shape {modality.shape}")
                output_dict[modality_name] = modality
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
