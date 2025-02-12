"""Masking module."""

import logging
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, NamedTuple

import numpy as np
import torch
from class_registry import ClassRegistry
from einops import rearrange, repeat
from olmo_core.config import Config

from helios.data.dataset import HeliosSample
from helios.types import ArrayTensor

logger = logging.getLogger(__name__)


class MaskValue(Enum):
    """Masks can take 3 possible values.

    ONLINE_ENCODER: The token is seen by the online encoder
    TARGET_ENCODER_ONLY: The token is seen by the target encoder only
    DECODER_ONLY: The token is seen by the decoder only
    """

    ONLINE_ENCODER = 0
    TARGET_ENCODER_ONLY = 1
    DECODER_ONLY = 2
    MISSING = 3


# SHould be return type of masking strategy
class MaskedHeliosSample(NamedTuple):
    """A masked sample of the data from the Helios dataset.

    This is a namedtuple that contains the data for a single sample from the Helios dataset.
    latlon and timestamps are the same for all modalities.
    For each modality. we have an ArrayTensor named by modality, and a mask for each modality named by modality_mask.
    we also have a mask for the latlon called latlon_mask

    Args:
        s2: ArrayTensor  # [B, H, W, T, len(S2_bands)]
        s2_mask: ArrayTensor  # [B, H, W, T, len(S2_band_groups)]
        latlon: ArrayTensor  # [B, 2]
        latlon_mask: ArrayTensor  # [B, len(latlon_band_groups)]
        timestamps: ArrayTensor  # [B, T, D=3], where D=[day, month, year]
    """

    s2: ArrayTensor
    s2_mask: ArrayTensor
    latlon: ArrayTensor  # [B, 2]
    latlon_mask: ArrayTensor
    timestamps: (
        ArrayTensor  # [B, T, D=3], where D=[day, month, year] (months are zero indexed)
    )

    def as_dict(self) -> dict[str, Any]:
        """Convert the namedtuple to a dictionary.

        Returns:
            Dictionary representation of the namedtuple.
        """
        return_dict = {}
        for field in self._fields:
            val = getattr(self, field)
            return_dict[field] = val
        return return_dict

    @property
    def height(self) -> int:
        """Get the height of the data."""
        return self.s2.shape[1]

    @property
    def width(self) -> int:
        """Get the width of the data."""
        return self.s2.shape[2]

    @property
    def time(self) -> int:
        """Get the number of time steps in the data."""
        return self.timestamps.shape[2]

    @staticmethod
    def get_masked_modality_name(modality: str) -> str:
        """Get the masked modality name."""
        return f"{modality}_mask"

    @classmethod
    def from_heliossample(
        cls,
        sample: HeliosSample,
        modalities_to_channel_groups_dict: dict[str, dict[str, list[int]]],
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
                    masked_sample_dict[key] = torch.empty(sample.shape(key))
                    masked_sample_dict[f"{key}_mask"] = (
                        torch.ones(
                            sample.shape(
                                key, len(modalities_to_channel_groups_dict[key])
                            )
                        )
                        * MaskValue.MISSING.value
                    )
                else:
                    masked_sample_dict[key] = t
                    masked_sample_dict[f"{key}_mask"] = (
                        torch.ones(
                            sample.shape(
                                key, len(modalities_to_channel_groups_dict[key])
                            )
                        )
                        * MaskValue.ONLINE_ENCODER.value
                    )

        return MaskedHeliosSample(**masked_sample_dict)


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

    @staticmethod
    def _create_mask_per_static_modality(
        b: int,
        encode_ratio: float,
        decode_ratio: float,
        channel_groups_dict: dict[str, list[int]],
        return_tensor_device: torch.device | None = None,
    ) -> ArrayTensor:
        num_tokens_per_instance = int(b * len(channel_groups_dict))
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
        # hopefully this will allow for reproducibility, since random is seeded
        rng = np.random.default_rng(random.randint(0, 100))
        flat_mask_tokens = rng.permuted(flat_mask_tokens, axis=0)
        static_mask = rearrange(
            flat_mask_tokens, "(b t) -> b t", b=b, t=len(channel_groups_dict)
        )
        if return_tensor_device:
            return torch.from_numpy(static_mask).to(return_tensor_device)
        else:
            return static_mask

    @staticmethod
    def _create_mask_per_space_time_modality(
        b: int,
        h: int,
        w: int,
        t: int,
        encode_ratio: float,
        decode_ratio: float,
        patch_size: int,
        channel_groups_dict: dict[str, list[int]],
        return_tensor_device: torch.device | None = None,
    ) -> ArrayTensor:
        h_p, w_p = int(h / patch_size), int(w / patch_size)
        num_tokens_per_instance = int(h_p * w_p * t * len(channel_groups_dict))
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
        # hopefully this will allow for reproducibility, since random is seeded
        rng = np.random.default_rng(random.randint(0, 100))
        b_flat_tokens = rng.permuted(b_flat_tokens, axis=1)
        b_flat_tokens = rearrange(
            b_flat_tokens,
            "b (h w t c) -> b h w t c",
            h=h_p,
            w=w_p,
            t=t,
            c=len(channel_groups_dict),
        )
        space_time_mask = np.repeat(
            np.repeat(b_flat_tokens, repeats=patch_size, axis=1),
            repeats=patch_size,
            axis=2,
        )
        if return_tensor_device:
            return torch.from_numpy(space_time_mask).to(return_tensor_device)
        else:
            return space_time_mask

    def apply_mask(self, batch: HeliosSample, **kwargs: Any) -> MaskedHeliosSample:
        """Apply random masking to the input data.

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
        # in the function signature?
        patch_size: int = kwargs["patch_size"]
        # TODO: this should be shared across train module
        modalities_to_channel_groups_dict: dict[str, dict[str, list[int]]] = kwargs[
            "modalities_to_channel_groups_dict"
        ]
        encode_ratio: float = kwargs["encode_ratio"]
        decode_ratio: float = kwargs["decode_ratio"]

        if (batch.h % patch_size != 0) or (batch.w % patch_size != 0):
            raise ValueError(
                f"h {batch.h} or w {batch.w} not divisible by patch size {patch_size}"
            )

        output_dict = {}
        for modality_name in batch._fields:
            if modality_name == "latlon":
                continue
            modality = getattr(batch, modality_name)
            if modality_name == "timestamps":
                output_dict[modality_name] = modality
                continue

            if isinstance(modality, torch.Tensor):
                return_device: torch.device | None = modality.device
            else:
                return_device = None
            if len(modality.shape) == 5:
                b, _, t, h, w = modality.shape

                mask = self._create_mask_per_space_time_modality(
                    b,
                    h,
                    w,
                    t,
                    encode_ratio,
                    decode_ratio,
                    patch_size,
                    modalities_to_channel_groups_dict[modality_name],
                    return_device,
                )
            elif len(modality.shape) == 2:
                b = modality.shape[0]
                mask = self._create_mask_per_static_modality(
                    b,
                    encode_ratio,
                    decode_ratio,
                    modalities_to_channel_groups_dict[modality_name],
                    return_device,
                )
            else:
                raise ValueError(f"Unsupported modality shape {modality.shape}")
            modality = rearrange(modality, "b c t h w -> b h w t c")
            output_dict[modality_name] = modality
            # TODO:Channels for mask are already in channel groups but not for tokens
            output_dict[f"{modality_name}_mask"] = mask
            logger.info(
                f" After maskingModality: {modality_name} shape: {modality.shape} mask shape: {mask.shape}"
            )

        # TODO: Temporary internal hack for not dealing with lat lons yet
        output_dict["latlon"] = None
        output_dict["latlon_mask"] = None
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
