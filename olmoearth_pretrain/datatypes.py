"""Data structures for OlmoEarth Pretrain."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, fields, replace
from enum import Enum
from typing import TYPE_CHECKING, Any, TypeVar, cast

import numpy as np
import torch
from einops import rearrange
from torch import Tensor
from torch.distributed import DeviceMesh
from torch.distributed.tensor import distribute_tensor

from olmoearth_pretrain.data.constants import MISSING_VALUE, TIMESTAMPS, Modality
from olmoearth_pretrain.types import ArrayTensor

if TYPE_CHECKING:
    from olmoearth_pretrain.nn.pooling import PoolingType

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


# timestamps is never considered a "modality" - it's metadata about when samples were captured
TIMESTAMPS_FIELD = "timestamps"


# =============================================================================
# Mixins for shared methods
# =============================================================================


class ModalityMethodsMixin:
    """Shared methods for all modality containers."""

    def as_dict(self, include_nones: bool = False) -> dict[str, Any]:
        """Convert to a dictionary.

        Args:
            include_nones: Whether to include None values in the output.

        Returns:
            Dictionary representation.
        """
        result = {}
        for f in fields(self):  # type: ignore[arg-type]
            val = getattr(self, f.name)
            if include_nones or val is not None:
                result[f.name] = val
        return result

    @property
    def modalities(self) -> list[str]:
        """Get the present modalities (excludes masks and timestamps)."""
        return [
            f.name
            for f in fields(self)  # type: ignore[arg-type]
            if not f.name.endswith("_mask")
            and f.name != TIMESTAMPS_FIELD
            and getattr(self, f.name) is not None
        ]

    @property
    def modalities_with_timestamps(self) -> list[str]:
        """Get all modalities including timestamps if present (excludes masks)."""
        has_timestamps = False
        result = []
        for f in fields(self):  # type: ignore[arg-type]
            if f.name == TIMESTAMPS_FIELD:
                has_timestamps = True
            if not f.name.endswith("_mask") and getattr(self, f.name) is not None:
                result.append(f.name)
        if not has_timestamps:
            raise ValueError(f"{type(self).__name__} does not have a timestamps field")
        return result

    @property
    def batch_size(self) -> int:
        """Get the batch size of the data."""
        for f in fields(self):  # type: ignore[arg-type]
            val = getattr(self, f.name)
            if val is not None:
                return val.shape[0]
        raise ValueError("No data to get batch size from")


class MaskedModalityMethodsMixin(ModalityMethodsMixin):
    """Shared methods for masked modality containers (with _mask fields)."""

    @staticmethod
    def get_masked_modality_name(modality: str) -> str:
        """Get the masked modality name."""
        return f"{modality}_mask"

    @staticmethod
    def get_unmasked_modality_name(modality_mask_name: str) -> str:
        """Get the unmasked modality name."""
        return modality_mask_name.replace("_mask", "")


# TypeVar for self-returning methods
T = TypeVar("T", bound="MaskedOlmoEarthSample")
TT = TypeVar("TT", bound="TokensAndMasks")


@dataclass
class OlmoEarthSample(ModalityMethodsMixin):
    """A sample of the data from the OlmoEarth Pretrain dataset.

    This dataclass contains the data of a single sample or a batch of samples.
    For each modality, we have an ArrayTensor named by the modality,
    along with the latlon and timestamps.
    """

    # Modality fields
    sentinel2_l2a: ArrayTensor | None = None  # [B, H, W, T, len(S2_bands)]
    sentinel1: ArrayTensor | None = None  # [B, H, W, T, len(S1_bands)]
    worldcover: ArrayTensor | None = None  # [B, H, W, 1, len(WC_bands)]
    openstreetmap_raster: ArrayTensor | None = None  # [B, H, W, 1, len(OSM_bands)]
    srtm: ArrayTensor | None = None  # [B, H, W, 1, len(SRTM_bands)]
    landsat: ArrayTensor | None = None  # [B, H, W, T, len(LANDSAT_bands)]
    # naip with different tile resolution is currently not used in favor of naip_10.
    naip: ArrayTensor | None = None  # [B, H, W, T, len(NAIP_bands)]
    # naip_10 is currently 4x the height/width of sentinel2_l2a.
    naip_10: ArrayTensor | None = None  # [B, H, W, T, len(NAIP_bands)]
    gse: ArrayTensor | None = None  # [B, H, W, 1, len(GSE_bands)]
    cdl: ArrayTensor | None = None  # [B, H, W, 1, len(CDL_bands)]
    worldpop: ArrayTensor | None = None  # [B, H, W, 1, len(WORLDPOP_bands)]
    worldcereal: ArrayTensor | None = None  # [B, H, W, 1, len(CDL_bands)]
    wri_canopy_height_map: ArrayTensor | None = None  # [B, H, W, 1, 1]
    # era5_10 is not spatially varying, so it has no height/width dimensions.
    era5_10: ArrayTensor | None = None  # [B, T, len(ERA5_bands)]
    latlon: ArrayTensor | None = None  # [B, 2]
    timestamps: ArrayTensor | None = None  # [B, T, D=3], where D=[day, month, year]

    def shape(self, attribute: str, mask: bool = False) -> Sequence[int]:
        """Returns the expected shape of an attribute.

        This is useful if you want to know what the shape of a
        missing attribute would have been for this sample.

        Args:
            attribute: The attribute to get the shape of.
            mask: Whether to get the shape of the mask.

        Returns:
            The shape of the attribute.
        """
        if attribute == "timestamps":
            if not mask:
                if self.timestamps is None:
                    raise ValueError("Timestamps are not present in the sample")
                return self.timestamps.shape
            else:
                raise ValueError("Timestamps are not maskable")
        else:
            return self.get_expected_shape(attribute, mask)

    @staticmethod
    def num_bands(attribute: str) -> int:
        """Get the number of channels for a given attribute."""
        if attribute == "timestamps":
            return len(TIMESTAMPS)
        else:
            return Modality.get(attribute).num_bands

    def distribute_tensors(self, device_mesh: DeviceMesh) -> OlmoEarthSample:
        """Distribute the tensors to the specified device mesh."""
        updates = {
            key: distribute_tensor(val, device_mesh)
            for key, val in self.as_dict(include_nones=False).items()
        }
        return replace(self, **updates)

    @property
    def height(self) -> int:
        """Get the height of the data at resolution_factor == 16."""
        for modality in self.modalities:
            modality_spec = Modality.get(modality)
            if not modality_spec.is_spatial:
                continue
            x = getattr(self, modality)
            if x is not None:
                if len(x.shape) == 5:
                    return x.shape[1] // modality_spec.image_tile_size_factor
                else:
                    if len(x.shape) != 4:
                        raise ValueError(f"Unexpected shape {x.shape} for {modality}")
                    return x.shape[0] // modality_spec.image_tile_size_factor
        raise ValueError("No modality with height or width present")

    @property
    def width(self) -> int:
        """Get the width of the data at resolution_factor == 16."""
        for modality in self.modalities:
            modality_spec = Modality.get(modality)
            if not modality_spec.is_spatial:
                continue
            x = getattr(self, modality)
            if x is not None:
                if len(x.shape) == 5:
                    return x.shape[2] // modality_spec.image_tile_size_factor
                else:
                    if len(x.shape) != 4:
                        raise ValueError(f"Unexpected shape {x.shape} for {modality}")
                    return x.shape[1] // modality_spec.image_tile_size_factor
        raise ValueError("No modality with height or width present")

    @property
    def time(self) -> int:
        """Get the number of time steps in the data."""
        if self.timestamps is None:
            raise ValueError("Timestamps are not present in the sample")
        return self.timestamps.shape[-2]

    @property
    def valid_time(self) -> int:
        """Get the minimum number of valid time steps in a batch."""
        return self.timesteps_with_at_least_one_modality.shape[0]

    @property
    def timesteps_with_at_least_one_modality(self) -> torch.Tensor:
        """Get timesteps with at least one modality present."""
        per_modality_present_masks = []
        for modality in self.modalities:
            modality_spec = Modality.get(modality)
            if modality_spec.is_multitemporal:
                data = getattr(self, modality)
                if isinstance(data, np.ndarray):
                    raise ValueError(
                        "timesteps_with_at_least_one_modality is not yet supported for numpy arrays"
                    )
                present_mask = (data != MISSING_VALUE).all(dim=(0, 1, 2, 4))
                per_modality_present_masks.append(present_mask)
        at_least_one_modality_present_timestep_mask = torch.stack(
            per_modality_present_masks, dim=1
        ).any(dim=1)
        timesteps_with_at_least_one_modality = torch.where(
            at_least_one_modality_present_timestep_mask
        )[0]
        return timesteps_with_at_least_one_modality

    @staticmethod
    def compute_expected_shape(
        attribute: str,
        height: int | None,
        width: int | None,
        time: int,
        mask: bool = False,
    ) -> tuple[int, ...]:
        """Get expected shape for a modality given dimensions.

        Args:
            attribute: The modality name.
            height: Height in pixels (required for spatial modalities).
            width: Width in pixels (required for spatial modalities).
            time: Number of timesteps.
            mask: If True, use num_band_sets instead of num_bands.

        Returns:
            Expected shape tuple for the modality.
        """
        modality_spec = Modality.get(attribute)
        num_bands = modality_spec.num_band_sets if mask else modality_spec.num_bands

        if modality_spec.is_spacetime_varying:
            if height is None or width is None:
                raise ValueError(
                    f"height and width required for spatial modality {attribute}"
                )
            return (
                height * modality_spec.image_tile_size_factor,
                width * modality_spec.image_tile_size_factor,
                time,
                num_bands,
            )
        elif modality_spec.is_space_only_varying:
            if height is None or width is None:
                raise ValueError(
                    f"height and width required for spatial modality {attribute}"
                )
            return (
                height * modality_spec.image_tile_size_factor,
                width * modality_spec.image_tile_size_factor,
                1,
                num_bands,
            )
        elif modality_spec.is_time_only_varying:
            return (time, num_bands)
        else:
            return (num_bands,)

    def get_expected_shape(self, attribute: str, mask: bool = False) -> tuple[int, ...]:
        """Get expected shape of an attribute using this sample's dimensions."""
        return OlmoEarthSample.compute_expected_shape(
            attribute, self.height, self.width, self.time, mask
        )

    def scale(self, s: float) -> OlmoEarthSample:
        """Multiply a OlmoEarthSample by a float."""
        updates = {k: cast(ArrayTensor, v) * s for k, v in self.as_dict().items()}
        return replace(self, **updates)

    def add(
        self, other: OlmoEarthSample, timestamps_to_keep: ArrayTensor
    ) -> OlmoEarthSample:
        """Add two OlmoEarthSamples together."""
        if not isinstance(other, OlmoEarthSample):
            raise ValueError("Addition only supported for OlmoEarthSamples")
        summed_dict: dict[str, ArrayTensor] = {}
        for key, val in self.as_dict(include_nones=False).items():
            assert val is not None
            other_val = getattr(other, key)
            if other_val is None:
                raise ValueError(
                    f"Add requires both OlmoEarthSamples to have the same modalities, other is missing {key}"
                )
            summed_dict[key] = val + other_val
        summed_dict["timestamps"] = timestamps_to_keep
        return replace(self, **summed_dict)

    def rotate(self) -> OlmoEarthSample:
        """Rotate the instances by one.

        If previously, we had a batch of three instances [B1, B2, B3],
        we will now have a batch of three instances [B2, B3, B1].
        """
        output_dict: dict[str, ArrayTensor] = {}
        for key, v in self.as_dict().items():
            if isinstance(v, np.ndarray):
                output_dict[key] = np.concatenate((v[1:], v[:1]), axis=0)
            elif isinstance(v, torch.Tensor):
                output_dict[key] = torch.cat((v[1:], v[:1]), dim=0)
        return replace(self, **output_dict)


@dataclass
class MaskedOlmoEarthSample(MaskedModalityMethodsMixin):
    """A masked sample of the data from the OlmoEarth Pretrain dataset.

    This dataclass contains the data for a single sample from the OlmoEarth
    Pretrain dataset. For each modality we have an ArrayTensor named by modality,
    and a mask for each modality named by modality_mask.
    """

    # Modality fields with masks
    sentinel2_l2a: Tensor | None = None
    sentinel2_l2a_mask: Tensor | None = None
    sentinel1: Tensor | None = None
    sentinel1_mask: Tensor | None = None
    worldcover: Tensor | None = None
    worldcover_mask: Tensor | None = None
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
    latlon: Tensor | None = None
    latlon_mask: Tensor | None = None
    timestamps: ArrayTensor | None = (
        None  # [B, T, D=3], where D=[day, month, year] (months are zero indexed)
    )

    @property
    def batch_size(self) -> int:
        """Get the batch size of the sample.

        Returns:
            The batch size (first dimension of timestamps tensor).
        """
        if self.timestamps is not None:
            return self.timestamps.shape[0]
        for f in fields(self):
            val = getattr(self, f.name)
            if val is not None:
                return val.shape[0]
        raise ValueError("No data to get batch size from")

    def to_device(self: T, device: torch.device, non_blocking: bool = True) -> T:
        """Move all tensors to the specified device.

        Args:
            device: The device to move the tensors to.
            non_blocking: Whether or not to use asynchronous GPU copies

        Returns:
            A new instance with all tensors moved to the specified device.
        """
        updates = {}
        for f in fields(self):
            val = getattr(self, f.name)
            if val is not None and hasattr(val, "to"):
                updates[f.name] = val.to(device, non_blocking=non_blocking)
        return replace(self, **updates)

    def unmask(self) -> MaskedOlmoEarthSample:
        """Return an unmasked MaskedOlmoEarthSample.

        All mask values are MaskValue.ONLINE_ENCODER except for MaskValue.MISSING,
        which remain MISSING.
        """
        updates = {}
        for f in fields(self):
            if f.name.endswith("_mask"):
                val = getattr(self, f.name)
                if val is not None:
                    updates[f.name] = val * (val == MaskValue.MISSING.value)
        return replace(self, **updates)

    @classmethod
    def from_olmoearthsample(
        cls,
        sample: OlmoEarthSample,
    ) -> MaskedOlmoEarthSample:
        """Transforms a OlmoEarthSample into a MaskedOlmoEarthSample.

        This function assumes modalities are uniformly missing.
        """
        masked_sample_dict: dict[str, Any] = {}
        for key, t in sample.as_dict(include_nones=True).items():
            if key == "timestamps":
                masked_sample_dict[key] = t
            else:
                if t is None:
                    masked_sample_dict[key] = None
                    masked_sample_dict[cls.get_masked_modality_name(key)] = None
                else:
                    masked_sample_dict[key] = t
                    masked_sample_dict[cls.get_masked_modality_name(key)] = (
                        torch.ones(sample.shape(key, mask=False))
                        * MaskValue.ONLINE_ENCODER.value
                    )

        return MaskedOlmoEarthSample(**masked_sample_dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MaskedOlmoEarthSample:
        """Create a MaskedOlmoEarthSample from a dictionary.

        Args:
            d: Dictionary representation of the MaskedOlmoEarthSample.
        """
        return cls(**d)


@dataclass
class TokensAndMasks(MaskedModalityMethodsMixin):
    """Embedded tokens with masks for computing loss.

    This is the output format from the encoder, containing embedded tokens
    and their corresponding masks for each modality.

    Shapes:
        - modality: (B, P_H, P_W, T, Band_Sets, D)
        - modality_mask: (B, P_H, P_W, T, Band_Sets)
        - era5_10: (B, T, Band_Sets, D) — no spatial dims (not spatially varying)
        - latlon: (B, D) — no spatial or temporal dims
    """

    # Modality fields with masks (no timestamps)
    sentinel2_l2a: Tensor | None = None
    sentinel2_l2a_mask: Tensor | None = None
    sentinel1: Tensor | None = None
    sentinel1_mask: Tensor | None = None
    worldcover: Tensor | None = None
    worldcover_mask: Tensor | None = None
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
    latlon: Tensor | None = None
    latlon_mask: Tensor | None = None

    def to_device(self: TT, device: torch.device, non_blocking: bool = True) -> TT:
        """Move all tensors to the specified device.

        Args:
            device: The device to move the tensors to.
            non_blocking: Whether or not to use asynchronous GPU copies

        Returns:
            A new instance with all tensors moved to the specified device.
        """
        updates = {}
        for f in fields(self):
            val = getattr(self, f.name)
            if val is not None and hasattr(val, "to"):
                updates[f.name] = val.to(device, non_blocking=non_blocking)
        return replace(self, **updates)

    @property
    def device(self) -> torch.device:
        """Get the device of the tokens and masks."""
        for f in fields(self):
            val = getattr(self, f.name)
            if val is not None:
                return val.device
        raise ValueError("No data to get device from")

    def get_shape_dict(self) -> dict[str, tuple]:
        """Return a dictionary of the shapes of the fields."""
        return {
            f.name: getattr(self, f.name).shape
            for f in fields(self)
            if getattr(self, f.name) is not None
        }

    @staticmethod
    def _flatten(x: Tensor) -> Tensor:
        return rearrange(x, "b ... d -> b (...) d")

    def _flatten_per_modality(
        self,
    ) -> tuple[list[Tensor], list[Tensor]]:
        """Flatten tokens and masks per modality (not concatenated)."""
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
        flattened_masks = [mask[:, :, 0] for mask in flattened_masks]
        return flattened_x, flattened_masks

    def flatten_tokens_and_masks_per_modality(
        self,
    ) -> tuple[list[Tensor], list[Tensor]]:
        """Flatten tokens and masks, returning separate tensors per modality.

        Returns:
            Tuple of (tokens_list, masks_list) where each element
            corresponds to one modality.
        """
        return self._flatten_per_modality()

    def flatten_all_tokens_and_masks(self) -> tuple[Tensor, Tensor]:
        """Flatten and concatenate all tokens and masks across modalities.

        Returns:
            Tuple of (tokens, masks) concatenated across all modalities.
            Tokens will have shape [B, T, D] and masks will have shape [B, T].
        """
        flattened_x, flattened_masks = self._flatten_per_modality()
        x = torch.cat(flattened_x, dim=1)
        masks = torch.cat(flattened_masks, dim=1)
        return x, masks

    def pool_spatially_and_concat_modalities(self) -> Tensor:
        """Pool the modalities across time to get spatial features and concatenate."""
        from olmoearth_pretrain.nn.pooling import (
            pool_spatially_and_concat_modalities as _pool,
        )

        return _pool(self)

    def pool_spatially(self, pooling_type: PoolingType) -> Tensor:
        """Pool the modalities across time to get spatial features."""
        from olmoearth_pretrain.nn.pooling import pool_spatially as _pool

        return _pool(self, pooling_type)

    def pool_instance_wise(self, pooling_type: PoolingType) -> Tensor:
        """Pool all the tokens in the instance."""
        from olmoearth_pretrain.nn.pooling import pool_instance_wise as _pool

        return _pool(self, pooling_type)

    def pool_unmasked_tokens(
        self,
        pooling_type: PoolingType | None = None,
        spatial_pooling: bool = False,
        concat_features: bool = False,
    ) -> Tensor:
        """Pool the unmasked tokens.

        Args:
            pooling_type: Pooling type for the tokens. Defaults to MAX.
            spatial_pooling: Whether to keep the spatial dimensions when pooling.
            concat_features: Whether to concatenate the features instead of averaging.
        """
        from olmoearth_pretrain.nn.pooling import pool_unmasked_tokens as _pool

        return _pool(self, pooling_type, spatial_pooling, concat_features)
