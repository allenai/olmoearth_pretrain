"""Data structures for OlmoEarth Pretrain."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from enum import Enum
from typing import Any, NamedTuple, cast

import numpy as np
import torch
from einops import rearrange
from torch import Tensor
from torch.distributed import DeviceMesh
from torch.distributed.tensor import distribute_tensor

from olmoearth_pretrain.data.constants import MISSING_VALUE, TIMESTAMPS, Modality
from olmoearth_pretrain.types import ArrayTensor

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

# Multitemporal modalities that may be stored on their OWN time axis (every-capture
# data): ``{modality}_time_index`` [B, T_m] maps each of the modality's timestep
# slots to its row in ``timestamps`` (-1 = padding slot). When it is None the
# modality's slots ARE the ``timestamps`` rows (the shared-grid layout).
TIME_INDEXED_MODALITIES = ("sentinel2_l2a", "sentinel1", "landsat", "landsat_l2")
TIME_INDEX_SUFFIX = "_time_index"


def time_index_field(modality: str) -> str:
    """Name of the per-modality time-index field."""
    return f"{modality}{TIME_INDEX_SUFFIX}"


TIME_INDEX_FIELDS = tuple(time_index_field(m) for m in TIME_INDEXED_MODALITIES)

# Per-pixel cloud flags of sentinel2_l2a's own timesteps ([B, H, W, T_s2] uint8,
# 1 = cloud or cloud shadow in the S2 scene classification). Loss-side metadata
# (cloudy S2 tokens are not decoder targets), never a model input or modality.
S2_CLOUD_FIELD = "sentinel2_l2a_cloud"


# =============================================================================
# Shared standalone helpers (called by NamedTuple methods to avoid duplication)
# =============================================================================


def _as_dict(obj: NamedTuple, include_nones: bool = False) -> dict[str, Any]:
    """Convert a NamedTuple to a dict, optionally including None values."""
    result = {}
    for name in obj._fields:
        val = getattr(obj, name)
        if include_nones or val is not None:
            result[name] = val
    return result


def _modalities(obj: NamedTuple) -> list[str]:
    """Get present modalities (excludes masks, timestamps, time indices, cloud flags)."""
    return [
        name
        for name in obj._fields
        if not name.endswith("_mask")
        and not name.endswith(TIME_INDEX_SUFFIX)
        and name not in (TIMESTAMPS_FIELD, S2_CLOUD_FIELD)
        and getattr(obj, name) is not None
    ]


def _modality_timestamps(obj: NamedTuple) -> dict[str, Any] | None:
    """Per-modality ``[B, T_m, 3]`` timestamps for modalities on their own time axis.

    Gathers ``timestamps`` rows by each set ``{modality}_time_index`` (padding
    slots, index -1, read row 0; their data is MISSING). None when no modality
    has a time index (the shared-grid layout).
    """
    timestamps = getattr(obj, TIMESTAMPS_FIELD)
    out = {}
    for modality in TIME_INDEXED_MODALITIES:
        index = getattr(obj, time_index_field(modality))
        if index is None:
            continue
        index = index.long().clamp(min=0)
        out[modality] = timestamps.gather(
            1, index.unsqueeze(-1).expand(-1, -1, timestamps.shape[-1])
        )
    return out or None


def _get_masked_modality_name(modality: str) -> str:
    return f"{modality}_mask"


def _get_unmasked_modality_name(modality_mask_name: str) -> str:
    return modality_mask_name.replace("_mask", "")


class OlmoEarthSample(NamedTuple):
    """A sample of the data from the OlmoEarth Pretrain dataset.

    This NamedTuple contains the data of a single sample or a batch of samples.
    For each modality, we have an ArrayTensor named by the modality,
    along with the latlon and timestamps.
    """

    # Modality fields
    sentinel2_l2a: ArrayTensor | None = None  # [B, H, W, T, len(S2_bands)]
    sentinel1: ArrayTensor | None = None  # [B, H, W, T, len(S1_bands)]
    worldcover: ArrayTensor | None = None  # [B, H, W, 1, len(WC_bands)]
    # worldcover_onehot is derived from worldcover at load time (one channel per class).
    worldcover_onehot: ArrayTensor | None = (
        None  # [B, H, W, 1, len(WORLDCOVER_CLASSES)]
    )
    openstreetmap_raster: ArrayTensor | None = None  # [B, H, W, 1, len(OSM_bands)]
    srtm: ArrayTensor | None = None  # [B, H, W, 1, len(SRTM_bands)]
    landsat: ArrayTensor | None = None  # [B, H, W, T, len(LANDSAT_bands)]
    landsat_l2: ArrayTensor | None = None  # [B, H, W, T, len(LANDSAT_L2_bands)]
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
    # ndvi is computed from S2 L2A bands B04 (Red) and B08 (NIR), not loaded from file.
    ndvi: ArrayTensor | None = None  # [B, H, W, T, 1]
    eurocrops: ArrayTensor | None = None  # [B, H, W, 1, 1]
    latlon: ArrayTensor | None = None  # [B, 2]
    timestamps: ArrayTensor | None = None  # [B, T, D=3], where D=[day, month, year]
    # Own-time-axis layout (see TIME_INDEXED_MODALITIES): [B, T_m] rows of timestamps.
    sentinel2_l2a_time_index: ArrayTensor | None = None
    sentinel1_time_index: ArrayTensor | None = None
    landsat_time_index: ArrayTensor | None = None
    landsat_l2_time_index: ArrayTensor | None = None
    sentinel2_l2a_cloud: ArrayTensor | None = None  # [B, H, W, T_s2], S2_CLOUD_FIELD

    def as_dict(self, include_nones: bool = False) -> dict[str, ArrayTensor | None]:
        """Convert to a dictionary.

        Args:
            include_nones: Whether to include None values.
        """
        return _as_dict(self, include_nones=include_nones)

    @property
    def modalities(self) -> list[str]:
        """Get the present modalities (excludes masks and timestamps)."""
        return _modalities(self)

    @property
    def modalities_with_timestamps(self) -> list[str]:
        """Get all modalities including timestamps if present (excludes masks)."""
        result = []
        for name in self._fields:
            if not name.endswith("_mask") and getattr(self, name) is not None:
                result.append(name)
        return result

    @property
    def batch_size(self) -> int:
        """Get the batch size of the data."""
        vals = [
            cast(ArrayTensor, x).shape[0]
            for x in self.as_dict(include_nones=False).values()
        ]
        if len(set(vals)) == 1:
            return vals[0]
        else:
            return 1

    def shape(self, attribute: str, mask: bool = False) -> Sequence[int]:
        """Returns the expected shape of an attribute."""
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
        if attribute.endswith(TIME_INDEX_SUFFIX):
            return 1  # one timeline row per timestep slot
        if attribute == S2_CLOUD_FIELD:
            return 1  # one cloud flag per pixel and timestep
        return Modality.get(attribute).num_bands

    def to_device(
        self, device: torch.device, non_blocking: bool = True
    ) -> OlmoEarthSample:
        """Move all tensors to the specified device."""
        return OlmoEarthSample(
            **{
                key: val.to(device, non_blocking=non_blocking)
                for key, val in self.as_dict(include_nones=False).items()
                if val is not None
            }
        )

    def distribute_tensors(self, device_mesh: DeviceMesh) -> OlmoEarthSample:
        """Distribute the tensors to the specified device mesh."""
        return OlmoEarthSample(
            **{
                key: distribute_tensor(val, device_mesh)
                for key, val in self.as_dict(include_nones=False).items()
            }
        )

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
            assert height is not None and width is not None, (
                f"height and width required for spatial modality {attribute}"
            )
            return (
                height * modality_spec.image_tile_size_factor,
                width * modality_spec.image_tile_size_factor,
                time,
                num_bands,
            )
        elif modality_spec.is_space_only_varying:
            assert height is not None and width is not None, (
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
        return OlmoEarthSample(
            **{k: cast(ArrayTensor, v) * s for k, v in self.as_dict().items()}
        )

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
        return OlmoEarthSample(**summed_dict)

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
        return OlmoEarthSample(**output_dict)


class MaskedOlmoEarthSample(NamedTuple):
    """A masked sample of the data from the OlmoEarth Pretrain dataset.

    For each modality we have an ArrayTensor named by modality,
    and a mask for each modality named by modality_mask.
    """

    timestamps: (
        ArrayTensor  # [B, T, D=3], where D=[day, month, year] (months are zero indexed)
    )
    sentinel2_l2a: Tensor | None = None
    sentinel2_l2a_mask: Tensor | None = None
    sentinel1: Tensor | None = None
    sentinel1_mask: Tensor | None = None
    worldcover: Tensor | None = None
    worldcover_mask: Tensor | None = None
    worldcover_onehot: Tensor | None = None
    worldcover_onehot_mask: Tensor | None = None
    latlon: Tensor | None = None  # [B, 2]
    latlon_mask: Tensor | None = None
    openstreetmap_raster: Tensor | None = None
    openstreetmap_raster_mask: Tensor | None = None
    srtm: Tensor | None = None
    srtm_mask: Tensor | None = None
    landsat: Tensor | None = None
    landsat_mask: Tensor | None = None
    landsat_l2: Tensor | None = None
    landsat_l2_mask: Tensor | None = None
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
    ndvi: Tensor | None = None
    ndvi_mask: Tensor | None = None
    eurocrops: Tensor | None = None
    eurocrops_mask: Tensor | None = None
    # Own-time-axis layout (see TIME_INDEXED_MODALITIES): [B, T_m] rows of timestamps.
    sentinel2_l2a_time_index: Tensor | None = None
    sentinel1_time_index: Tensor | None = None
    landsat_time_index: Tensor | None = None
    landsat_l2_time_index: Tensor | None = None

    def as_dict(self, include_nones: bool = False) -> dict[str, Any]:
        """Convert to a dictionary.

        Args:
            include_nones: Whether to include None values.
        """
        return _as_dict(self, include_nones=include_nones)

    @property
    def modalities(self) -> list[str]:
        """Get the present modalities (excludes masks and timestamps)."""
        return _modalities(self)

    @staticmethod
    def get_masked_modality_name(modality: str) -> str:
        """Get the masked modality name."""
        return _get_masked_modality_name(modality)

    @staticmethod
    def get_unmasked_modality_name(modality_mask_name: str) -> str:
        """Get the unmasked modality name."""
        return _get_unmasked_modality_name(modality_mask_name)

    @property
    def batch_size(self) -> int:
        """Get the batch size of the sample."""
        return self.timestamps.shape[0]

    def to_device(
        self, device: torch.device, non_blocking: bool = True
    ) -> MaskedOlmoEarthSample:
        """Move all tensors to the specified device.

        Masks compacted to uint8 for the dataloader transfer (see
        ``MaskedOlmoEarthSample.with_uint8_masks``) are restored to int64 here.
        """
        moved = {}
        for key, val in self.as_dict(include_nones=False).items():
            if val is None or not hasattr(val, "to"):
                continue
            val = val.to(device, non_blocking=non_blocking)
            if key in _MASKED_SAMPLE_MASK_FIELDS and val.dtype == torch.uint8:
                val = val.long()
            moved[key] = val
        return MaskedOlmoEarthSample(**moved)

    def modality_timestamps(self) -> dict[str, Tensor] | None:
        """Per-modality [B, T_m, 3] timestamps (own-time-axis layout), else None."""
        return _modality_timestamps(self)

    def with_uint8_masks(self) -> MaskedOlmoEarthSample:
        """Cast masks to uint8 (values 0-3) to shrink dataloader payloads 8x.

        ``to_device`` casts them back to int64.
        """
        updates = {}
        for name in _MASKED_SAMPLE_MASK_FIELDS:
            val = getattr(self, name)
            if isinstance(val, torch.Tensor) and val.dtype == torch.int64:
                updates[name] = val.to(torch.uint8)
        return self._replace(**updates)

    def unmask(self) -> MaskedOlmoEarthSample:
        """Return an unmasked MaskedOlmoEarthSample.

        All mask values are MaskValue.ONLINE_ENCODER except for MaskValue.MISSING,
        which remain MISSING.
        """
        updates = {}
        for name in _MASKED_SAMPLE_MASK_FIELDS:
            val = getattr(self, name)
            if val is not None:
                updates[name] = val * (val == MaskValue.MISSING.value)
        return self._replace(**updates)

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
            if key == S2_CLOUD_FIELD:
                continue  # consumed by the collator, not part of the masked sample
            if key == "timestamps" or key.endswith(TIME_INDEX_SUFFIX):
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
        """Create a MaskedOlmoEarthSample from a dictionary."""
        return cls(**d)


# Pre-computed tuple of mask field names for faster iteration in unmask()
_MASKED_SAMPLE_MASK_FIELDS: tuple[str, ...] = tuple(
    f for f in MaskedOlmoEarthSample._fields if f.endswith("_mask")
)


class TokensAndMasks(NamedTuple):
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
    worldcover_onehot: Tensor | None = None
    worldcover_onehot_mask: Tensor | None = None
    openstreetmap_raster: Tensor | None = None
    openstreetmap_raster_mask: Tensor | None = None
    srtm: Tensor | None = None
    srtm_mask: Tensor | None = None
    landsat: Tensor | None = None
    landsat_mask: Tensor | None = None
    landsat_l2: Tensor | None = None
    landsat_l2_mask: Tensor | None = None
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
    ndvi: Tensor | None = None
    ndvi_mask: Tensor | None = None
    eurocrops: Tensor | None = None
    eurocrops_mask: Tensor | None = None
    latlon: Tensor | None = None
    latlon_mask: Tensor | None = None

    def as_dict(self, include_nones: bool = False) -> dict[str, Any]:
        """Convert to a dictionary.

        Args:
            include_nones: Whether to include None values.
        """
        return _as_dict(self, include_nones=include_nones)

    @property
    def modalities(self) -> list[str]:
        """Get the present modalities (excludes masks and timestamps)."""
        return _modalities(self)

    @staticmethod
    def get_masked_modality_name(modality: str) -> str:
        """Get the masked modality name."""
        return _get_masked_modality_name(modality)

    @staticmethod
    def get_unmasked_modality_name(modality_mask_name: str) -> str:
        """Get the unmasked modality name."""
        return _get_unmasked_modality_name(modality_mask_name)

    @property
    def batch_size(self) -> int:
        """Get the batch size."""
        for name in self._fields:
            val = getattr(self, name)
            if val is not None:
                return val.shape[0]
        raise ValueError("No data to get batch size from")

    def to_device(
        self, device: torch.device, non_blocking: bool = True
    ) -> TokensAndMasks:
        """Move all tensors to the specified device."""
        return TokensAndMasks(
            **{
                key: val.to(device, non_blocking=non_blocking)
                for key, val in self.as_dict(include_nones=False).items()
                if val is not None and hasattr(val, "to")
            }
        )

    @property
    def device(self) -> torch.device:
        """Get the device of the tokens and masks."""
        for name in self._fields:
            val = getattr(self, name)
            if val is not None:
                return val.device
        raise ValueError("No data to get device from")

    def get_shape_dict(self) -> dict[str, tuple]:
        """Return a dictionary of the shapes of the fields."""
        return {
            name: getattr(self, name).shape
            for name in self._fields
            if getattr(self, name) is not None
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
        """Flatten tokens and masks, returning separate tensors per modality."""
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
