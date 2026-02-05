"""Data structures for OlmoEarth Pretrain."""

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum
from math import floor
from typing import Any, NamedTuple, cast

import numpy as np
import torch
from torch.distributed import DeviceMesh
from torch.distributed.tensor import distribute_tensor

from olmoearth_pretrain.data.constants import MISSING_VALUE, TIMESTAMPS, Modality
from olmoearth_pretrain.types import ArrayTensor


class OlmoEarthSample(NamedTuple):
    """A sample of the data from the OlmoEarth Pretrain dataset.

    This is a namedtuple that contains the data of a single sample or a batch of samples from the OlmoEarth Pretrain dataset.
    For each modality, we have an ArrayTensor named by the modality, along with the latlon and timestamps.
    """

    sentinel2_l2a: ArrayTensor | None = None  # [B, H, W, T, len(S2_bands)]
    latlon: ArrayTensor | None = None  # [B, 2]
    timestamps: ArrayTensor | None = None  # [B, T, D=3], where D=[day, month, year]
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

    # TODO: Add unit tests for this
    def shape(self, attribute: str, mask: bool = False) -> Sequence[int]:
        """Returns the expected shape of an attribute.

        This is useful if you want to know what the shape of a
        missing attribute would have been for this sample.

        Args:
            attribute: The attribute to get the shape of, e.g., "sentinel2", "timestamps", etc.
            mask: Whether to get the shape of the mask.

        Returns:
            The shape of the attribute.
        """
        # It is safe to assume we always have Sentinel2, timestamps, and latlon
        # If other attributes are missing, we use Sentinel2 to get its partial shape (B, H, W, T)
        # For static modality like worldcover, we specify the T dimension as 1
        if attribute == "timestamps":
            if not mask:
                if self.timestamps is None:
                    raise ValueError("Timestamps are not present in the sample")
                return self.timestamps.shape
            else:
                # timestamps is a special case which is not in Modality
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

    def as_dict(self, ignore_nones: bool = True) -> dict[str, ArrayTensor | None]:
        """Convert the namedtuple to a dictionary.

        Args:
            ignore_nones: Whether to ignore None values.

        Returns:
            Dictionary representation of the namedtuple.
        """
        return_dict = {}
        for field in self._fields:
            val = getattr(self, field)
            if ignore_nones and (val is None):
                continue
            else:
                return_dict[field] = val
        return return_dict

    @property
    def modalities(self) -> list[str]:
        """Get the modalities present in the sample.

        Includes timestamps and latlon
        """
        return [modality for modality in self.as_dict(ignore_nones=True).keys()]

    def to_device(
        self, device: torch.device, non_blocking: bool = True
    ) -> OlmoEarthSample:
        """Move all tensors to the specified device.

        Args:
            device: The device to move the tensors to.
            non_blocking: Whether or not to use asynchronous GPU copies

        Returns:
            A new OlmoEarthSample with all tensors moved to the specified device.
        """
        return OlmoEarthSample(
            **{
                key: val.to(device, non_blocking=non_blocking)
                for key, val in self.as_dict(ignore_nones=True).items()
                if val is not None
            }
        )

    def distribute_tensors(self, device_mesh: DeviceMesh) -> OlmoEarthSample:
        """Distribute the tensors to the specified device mesh."""
        return OlmoEarthSample(
            **{
                key: distribute_tensor(val, device_mesh)
                for key, val in self.as_dict(ignore_nones=True).items()
            }
        )

    @property
    def batch_size(self) -> int:
        """Get the batch size of the data."""
        vals = [
            cast(ArrayTensor, x).shape[0]
            for x in self.as_dict(ignore_nones=True).values()
        ]
        if len(set(vals)) == 1:
            return vals[0]
        else:
            return 1

    @property
    def height(self) -> int:
        """Get the height of the data at resolution_factor == 16."""
        for modality in self.modalities:
            if modality == "timestamps":
                continue
            modality_spec = Modality.get(modality)
            if not modality_spec.is_spatial:
                continue
            x = getattr(self, modality)
            if x is not None:
                if len(x.shape) == 5:
                    return x.shape[1] // modality_spec.image_tile_size_factor
                else:
                    # no batch dimension
                    if len(x.shape) != 4:
                        raise ValueError(f"Unexpected shape {x.shape} for {modality}")
                    return x.shape[0] // modality_spec.image_tile_size_factor
        raise ValueError("No modality with height or width present")

    @property
    def width(self) -> int:
        """Get the width of the data at resolution_factor == 16."""
        for modality in self.modalities:
            if modality == "timestamps":
                continue
            modality_spec = Modality.get(modality)
            if not modality_spec.is_spatial:
                continue
            x = getattr(self, modality)
            if x is not None:
                if len(x.shape) == 5:
                    return x.shape[2] // modality_spec.image_tile_size_factor
                else:
                    # no batch dimension
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
            if modality == "timestamps":
                continue
            modality_spec = Modality.get(modality)
            if modality_spec.is_multitemporal:
                data = getattr(self, modality)
                if isinstance(data, np.ndarray):
                    raise ValueError(
                        "timesteps_with_at_least_one_modality is not yet supported for numpy arrays"
                    )
                # Get all timestamps that are present for all samples for the given modality
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

    def _get_max_t_within_token_budget(
        self, h_w_p: int, max_tokens_per_instance: int
    ) -> int:
        """Find max t possible when subsetting.

        Given a sampled h_w_p (the number of tokens along the h and w dimensions)
        return the maximum t allowed within the
        max_tokens budget so that the patchified
        OlmoEarthSample will have fewer than max_tokens tokens.

        This function assumes we apply (H, W, T=1 patchifying)
        """
        used_tokens = 0
        time_multiply_tokens = 0
        for attribute in self.as_dict(ignore_nones=True).keys():
            if attribute == "timestamps":
                continue
            modality_spec = Modality.get(attribute)
            if modality_spec.is_spacetime_varying:
                # for now, lets assume fixed resolution
                time_multiply_tokens += (h_w_p**2) * modality_spec.num_band_sets
            elif modality_spec.is_space_only_varying:
                # for now, lets assume fixed resolution
                used_tokens += (h_w_p**2) * modality_spec.num_band_sets
            elif modality_spec.is_time_only_varying:
                time_multiply_tokens += modality_spec.num_band_sets
            elif modality_spec.is_static_in_space_and_time:
                used_tokens += modality_spec.num_band_sets
        if time_multiply_tokens == 0:
            # no time-varying inputs, so our return value of t
            # doesn't matter
            return 1
        remaining_tokens = max_tokens_per_instance - used_tokens
        max_t_within_budget = remaining_tokens / time_multiply_tokens
        if max_t_within_budget < 1:
            raise ValueError(
                f"patch_size too small for this sample and budget, h_w_p: {h_w_p}, max_tokens: {max_tokens_per_instance}"
            )

        return min(floor(max_t_within_budget), self.time)

    @staticmethod
    def _get_valid_start_ts(
        missing_timesteps: dict[str, Any], max_t: int, current_length: int
    ) -> list[int]:
        """Get valid starting timesteps."""
        import logging

        logger = logging.getLogger(__name__)
        if current_length > max_t:
            # We can randomly sample from the range of valid starting timesteps because current_length exceeds max_t
            if not missing_timesteps:
                # No missing timesteps info available - all timesteps are potentially valid
                # Create a range of all possible starting positions that fit within max_t
                valid_start_ts = list(range(current_length - max_t + 1))
            else:
                # We have missing timesteps info - need to find valid starting positions
                # that ensure we have at least some present data at the chosen start_t
                start_ts = set()
                for modality in missing_timesteps:
                    valid_timesteps = np.flatnonzero(missing_timesteps[modality])
                    valid_timesteps = valid_timesteps[
                        valid_timesteps + max_t <= current_length
                    ]
                    start_ts.update(valid_timesteps)
                valid_start_ts = list(start_ts)
        else:
            # Picking the first timestep aims to maximize the number of present timesteps
            valid_start_ts = [0]
        if len(valid_start_ts) == 0:
            logger.warning(
                f"No valid start timesteps found for {missing_timesteps} with max_t {max_t} and current_length {current_length}"
            )
            raise ValueError(
                f"No valid start timesteps found for {missing_timesteps} with max_t {max_t} and current_length {current_length}"
            )
        return sorted(valid_start_ts)

    def subset_default(
        self,
        patch_size: int,
        max_tokens_per_instance: int | None,
        sampled_hw_p: int,
        current_length: int,
        missing_timesteps_masks: dict[str, Any] = {},
    ) -> OlmoEarthSample:
        """Subset a OlmoEarthSample using default rectangular cropping.

        Args:
            patch_size: The patch size being applied to this sample.
            max_tokens_per_instance: The token budget when subsetting. This is used
                to determine the maximum number of timesteps possible for a given
                height and width. If None, this operation is a no-op.
            sampled_hw_p: The number of tokens in the height and width dimensions.
            current_length: The current maximum sequence length of the sample.
            missing_timesteps_masks: A dictionary of missing timesteps masks.

        Returns:
            A subsetted OlmoEarthSample with rectangular cropping applied.
        """
        if max_tokens_per_instance is None:
            return self
        max_t = self._get_max_t_within_token_budget(
            sampled_hw_p, max_tokens_per_instance
        )
        valid_start_ts = self._get_valid_start_ts(
            missing_timesteps_masks, max_t, current_length
        )
        start_t = np.random.choice(valid_start_ts)
        new_data_dict: dict[str, ArrayTensor] = {}

        sampled_hw = sampled_hw_p * patch_size
        start_h = np.random.choice(self.height - sampled_hw + 1)
        start_w = np.random.choice(self.width - sampled_hw + 1)

        for attribute, modality in self.as_dict(ignore_nones=True).items():
            assert modality is not None
            if attribute == "timestamps":
                new_data_dict[attribute] = modality[start_t : start_t + max_t]
                continue
            modality_spec = Modality.get(attribute)
            if modality_spec.is_spacetime_varying:
                new_data_dict[attribute] = modality[
                    start_h * modality_spec.image_tile_size_factor : (
                        start_h + sampled_hw
                    )
                    * modality_spec.image_tile_size_factor,
                    start_w * modality_spec.image_tile_size_factor : (
                        start_w + sampled_hw
                    )
                    * modality_spec.image_tile_size_factor,
                    start_t : start_t + max_t,
                ]
            elif modality_spec.is_space_only_varying:
                new_data_dict[attribute] = modality[
                    start_h * modality_spec.image_tile_size_factor : (
                        start_h + sampled_hw
                    )
                    * modality_spec.image_tile_size_factor,
                    start_w * modality_spec.image_tile_size_factor : (
                        start_w + sampled_hw
                    )
                    * modality_spec.image_tile_size_factor,
                ]
            elif modality_spec.is_time_only_varying:
                new_data_dict[attribute] = modality[start_t : start_t + max_t]
            elif modality_spec.is_static_in_space_and_time:
                new_data_dict[attribute] = modality

        return OlmoEarthSample(**new_data_dict)

    def subset_cutmix(
        self,
        patch_size: int,
        max_tokens_per_instance: int | None,
        sampled_hw_p: int,
        current_length: int,
        missing_timesteps_masks: dict[str, Any] = {},
    ) -> OlmoEarthSample:
        """Subset a OlmoEarthSample using CutMix patch sampling.

        Args:
            patch_size: The patch size being applied to this sample.
            max_tokens_per_instance: The token budget when subsetting. This is used
                to determine the maximum number of timesteps possible for a given
                height and width. If None, this operation is a no-op.
            sampled_hw_p: The number of tokens in the height and width dimensions.
            current_length: The current maximum sequence length of the sample.
            missing_timesteps_masks: A dictionary of missing timesteps masks.

        Returns:
            A subsetted OlmoEarthSample with CutMix patch sampling applied.
        """
        if max_tokens_per_instance is None:
            return self
        max_t = self._get_max_t_within_token_budget(
            sampled_hw_p, max_tokens_per_instance
        )
        valid_start_ts = self._get_valid_start_ts(
            missing_timesteps_masks, max_t, current_length
        )
        start_t = np.random.choice(valid_start_ts)
        new_data_dict: dict[str, ArrayTensor] = {}

        height_p, width_p = self.height // patch_size, self.width // patch_size
        h_p_indices = np.random.choice(height_p, size=sampled_hw_p, replace=False)
        w_p_indices = np.random.choice(width_p, size=sampled_hw_p, replace=False)
        h_indices = [
            i
            for h_p in h_p_indices
            for i in range(h_p * patch_size, (h_p + 1) * patch_size)
        ]
        w_indices = [
            i
            for w_p in w_p_indices
            for i in range(w_p * patch_size, (w_p + 1) * patch_size)
        ]
        hh, ww = np.meshgrid(h_indices, w_indices, indexing="ij")

        for attribute, modality in self.as_dict(ignore_nones=True).items():
            assert modality is not None
            if attribute == "timestamps":
                new_data_dict[attribute] = modality[start_t : start_t + max_t]
                continue
            modality_spec = Modality.get(attribute)
            if modality_spec.is_spacetime_varying:
                new_data_dict[attribute] = modality[
                    hh * modality_spec.image_tile_size_factor,
                    ww * modality_spec.image_tile_size_factor,
                    start_t : start_t + max_t,
                ]
            elif modality_spec.is_space_only_varying:
                new_data_dict[attribute] = modality[
                    hh * modality_spec.image_tile_size_factor,
                    ww * modality_spec.image_tile_size_factor,
                ]
            elif modality_spec.is_time_only_varying:
                new_data_dict[attribute] = modality[start_t : start_t + max_t]
            elif modality_spec.is_static_in_space_and_time:
                new_data_dict[attribute] = modality

        return OlmoEarthSample(**new_data_dict)

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
        for key, val in self.as_dict(ignore_nones=True).items():
            assert val is not None  # keep mypy happy. True because ignore_nones=True
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

    def unmask(self) -> MaskedOlmoEarthSample:
        """Return an unmasked MaskedOlmoEarthSample.

        All mask values are MaskValue.ONLINE_ENCODER except for MaskValue.MISSING,
        which remain MISSING.
        """
        updates = {}
        for field in _MASKED_SAMPLE_MASK_FIELDS:
            val = getattr(self, field)
            if val is not None:
                updates[field] = val * (val == MaskValue.MISSING.value)
        return self._replace(**updates)

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

    @classmethod
    def from_olmoearthsample(
        cls,
        sample: OlmoEarthSample,
    ) -> MaskedOlmoEarthSample:
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
    def from_dict(cls, dict: dict[str, Any]) -> MaskedOlmoEarthSample:
        """Create a MaskedOlmoEarthSample from a dictionary, creating empty tensors for missing modalities.

        Args:
            dict: Dictionary representation of the MaskedOlmoEarthSample.
        """
        return cls(**dict)

    def to_device(
        self, device: torch.device, non_blocking: bool = True
    ) -> MaskedOlmoEarthSample:
        """Move all tensors to the specified device.

        Args:
            device: The device to move the tensors to.
            non_blocking: Whether or not to use asynchronous GPU copies

        Returns:
            A new MaskedOlmoEarthSample with all tensors moved to the specified device.
        """
        return MaskedOlmoEarthSample(
            **{
                key: val.to(device, non_blocking=non_blocking)
                for key, val in self.as_dict(return_none=False).items()
            }
        )

    @property
    def batch_size(self) -> int:
        """Get the batch size of the sample.

        Returns:
            The batch size (first dimension of timestamps tensor).
        """
        return self.timestamps.shape[0]

    def pop(
        self, modality: str, remove_mask: bool = True
    ) -> tuple[MaskedOlmoEarthSample, ArrayTensor | None]:
        """Like pop on a dict.

        We need to return the OlmoEarthSample too since tuples are immutable.
        """
        d = self.as_dict()
        return_val = d.pop(modality, None)
        if remove_mask and hasattr(self, self.get_masked_modality_name(modality)):
            _ = d.pop(self.get_masked_modality_name(modality), None)
        new_sample = MaskedOlmoEarthSample(**d)
        return new_sample, return_val


# Pre-computed tuple of mask field names for faster iteration in unmask()
_MASKED_SAMPLE_MASK_FIELDS: tuple[str, ...] = tuple(
    f for f in MaskedOlmoEarthSample._fields if f.endswith("_mask")
)
