"""Data structures for OlmoEarth Pretrain."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any, NamedTuple

import torch

from olmoearth_pretrain.types import ArrayTensor

if TYPE_CHECKING:
    from olmoearth_pretrain.data.dataset import OlmoEarthSample


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
