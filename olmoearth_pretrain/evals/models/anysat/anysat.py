"""AnySat wrapper to ingest MaskedOlmoEarthSample."""

import logging
from dataclasses import dataclass

import torch
from einops import rearrange
from olmo_core.config import Config
from torch import nn

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.nn.flexi_vit import PoolingType
from olmoearth_pretrain.train.masking import MaskedOlmoEarthSample

logger = logging.getLogger(__name__)


class AnySat(nn.Module):
    """AnySat wrapper for MaskedHelioSample."""

    # these are the bands which AnySat accepts
    # https://github.com/gastruc/AnySat?tab=readme-ov-file#format-your-data
    ANYSAT_S2_BAND_ORDERING = [
        "B02",
        "B03",
        "B04",
        "B05",
        "B06",
        "B07",
        "B08",
        "B8A",
        "B11",
        "B12",
    ]
    ANYSAT_S1_BAND_ORDERING = ["vv", "vh", "ratio"]
    ANYSAT_L8_BAND_ORDERING = [
        "B8",
        "B1",
        "B2",
        "B3",
        "B4",
        "B5",
        "B6",
        "B7",
        "B9",
        "B10",
        "B11",
    ]

    helios_modalities_to_anysat_names = {
        Modality.SENTINEL2_L2A.name: "s2",
        Modality.LANDSAT.name: "l8",
        Modality.SENTINEL1.name: "s1",
    }

    supported_modalities = [
        Modality.SENTINEL2_L2A.name,
        Modality.LANDSAT.name,
        Modality.SENTINEL1.name,
    ]

    supports_multiple_modalities_at_once = True
    resolution: int = 10

    def __init__(self, patch_size: int = 4) -> None:
        """AnySat wrapper."""
        super().__init__()
        self.model = torch.hub.load(
            "gastruc/anysat",
            "anysat",
            pretrained=True,
            flash_attn=False,
            force_reload=True,
        )

        self.modality_to_band_indices = {
            Modality.SENTINEL2_L2A.name: [
                Modality.SENTINEL2_L2A.band_order.index(v)
                for v in self.ANYSAT_S2_BAND_ORDERING
            ],
            Modality.SENTINEL1.name: [
                Modality.SENTINEL1.band_order.index(v)
                for v in self.ANYSAT_S1_BAND_ORDERING
                if v in Modality.SENTINEL1.band_order
            ],
            Modality.LANDSAT.name: [
                Modality.LANDSAT.band_order.index(v)
                for v in self.ANYSAT_L8_BAND_ORDERING
                if v in Modality.LANDSAT.band_order
            ],
        }
        # Patch size in pixels
        self.patch_size = patch_size

    @staticmethod
    def calculate_day_of_year(timestamp: torch.Tensor) -> torch.Tensor:
        """Calculate day of year from timestamp.

        Args:
            timestamp: Tensor of shape (..., 3) where last dim is [day, month, year]

        Returns:
            Tensor of same shape as input without last dim, with day of year as int
        """
        # timestamp[..., 0] = day, timestamp[..., 1] = month, timestamp[..., 2] = year
        day = timestamp[..., 0]
        month = timestamp[..., 1]
        year = timestamp[..., 2]

        # Days in months for non-leap years
        days_in_month = torch.tensor(
            [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], device=timestamp.device
        )

        # Check for leap year: (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
        is_leap = ((year % 4 == 0) & (year % 100 != 0)) | (year % 400 == 0)

        # Cumulative days at the start of each month (0 for Jan, 31 for Feb, etc.)
        cum_days = torch.cat(
            [
                torch.zeros(1, device=timestamp.device, dtype=days_in_month.dtype),
                days_in_month.cumsum(0)[:-1],
            ]
        )

        # Get cumulative days for the given month
        # month is 1-based (Jan=1), so subtract 1 for indexing
        month_idx = month.long() - 1
        cum_days_for_month = cum_days[month_idx]

        # Add 1 if leap year and month > 2
        leap_day = (is_leap & (month > 2)).long()

        doy = cum_days_for_month + day + leap_day
        return doy

    @staticmethod
    def _calculate_patch_size(h: int) -> int:
        """Calculate the patch size in pixels based on the height of the input data.

        Args:
            h: Height of the input data in pixels

        Returns:
            Patch size in pixels
        """
        # Avoid having more than 32x32 patches per tile as suggested by the authors
        h_adjusted = h // 32
        patch_size_exp = (h_adjusted).bit_length()
        patch_size = 2**patch_size_exp
        return patch_size

    def _process_modality_data(self, data: torch.Tensor, modality: str) -> torch.Tensor:
        """Process individual modality data.

        Args:
            data: Input tensor of shape [B, H, W, T, C]
            modality: What modality data is

        Returns:
            tensor of shape [B, T, C, H, W]
        """
        # no model specific normalization - the authors recommend
        # "standard dataset normalization"
        data = rearrange(data, "b h w t c -> b t c h w")
        data = data[:, :, self.modality_to_band_indices[modality], :, :]
        if modality == Modality.SENTINEL1.name:
            # add the ratio
            ratio_band = data[:, :, :1, :, :] / (data[:, :, 1:, :, :] + 1e-6)
            data = torch.concat((data, ratio_band), dim=2)
        return data

    def prepare_input(
        self,
        masked_olmoearth_sample: MaskedOlmoEarthSample,
    ) -> dict[str, torch.Tensor]:
        """Prepare input for the AnySat model from MaskedOlmoEarthSample."""
        input_data: dict[str, dict[str, torch.Tensor]] = {}

        for modality in masked_olmoearth_sample.modalities:
            if modality not in self.helios_modalities_to_anysat_names.keys():
                logger.warning(
                    f"Skipping modality {modality} as it is not in the supported "
                    f"modalities list {self.helios_modalities_to_anysat_names.keys()}"
                )
                continue

            data = getattr(masked_olmoearth_sample, modality)

            if data is None:
                continue

            processed_data = self._process_modality_data(data, modality)
            # Process the modality data
            input_data[self.helios_modalities_to_anysat_names[modality]] = (
                processed_data
            )
            num_timesteps = processed_data.shape[1]
            if num_timesteps > 1:
                assert masked_olmoearth_sample.timestamps is not None

            doy = self.calculate_day_of_year(masked_olmoearth_sample.timestamps)
            input_data[f"{self.helios_modalities_to_anysat_names[modality]}_dates"] = (
                doy
            )
        return input_data

    def forward(
        self,
        masked_olmoearth_sample: MaskedOlmoEarthSample,
        pooling: PoolingType = PoolingType.MEAN,
        spatial_pool: bool = False,
    ) -> torch.Tensor:
        """Forward pass through the AnySat model."""
        processed_inputs = self.prepare_input(masked_olmoearth_sample)
        if pooling == PoolingType.MAX:
            raise ValueError("Unsupported pooling type MAX for AnySat.")

        hs = []
        for key, val in processed_inputs.items():
            if not key.endswith("dates"):
                hs.append(val.shape[-1])
        if len(set(hs)) != 1:
            raise RuntimeError("Expected all inputs to have the same dimension")
        # If patch size is specified, use it, otherwise, caculate the maximum patch size
        # based on the height of the input data
        patch_size_meters = (
            max(self.patch_size, self._calculate_patch_size(hs[0])) * self.resolution
        )
        logger.info(f"Using patch size {patch_size_meters} for AnySat")

        # from the README (https://github.com/gastruc/AnySat/blob/main/README.md):
        # "The sub patches are 1x1 pixels for time series and 10x10 pixels for VHR images.
        # If using output='dense', specify the output_modality."
        # Let's preferentially use output_modality in this order: [s2, s1, landsat]
        input_modalities = [
            k for k in processed_inputs.keys() if not k.endswith("dates")
        ]
        if "s2" in input_modalities:
            output_modality = "s2"
        elif "s1" in input_modalities:
            output_modality = "s1"
        elif "l8" in input_modalities:
            output_modality = "l8"
        else:
            raise RuntimeError(
                f"Expected one of s2, s1, l8 in input modalities, got {input_modalities}"
            )

        if spatial_pool:
            output = "dense"
        else:
            output = "tile"
        output_patches = self.model(
            x=processed_inputs,
            patch_size=patch_size_meters,
            output=output,
            output_modality=output_modality,
        )
        return output_patches


@dataclass
class AnySatConfig(Config):
    """olmo_core style config for AnySat."""

    patch_size: int = 4

    def build(self) -> AnySat:
        """Build the Croma model."""
        return AnySat(patch_size=self.patch_size)
