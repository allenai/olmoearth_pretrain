"""OlmoEarth Pretrain wrapper for CROMA."""

import logging
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from einops import rearrange
from olmo_core.config import Config
from torch import nn
from upath import UPath

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.nn.flexi_vit import PoolingType
from olmoearth_pretrain.train.masking import MaskedOlmoEarthSample

from .use_croma import PretrainedCROMA

logger = logging.getLogger(__name__)

HELIOS_SENTINEL2_BANDS = [
    Modality.SENTINEL2_L2A.band_order.index(b)
    for b in [
        "B01",
        "B02",
        "B03",
        "B04",
        "B05",
        "B06",
        "B07",
        "B08",
        "B8A",
        "B09",
        "B11",
        "B12",
    ]
]

CROMA_SIZES = ["base", "large"]


class Croma(nn.Module):
    """Class containing the Croma model that can ingest MaskedOlmoEarthSample objects."""

    patch_size: int = 8
    image_resolution: int = 120
    supported_modalities: list[str] = [
        Modality.SENTINEL2_L2A.name,
        Modality.SENTINEL1.name,
    ]
    supports_multiple_modalities_at_once = True

    def __init__(
        self,
        size: str = "base",
        load_directory: str = "/weka/dfive-default/helios/models/croma",
    ):
        """Initialize the Croma wrapper.

        Args:
            size: The model size
            load_directory: The directory to load from
        """
        super().__init__()
        load_dir = UPath(load_directory)
        if size not in CROMA_SIZES:
            raise ValueError(f"Invalid size: {size}. Must be one of {CROMA_SIZES}")
        load_path = load_dir / f"CROMA_{size}.pt"

        self.model = PretrainedCROMA(
            pretrained_path=str(load_path),
            size=size,
            modality="both",
            image_resolution=self.image_resolution,
        )

    def _process_modality_data(
        self, data: torch.Tensor, modality: str
    ) -> list[torch.Tensor]:
        """Process individual modality data.

        Args:
            data: Input tensor of shape [B, H, W, T, C]
            modality: What modality data is

        Returns:
            list of tensors of shape [B, C, H, W]
        """
        # Rearrange from "b h w t c -> b (c t) h w" for DinoV2/Panopticon format
        t_dim = data.shape[3]

        # Get original dimensions
        original_height = data.shape[2]
        data_list = []

        for i in range(t_dim):
            data_i = rearrange(data[:, :, :, i, :], "b h w c -> b c h w")

            # Rearrange sen2 data
            if modality == "sentinel2_l2a":
                data_i = data_i[:, HELIOS_SENTINEL2_BANDS, :, :]

            new_height = (
                self.model.patch_size if original_height == 1 else self.image_resolution
            )

            data_i = F.interpolate(
                data_i,
                size=(new_height, new_height),
                mode="bilinear",
                align_corners=False,
            )
            data_list.append(data_i)
        return data_list

    def prepare_input(
        self,
        masked_olmoearth_sample: MaskedOlmoEarthSample,
    ) -> list[dict[str, torch.Tensor]]:
        """Prepare input for the CHROMA model from MaskedOlmoEarthSample."""
        input_data_timesteps: dict[int, dict[str, torch.Tensor]] = {}
        for modality in masked_olmoearth_sample.modalities:
            if modality not in self.supported_modalities:
                logger.warning(
                    f"Skipping modality {modality} as it is not in the supported modalities list {self.supported_modalities}"
                )
                continue

            data = getattr(masked_olmoearth_sample, modality)

            if data is None:
                continue

            # Process the modality data
            processed_data = self._process_modality_data(data, modality)
            croma_modality = (
                "optical_images"
                if modality == Modality.SENTINEL2_L2A.name
                else "SAR_images"
            )
            for i, data_i in enumerate(processed_data):
                # start the list if it doesn't exist
                if i not in input_data_timesteps:
                    input_data_timesteps[i] = {}
                input_data_timesteps[i][croma_modality] = data_i

        if not input_data_timesteps:
            raise ValueError("No valid modalities found for processing")
        return [input_data_timesteps[i] for i in sorted(input_data_timesteps.keys())]

    def forward(
        self,
        masked_olmoearth_sample: MaskedOlmoEarthSample,
        pooling: PoolingType = PoolingType.MEAN,
        spatial_pool: bool = False,
    ) -> torch.Tensor:
        """Forward pass through croma model."""
        # Configure model modality

        if masked_olmoearth_sample.sentinel2_l2a is not None:
            if masked_olmoearth_sample.sentinel1 is not None:
                self.model.modality = "both"
            else:
                self.model.modality = "optical"
        else:
            self.model.modality = "SAR"

        # Prepare input
        per_timestep_inputs = self.prepare_input(masked_olmoearth_sample)

        output_features = []
        output_keys = {
            "optical": "optical_encodings",
            "SAR": "SAR_encodings",
            "both": "joint_encodings",
        }
        output_key = output_keys[self.model.modality]
        for data in per_timestep_inputs:
            timestep_output = self.model(**data)[output_key]
            if not spatial_pool:
                timestep_output = timestep_output.mean(dim=1)
            else:
                side = math.isqrt(timestep_output.shape[1])
                timestep_output = rearrange(
                    timestep_output, "b (h w) c -> b h w c", h=side, w=side
                )
            output_features.append(timestep_output.unsqueeze(0))
        # stack in the timestep dimension and take the mean or maybe the max?
        if pooling == PoolingType.MEAN:
            output_features = torch.cat(output_features, dim=0).mean(dim=0)
        elif pooling == PoolingType.MAX:
            output_features = torch.max(torch.cat(output_features, dim=0), dim=0)[0]
        return output_features


@dataclass
class CromaConfig(Config):
    """olmo_core style config for CromaWrapper."""

    size: str = "base"
    load_directory: str = "/weka/dfive-default/helios/models/croma"

    def build(self) -> Croma:
        """Build the Croma model."""
        return Croma(
            size=self.size,
            load_directory=self.load_directory,
        )
