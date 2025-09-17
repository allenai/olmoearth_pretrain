"""Helios wrapper for Satlas."""

import logging
from dataclasses import dataclass

import satlaspretrain_models
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from olmo_core.config import Config
from satlaspretrain_models.utils import Backbone
from upath import UPath

from helios.data.constants import Modality
from helios.nn.flexihelios import PoolingType
from helios.train.masking import MaskedHeliosSample

logger = logging.getLogger(__name__)


HELIOS_TO_SATLAS_S2 = [
    Modality.SENTINEL2_L2A.band_order.index(b)
    for b in ["B04", "B03", "B02", "B05", "B06", "B07", "B08", "B11", "B12"]
]


class SatlasPretrain(nn.Module):
    """Class containing the Satlas model that can ingest MaskedHeliosSample objects."""

    patch_size: int = 8
    image_resolution: int = 120
    supported_modalities: list[str] = [
        Modality.SENTINEL2_L2A.name,
        Modality.SENTINEL1.name,
        Modality.LANDSAT.name,
    ]

    def __init__(self, load_directory: str, size: str = "base") -> None:
        """Initialize the Satlas wrapper.

        Args:
            size: The model size
            load_directory: The directory to load from
        """
        super().__init__()
        self.satlas: satlaspretrain_models.Model | None = None
        self.size = Backbone.SWINB if size == "base" else Backbone.SWINT
        self.dim = 1024 if size == "base" else 768
        self.load_directory = UPath(load_directory)
        self.init_modality: str | None = None

        self.image_resolution = 512

    def _initialize_model(self, modality: str, multitemporal: bool) -> None:
        # check init modality to see if we need to reinitialize the model
        weights = torch.load(
            self.load_directory / "satlas-model-v1-lowres-band.pth", map_location="cpu"
        )  # todo: how to select this?
        num_channels = 9  # this also needs to be selected based on the modality
        self.satlas = satlaspretrain_models.Model(
            num_channels=num_channels,
            multi_image=False,
            backbone=self.size,
            fpn=False,
            head=None,
            num_categories=None,
            weights=weights,
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
        t_dim = data.shape[3]

        # Get original dimensions
        original_height = data.shape[2]
        data_list = []

        for i in range(t_dim):
            data_i = rearrange(data[:, :, :, i, :], "b h w c -> b c h w")

            # Rearrange sen2 data
            if modality == "sentinel2_l2a":
                data_i = data_i[:, HELIOS_TO_SATLAS_S2, :, :]

            new_height = (
                self.patch_size if original_height == 1 else self.image_resolution
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
        masked_helios_sample: MaskedHeliosSample,
    ) -> list[torch.Tensor]:
        """Prepare input for the Satlas model from MaskedHeliosSample."""
        if len(masked_helios_sample.modalities) != 1:
            raise RuntimeError(
                f"Satlas only supports one modality. Received {len(masked_helios_sample.modalities)}: {masked_helios_sample.modalities}"
            )
        modality = masked_helios_sample.modalities[0]
        is_multitemporal = masked_helios_sample.timestamps.shape[1] > 1
        self._initialize_model(modality, is_multitemporal)

        data = getattr(masked_helios_sample, modality)
        return self._process_modality_data(data, modality)

    def forward(
        self,
        masked_helios_sample: MaskedHeliosSample,
        pooling: PoolingType = PoolingType.MEAN,
        spatial_pool: bool = False,
    ) -> torch.Tensor:
        """Forward pass through the satlas model."""
        processed_inputs = self.prepare_input(masked_helios_sample)
        outputs_list: list[torch.Tensor] = []
        for per_t_input in processed_inputs:
            assert self.satlas is not None  # since we called self.prepare_input
            output = self.satlas(per_t_input)[-1]
            # output shape for atto: (bsz, 320, 7, 7)
            # output shape for tiny: (bsz, 768, 6, 6)
            if not spatial_pool:
                # then we don't want to keep the spatial dimensions
                output = output.mean(dim=-1).mean(dim=-1)
            else:
                output = rearrange(output, "b c h w -> b h w c")
            outputs_list.append(output)

        # stack in the timestep dimension and take the mean or maybe the max?
        if pooling == PoolingType.MEAN:
            output_features = torch.cat(outputs_list, dim=0).mean(dim=0)
        elif pooling == PoolingType.MAX:
            output_features = torch.max(torch.cat(outputs_list, dim=0), dim=0)[0]
        return output_features


@dataclass
class SatlasConfig(Config):
    """olmo_core style config for Satlas Wrapper."""

    size: str = "base"
    load_directory: str = "/weka/dfive-default/helios/models/satlas"

    def build(self) -> SatlasPretrain:
        """Build the Satlas model."""
        return SatlasPretrain(
            size=self.size,
            load_directory=self.load_directory,
        )
