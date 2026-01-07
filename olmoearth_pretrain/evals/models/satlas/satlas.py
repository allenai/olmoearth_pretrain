"""OlmoEarth Pretrain wrapper for Satlas."""

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

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.nn.flexi_vit import PoolingType
from olmoearth_pretrain.train.masking import MaskedOlmoEarthSample

logger = logging.getLogger(__name__)


HELIOS_TO_SATLAS = {
    Modality.SENTINEL2_L2A.name: [
        Modality.SENTINEL2_L2A.band_order.index(b)
        # TODO @favyenb is this band ordering correct?
        for b in ["B04", "B03", "B02", "B05", "B06", "B07", "B08", "B11", "B12"]
    ],
    Modality.SENTINEL1.name: [
        Modality.SENTINEL1.band_order.index(b) for b in ["vv", "vh"]
    ],
    # Our Landsat models input 11 bands, B1-B11 in order, of Landsat-8 and Landsat-9 images.
    Modality.LANDSAT.name: [
        Modality.LANDSAT.band_order.index(b)
        for b in ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10", "B11"]
    ],
}


class Satlas(nn.Module):
    """Class containing the Satlas model that can ingest MaskedOlmoEarthSample objects."""

    patch_size: int = 8
    min_image_resolution: int = 32
    supported_modalities: list[str] = [
        Modality.SENTINEL2_L2A.name,
        Modality.SENTINEL1.name,
        Modality.LANDSAT.name,
    ]
    supports_multiple_modalities_at_once = False

    modality_size_to_weights = {
        "base": {
            Modality.SENTINEL2_L2A.name: "sentinel2_swinb_si_ms.pth",
            Modality.LANDSAT.name: "landsat_swinb_si.pth",
            Modality.SENTINEL1.name: "sentinel1_swinb_si.pth",
        },
        "tiny": {
            Modality.SENTINEL2_L2A.name: "sentinel2_swint_si_ms.pth",
        },
    }

    size_to_backbone = {"base": Backbone.SWINB, "tiny": Backbone.SWINT}
    model: satlaspretrain_models.Model

    def __init__(
        self,
        load_directory: str,
        size: str = "base",
        use_pretrained_normalizer: bool = True,
    ) -> None:
        """Initialize the Satlas wrapper.

        Args:
            size: The model size
            load_directory: The directory to load from
            use_pretrained_normalizer: Whether or not to apply satlas pretraining normalization
        """
        super().__init__()
        self.size = size
        self.load_directory = UPath(load_directory)
        # need to have some model at init so that the trainer can build correctly
        supported_modalities = self.modality_size_to_weights[self.size].keys()
        self.models = nn.ModuleDict(
            {modality: self._load_model(modality) for modality in supported_modalities}
        )
        self.dim = 1024 if size == "base" else 768
        self.use_pretrained_normalizer = use_pretrained_normalizer

    def _load_model(self, modality: str) -> satlaspretrain_models.Model:
        # check init modality to see if we need to reinitialize the model
        weights = torch.load(
            self.load_directory / self.modality_size_to_weights[self.size][modality],
            map_location="cpu",
        )
        return satlaspretrain_models.Model(
            num_channels=len(HELIOS_TO_SATLAS[modality]),
            multi_image=False,
            backbone=self.size_to_backbone[self.size],
            fpn=False,
            head=None,
            num_categories=None,
            weights=weights,
        )

    @staticmethod
    def normalize(image: torch.Tensor, modality: str) -> torch.Tensor:
        """https://github.com/allenai/satlas/blob/main/Normalization.md."""
        if modality == Modality.SENTINEL2_L2A.name:
            return torch.clip(image / 8160, 0, 1)
        elif modality == Modality.LANDSAT.name:
            return torch.clip((image - 4000) / 16320, 0, 1)
        elif modality == Modality.SENTINEL1.name:
            return torch.clip(image / 255, 0, 1)
        else:
            raise ValueError(f"Unexpected modality {modality}")

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
            data_i = data_i[:, HELIOS_TO_SATLAS[modality], :, :]

            new_height = (
                self.min_image_resolution
                if original_height < self.min_image_resolution
                else original_height
            )

            data_i = F.interpolate(
                data_i,
                size=(new_height, new_height),
                mode="bilinear",
                align_corners=False,
            )
            if self.use_pretrained_normalizer:
                data_i = self.normalize(data_i, modality)
            data_list.append(data_i)

        return data_list

    def prepare_input(
        self,
        masked_olmoearth_sample: MaskedOlmoEarthSample,
    ) -> tuple[list[torch.Tensor], str]:
        """Prepare input for the Satlas model from MaskedOlmoEarthSample."""
        if len(masked_olmoearth_sample.modalities) != 1:
            raise RuntimeError(
                f"Satlas only supports one modality. Received {len(masked_olmoearth_sample.modalities)}: {masked_olmoearth_sample.modalities}"
            )
        modality = masked_olmoearth_sample.modalities[0]

        data = getattr(masked_olmoearth_sample, modality)
        return self._process_modality_data(data, modality), modality

    def forward(
        self,
        masked_olmoearth_sample: MaskedOlmoEarthSample,
        pooling: PoolingType = PoolingType.MEAN,
        spatial_pool: bool = False,
    ) -> torch.Tensor:
        """Forward pass through the satlas model."""
        processed_inputs, modality = self.prepare_input(masked_olmoearth_sample)
        outputs_list: list[torch.Tensor] = []
        for per_t_input in processed_inputs:
            # we only take the last feature map from satlas for consistency with
            # the other models. For segmentation tasks, multi scale feature maps
            # may be preferred. An example of this is in
            # https://github.com/allenai/rslearn/blob/master/rslearn/models/swin.py
            output = self.models[modality](per_t_input)[-1]
            # output shape for atto: (bsz, 320, 7, 7)
            # output shape for tiny: (bsz, 768, 6, 6)
            if not spatial_pool:
                # then we don't want to keep the spatial dimensions
                output = output.mean(dim=-1).mean(dim=-1)
            else:
                output = rearrange(output, "b c h w -> b h w c")
            outputs_list.append(output.unsqueeze(0))

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
    use_pretrained_normalizer: bool = True

    def build(self) -> Satlas:
        """Build the Satlas model."""
        return Satlas(
            size=self.size,
            load_directory=self.load_directory,
            use_pretrained_normalizer=self.use_pretrained_normalizer,
        )
