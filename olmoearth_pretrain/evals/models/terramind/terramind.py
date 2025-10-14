"""Terramind models."""

import logging
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from einops import rearrange
from olmo_core.config import Config
from terratorch.registry import BACKBONE_REGISTRY
from torch import nn

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.nn.flexi_vit import PoolingType
from olmoearth_pretrain.train.masking import MaskedOlmoEarthSample

logger = logging.getLogger(__name__)

TERRAMIND_SIZES = ["base", "large"]
HELIOS_TO_TERRAMIND_SENTINEL2_BANDORDER = [
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

# TerraMind band orders and standardization values
PRETRAINED_BANDS = {
    "S2L2A": {
        "B01": [1390.458, 2106.761],
        "B02": [1503.317, 2141.107],
        "B03": [1718.197, 2038.973],
        "B04": [1853.910, 2134.138],
        "B05": [2199.100, 2085.321],
        "B06": [2779.975, 1889.926],
        "B07": [2987.011, 1820.257],
        "B08": [3083.234, 1871.918],
        "B8A": [3132.220, 1753.829],
        "B09": [3162.988, 1797.379],
        "B11": [2424.884, 1434.261],
        "B12": [1857.648, 1334.311],
    },
    "S2L1C": {
        "B01": [2357.089, 1624.683],
        "B02": [2137.385, 1675.806],
        "B03": [2018.788, 1557.708],
        "B04": [2082.986, 1833.702],
        "B05": [2295.651, 1823.738],
        "B06": [2854.537, 1733.977],
        "B07": [3122.849, 1732.131],
        "B08": [3040.560, 1679.732],
        "B8A": [3306.481, 1727.26],
        "B09": [1473.847, 1024.687],
        "B10": [506.070, 442.165],
        "B11": [2472.825, 1331.411],
        "B12": [1838.929, 1160.419],
    },
    "RGB": {
        "Red": [87.271, 58.767],
        "Green": [80.931, 47.663],
        "Blue": [66.667, 42.631],
    },
    "S1GRD": {
        "vv": [-12.599, 5.195],
        "vh": [-20.293, 5.890],
    },
    "S1RTC": {
        "vv": [-10.93, 4.391],
        "vh": [-17.329, 4.459],
    },
    "DEM": {
        "DEM": [670.665, 951.272],
    },
}


class Terramind(nn.Module):
    """OlmoEarth Pretrain wrapper for Terramind."""

    patch_size: int = 16
    image_resolution: int = 224
    supported_modalities: list[str] = [
        Modality.SENTINEL2_L2A.name,
        Modality.SENTINEL1.name,
    ]
    current_modalities: list[str] = []
    tm_modalities = {
        Modality.SENTINEL2_L2A.name: "S2L2A",
        Modality.SENTINEL1.name: "S1RTC",
        Modality.SENTINEL2.name: "S2L1C",
    }
    h_modalities = {v: k for k, v in tm_modalities.items()}
    supports_multiple_modalities_at_once = True

    def _prepare_stats(self) -> None:
        self.stats: dict = {}
        for modality in PRETRAINED_BANDS:
            means = [
                PRETRAINED_BANDS[modality][band][0]
                for band in PRETRAINED_BANDS[modality]
            ]
            stds = [
                PRETRAINED_BANDS[modality][band][1]
                for band in PRETRAINED_BANDS[modality]
            ]
            if modality in self.h_modalities:
                h_mod = self.h_modalities[modality]
                self.stats[h_mod] = {}
                self.stats[h_mod]["means"] = means
                self.stats[h_mod]["stds"] = stds

    def _check_modalities(self, modalities: list[str]) -> bool:
        return set(modalities) == set(self.current_modalities)

    def _init_model(self, size: str, supported_modalities: list[str]) -> None:
        modalities = [self.tm_modalities[m] for m in supported_modalities]
        if size == "base":
            self.model = BACKBONE_REGISTRY.build(
                "terramind_v1_base", modalities=modalities, pretrained=True
            )
        elif size == "large":
            self.model = BACKBONE_REGISTRY.build(
                "terramind_v1_large", modalities=modalities, pretrained=True
            )
        else:
            raise ValueError(f"Invalid model size: {size}")
        self.current_modalities = supported_modalities

    def __init__(
        self,
        size: str = "base",
        use_pretrained_normalizer: bool = True,
    ) -> None:
        """Initialize terramind model."""
        super().__init__()
        self.size = size
        if size not in TERRAMIND_SIZES:
            raise ValueError(f"Invalid model size: {size}")
        self._prepare_stats()
        self._init_model(self.size, self.supported_modalities)
        self.use_pretrained_normalizer = use_pretrained_normalizer

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
                data_i = data_i[:, HELIOS_TO_TERRAMIND_SENTINEL2_BANDORDER, :, :]

            if self.use_pretrained_normalizer:
                # Normalize
                for j in range(data_i.shape[1]):
                    data_i[:, j, :, :] = (
                        data_i[:, j, :, :] - self.stats[modality]["means"][j]
                    ) / self.stats[modality]["stds"][j]

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
        masked_olmoearth_sample: MaskedOlmoEarthSample,
    ) -> list[dict[str, torch.Tensor]]:
        """Prepare input for the Terramind model from MaskedOlmoEarthSample."""
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
            terramind_modality = self.tm_modalities[modality]
            for i, data_i in enumerate(processed_data):
                # start the list if it doesn't exist
                if i not in input_data_timesteps:
                    input_data_timesteps[i] = {}
                input_data_timesteps[i][terramind_modality] = data_i

        if not input_data_timesteps:
            raise ValueError("No valid modalities found for processing")
        return [input_data_timesteps[i] for i in sorted(input_data_timesteps.keys())]

    def forward(
        self,
        masked_olmoearth_sample: MaskedOlmoEarthSample,
        pooling: PoolingType = PoolingType.MEAN,
        spatial_pool: bool = False,
    ) -> torch.Tensor:
        """Forward pass through terramind model."""
        # Configure model modality

        modalities = masked_olmoearth_sample.modalities
        if not self._check_modalities(modalities):
            # self._init_model(self.size, modalities)
            pass

        # Prepare input
        per_timestep_inputs = self.prepare_input(masked_olmoearth_sample)

        output_features = []

        for data in per_timestep_inputs:
            timestep_output = self.model(data)[-1]
            if not spatial_pool:
                # TODO: maybe right?
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
class TerramindConfig(Config):
    """olmo_core style config for Terramind."""

    size: str = "base"
    use_pretrained_normalizer: bool = True

    def validate(self) -> None:
        """Validate the Terramind config."""
        if self.size not in TERRAMIND_SIZES:
            raise ValueError(f"Invalid model size: {self.size}")

    def build(self) -> Terramind:
        """Build the Terramind model."""
        return Terramind(
            size=self.size,
            use_pretrained_normalizer=self.use_pretrained_normalizer,
        )
