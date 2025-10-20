"""OlmoEarth Pretrain wrapper for clay."""

import logging
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import yaml
from claymodel.module import ClayMAEModule
from einops import rearrange
from olmo_core.config import Config
from torch import nn

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.nn.flexi_vit import PoolingType
from olmoearth_pretrain.train.masking import MaskedOlmoEarthSample

logger = logging.getLogger(__name__)
CLAY_SENTINEL2_BANDS = [
    Modality.SENTINEL2_L2A.band_order.index(b)
    for b in [
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
]

CLAY_LANDSAT_BANDS = [
    Modality.LANDSAT.band_order.index(b)
    for b in [
        "B4",
        "B3",
        "B2",
        "B5",
        "B6",
        "B7",
    ]
]


class Clay(nn.Module):
    """Class containing the Clay model that can ingest MaskedOlmoEarthSample objects."""

    patch_size: int = 8
    image_resolution: int = 128
    use_cls_token: bool = True
    supported_modalities: list[str] = [
        Modality.SENTINEL2_L2A.name,
        Modality.SENTINEL1.name,
        Modality.LANDSAT.name,
    ]
    clay_modality_names: dict = {
        Modality.SENTINEL2_L2A.name: "sentinel-2-l2a",
        Modality.SENTINEL1.name: "sentinel-1-rtc",
        Modality.LANDSAT.name: "landsat-c2l1",
    }
    supports_multiple_modalities_at_once = True

    def _load_model(self, size: str, path: str, metadata: str) -> nn.Module:
        if size == "large":
            return ClayMAEModule.load_from_checkpoint(
                checkpoint_path=path,
                model_size="large",
                metadata_path=metadata,
                dolls=[16, 32, 64, 128, 256, 768, 1024],
                doll_weights=[1, 1, 1, 1, 1, 1, 1],
                mask_ratio=0.0,
                shuffle=False,
            )
        elif size == "base":
            raise ValueError("base size doesn't work in v1.5")
            return ClayMAEModule.load_from_checkpoint(
                checkpoint_path=path,
                model_size="base",
                metadata_path=metadata,
                dolls=[16, 32, 64, 128, 256, 768],
                doll_weights=[1, 1, 1, 1, 1, 1],
                mask_ratio=0.0,
                shuffle=False,
            )
        else:
            raise ValueError(f"No model size {size}")

    def __init__(
        self,
        size: str = "large",
        load_path: str = "/weka/dfive-default/helios/models/clay/clay-v1.5.ckpt",
        metadata_path: str = "olmoearth_pretrain/evals/models/clay/metadata.yaml",
        use_pretrained_normalizer: bool = True,
    ):
        """Initialize the Clay wrapper.

        Args:
            size: The model size
            load_path: The path to load from
            metadata_path: The path for metadata yaml
            use_pretrained_normalizer: Use Clay's metadata for saved mean/std
        """
        super().__init__()
        with open(metadata_path) as f:
            self.metadata = yaml.safe_load(f)
        self.model = self._load_model(size, load_path, metadata_path).model
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

        sensor = self.clay_modality_names[modality]
        sensor_meta = self.metadata[sensor]
        data_list = []

        means = torch.tensor(
            [sensor_meta["bands"]["mean"][band] for band in sensor_meta["band_order"]]
        )
        stds = torch.tensor(
            [sensor_meta["bands"]["std"][band] for band in sensor_meta["band_order"]]
        )
        means = means.view(1, -1, 1, 1).to(device=data.device)
        stds = stds.view(1, -1, 1, 1).to(device=data.device)

        for i in range(t_dim):
            data_i = rearrange(data[:, :, :, i, :], "b h w c -> b c h w")

            # Rearrange sen2 data
            if modality == Modality.SENTINEL2_L2A.name:
                data_i = data_i[:, CLAY_SENTINEL2_BANDS, :, :]

            # Rearrange landsat data
            if modality == Modality.LANDSAT.name:
                data_i = data_i[:, CLAY_LANDSAT_BANDS, :, :]

            if self.use_pretrained_normalizer:
                # Normalize data
                data_i = (data_i - means) / stds

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
        """Prepare input for the model from MaskedOlmoEarthSample."""
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

            sensor = self.clay_modality_names[modality]

            # Process the modality data
            processed_data = self._process_modality_data(data, modality)
            for i, data_i in enumerate(processed_data):
                # start the list if it doesn't exist
                if i not in input_data_timesteps:
                    input_data_timesteps[i] = {}
                input_data_timesteps[i][sensor] = data_i

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
        # Prepare input
        per_timestep_inputs = self.prepare_input(masked_olmoearth_sample)
        output_features = []

        for data in per_timestep_inputs:
            embedding_list = []
            for sensor in data.keys():
                sensor_data = data[sensor]
                device = sensor_data.device

                wavelengths = []
                for band in self.metadata[sensor]["band_order"]:
                    wavelengths.append(
                        self.metadata[sensor]["bands"]["wavelength"][band] * 1000
                    )  # Convert to nm

                cube = {
                    "platform": sensor,
                    "time": torch.zeros(sensor_data.shape[0], 4).to(device=device),
                    "latlon": torch.zeros(sensor_data.shape[0], 4).to(device=device),
                    "pixels": sensor_data,
                    "waves": torch.tensor(wavelengths),
                    "gsd": torch.tensor(self.metadata[sensor]["gsd"]),
                }
                embeddings, *_ = self.model.encoder(cube)
                if self.use_cls_token and not spatial_pool:
                    embeddings = embeddings[:, :1, :]  # cls_token
                else:
                    embeddings = embeddings[:, 1:, :]  # exclude cls_token
                embedding_list.append(embeddings)
            timestep_output = torch.stack(embedding_list, dim=0).mean(dim=0)
            if not spatial_pool:
                timestep_output = timestep_output.mean(dim=1)
            else:
                side = math.isqrt(timestep_output.shape[1])
                timestep_output = rearrange(
                    timestep_output, "b (h w) c -> b h w c", h=side, w=side
                )
            output_features.append(timestep_output)
        # stack in the timestep dimension and take the mean or maybe the max?
        if pooling == PoolingType.MEAN:
            output_features = torch.stack(output_features, dim=0).mean(dim=0)
        elif pooling == PoolingType.MAX:
            output_features = torch.stack(output_features, dim=0).max(dim=0)[0]
        return output_features


@dataclass
class ClayConfig(Config):
    """olmo_core style config for ClayWrapper."""

    size: str = "large"
    load_path: str = "/weka/dfive-default/helios/models/clay/clay-v1.5.ckpt"
    metadata_path: str = "olmoearth_pretrain/evals/models/clay/metadata.yaml"
    use_pretrained_normalizer: bool = True

    def build(self) -> Clay:
        """Build the Clay model."""
        return Clay(
            size=self.size,
            load_path=self.load_path,
            metadata_path=self.metadata_path,
            use_pretrained_normalizer=self.use_pretrained_normalizer,
        )
