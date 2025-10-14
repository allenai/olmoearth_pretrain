"""Panopticon model https://github.com/Panopticon-FM/panopticon?tab=readme-ov-file ."""

import logging
import math
from dataclasses import dataclass
from importlib import resources

import torch
import torch.nn.functional as F
import yaml
from einops import rearrange, repeat
from olmo_core.config import Config
from torch import nn

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.nn.flexi_vit import PoolingType
from olmoearth_pretrain.train.masking import MaskedOlmoEarthSample

logger = logging.getLogger(__name__)


class Panopticon(nn.Module):
    """Class containing the Panopticon model that can ingest MaskedOlmoEarthSample objects."""

    patch_size: int = 14
    supported_modalities: list[str] = [
        Modality.SENTINEL2_L2A.name,
        Modality.LANDSAT.name,
        Modality.SENTINEL1.name,
    ]
    supports_multiple_modalities_at_once = True

    def __init__(
        self,
        torchhub_id: str = "panopticon_vitb14",
    ):
        """Initialize the Panopticon wrapper.

        Args:
            torchhub_id: The torch hub model ID for panopticon
        """
        super().__init__()
        # Load the panopticon model
        self._load_model(torchhub_id)

    def _load_model(self, torchhub_id: str) -> None:
        """Load the panopticon model from torch hub."""
        import time

        # Hack to get around https://discuss.pytorch.org/t/torch-hub-load-gives-httperror-rate-limit-exceeded/124769
        torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
        for attempt in range(2):
            try:
                self.model = torch.hub.load(
                    "panopticon-FM/panopticon",
                    torchhub_id,
                )
                break
            except Exception as e:
                logger.warning(
                    f"Error loading panopticon model: {e}. Retrying in 5 seconds..."
                )
                time.sleep(5)
        else:
            raise RuntimeError(
                f"Failed to load panopticon model {torchhub_id} after retrying."
            )

    def _process_modality_data(self, data: torch.Tensor) -> list[torch.Tensor]:
        """Process individual modality data.

        Args:
            data: Input tensor of shape [B, H, W, T, C]

        Returns:
            Processed tensor of shape [B, C*T, H, W]
        """
        # Rearrange from "b h w t c -> b (c t) h w" for DinoV2/Panopticon format
        t_dim = data.shape[3]

        # Get original dimensions
        original_height = data.shape[2]
        data_list = []
        for i in range(t_dim):
            data_i = rearrange(data[:, :, :, i, :], "b h w c -> b c h w")

            new_height = self.patch_size if original_height == 1 else 224

            data_i = F.interpolate(
                data_i,
                size=(new_height, new_height),
                mode="bilinear",
                align_corners=False,
            )
            data_list.append(data_i)
        return data_list

    def _create_channel_ids(
        self, modality: str, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        """Create channel IDs for the panopticon model."""
        # Bands are in the EVAL_TO_HELIOS_S2_BANDS order so we need to use that to pull the information from the yaml files
        if modality == "sentinel2_l2a":
            modality_yaml_name = "sentinel2"
        elif modality == "landsat":
            modality_yaml_name = "landsat8"
        else:
            modality_yaml_name = modality
        with resources.open_text(
            "olmoearth_pretrain.evals.models.panopticon.sensors",
            f"{modality_yaml_name}.yaml",
        ) as f:
            sensor_config = yaml.safe_load(f)
        modality_spec = Modality.get(modality)
        # Data is prepared in helios band order so we need to tell panopticon whcich band it is
        chn_ids = []
        for band in modality_spec.band_order:
            if band == "B10" and modality == "sentinel2_l2a":
                # skipping B10 band for this eval I think because the helios dataloader skips it
                # is this true for everything or for geobench only?
                continue
            band = band.upper()
            chn_ids.append(sensor_config["bands"][band]["gaussian"]["mu"])
        chn_ids = torch.tensor(chn_ids, dtype=torch.float32, device=device)
        chn_ids = repeat(chn_ids, "c -> b c", b=batch_size)
        return chn_ids

    def prepare_input(
        self, masked_olmoearth_sample: MaskedOlmoEarthSample
    ) -> list[dict[str, torch.Tensor]]:
        """Prepare input for the panopticon model from MaskedOlmoEarthSample."""
        # Process each modality
        input_data_timesteps: dict[int, list[torch.Tensor]] = {}
        channel_ids_list: list[torch.Tensor] = []
        for modality in masked_olmoearth_sample.modalities:
            if modality in ["timestamps", "latlon"]:
                continue  # Skip non-image modalities
            data = getattr(masked_olmoearth_sample, modality)
            device = data.device
            if data is None:
                continue

            # Process the modality data
            processed_data = self._process_modality_data(data)
            for i, data_i in enumerate(processed_data):
                # start the list if it doesn't exist
                if i not in input_data_timesteps:
                    input_data_timesteps[i] = []
                input_data_timesteps[i].append(data_i)
            batch_size = processed_data[0].shape[0]
            # I need to convert the helios channel ordering to get the right panopticon channel value
            chn_ids = self._create_channel_ids(modality, batch_size, device)
            channel_ids_list.append(chn_ids)

        if not input_data_timesteps:
            raise ValueError("No valid modalities found for processing")
        # chn ids are shared across all the timesteps so we cna concatenate just once
        chn_ids = torch.cat(channel_ids_list, dim=1)
        per_timestep_panopticon_inputs = []
        for i, input_data_i in input_data_timesteps.items():
            # Concatenate all modality data along channel dimension
            concatenated_imgs = torch.cat(input_data_i, dim=1)
            panopticon_input = {
                "imgs": concatenated_imgs,
                "chn_ids": chn_ids,
            }
            per_timestep_panopticon_inputs.append(panopticon_input)
        # I want to return a list of panopticon inputs, one for each timestep
        return per_timestep_panopticon_inputs

    def forward(
        self,
        masked_olmoearth_sample: MaskedOlmoEarthSample,
        pooling: PoolingType = PoolingType.MEAN,
    ) -> torch.Tensor:
        """Forward pass through the panopticon model."""
        # Prepare input
        per_timestep_panopticon_inputs = self.prepare_input(masked_olmoearth_sample)
        # potentially will need to add a flag for segmentation
        output_features = []
        for panopticon_input in per_timestep_panopticon_inputs:
            timestep_output = self.model(panopticon_input)
            output_features.append(timestep_output.unsqueeze(0))
        # stack in the timestep dimension and take the mean or maybe the max?
        if pooling == PoolingType.MEAN:
            output_features = torch.cat(output_features, dim=0).mean(dim=0)
        elif pooling == PoolingType.MAX:
            output_features = torch.max(torch.cat(output_features, dim=0), dim=0)[0]
        return output_features

    def forward_features(
        self,
        masked_olmoearth_sample: MaskedOlmoEarthSample,
        pooling: PoolingType = PoolingType.MEAN,
    ) -> torch.Tensor:
        """Forward pass through the panopticon model."""
        per_timestep_panopticon_inputs = self.prepare_input(masked_olmoearth_sample)
        output_features = []
        for panopticon_input in per_timestep_panopticon_inputs:
            timestep_output = self.model.forward_features(panopticon_input)[
                "x_norm_patchtokens"
            ]
            num_tokens = timestep_output.shape[1]
            height = int(math.sqrt(num_tokens))
            timestep_output = rearrange(
                timestep_output, "b (h w) d -> b h w d", h=height, w=height
            )
            output_features.append(timestep_output.unsqueeze(0))
        # stack in the timestep dimension and take the mean or maybe the max?
        if pooling == PoolingType.MEAN:
            output_features = torch.cat(output_features, dim=0).mean(dim=0)
        elif pooling == PoolingType.MAX:
            output_features = torch.max(torch.cat(output_features, dim=0), dim=0)[0]
        return output_features

    def __call__(
        self,
        masked_olmoearth_sample: MaskedOlmoEarthSample,
        pooling: PoolingType = PoolingType.MEAN,
    ) -> torch.Tensor:
        """Make the wrapper callable."""
        return self.forward(masked_olmoearth_sample, pooling)


@dataclass
class PanopticonConfig(Config):
    """olmo_core style config for PanopticonWrapper."""

    torchhub_id: str = "panopticon_vitb14"

    def build(self) -> Panopticon:
        """Build the Panopticon model."""
        return Panopticon(
            torchhub_id=self.torchhub_id,
        )
