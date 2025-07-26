"""Wrapper for Running Evals on Panopticon"""

import logging
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torchvision import transforms
from torch import nn
import yaml
from helios.train.masking import MaskedHeliosSample
from helios.nn.flexihelios import PoolingType
from helios.data.constants import Modality
import math
from olmo_core.config import Config
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class PanopticonWrapper(nn.Module):
    """Wrapper for the Panopticon model that can ingest MaskedHeliosSample objects."""

    def __init__(
        self,
        torchhub_id: str = "panopticon_vitb14",
        patch_size: int = 14,
        device: str = "cuda"
    ):
        """Initialize the Panopticon wrapper.

        Args:
            torchhub_id: The torch hub model ID for panopticon
            patch_size: Patch size for the vision transformer (default 14)
            device: Device to run the model on
        """
        super().__init__()
        self.patch_size = patch_size
        self.device = device

        # Load the panopticon model
        self._load_model(torchhub_id)


    def _load_model(self, torchhub_id: str):
        """Load the panopticon model from torch hub."""
        import time
        # Hack to get around https://discuss.pytorch.org/t/torch-hub-load-gives-httperror-rate-limit-exceeded/124769
        torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
        for attempt in range(2):
            try:
                self.model = torch.hub.load(
                    'panopticon-FM/panopticon',
                    torchhub_id,
                )
                break
            except Exception as e:
                logger.warning(f"Error loading panopticon model: {e}. Retrying in 5 seconds...")
                time.sleep(5)
        else:
            raise RuntimeError(f"Failed to load panopticon model {torchhub_id} after retrying.")
        self.model = self.model.eval()
        self.model = self.model.to(device=self.device)
        logger.info(f"Loaded panopticon model {torchhub_id} on device {self.device}")

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

            if original_height == 1:
                # For pixel time series resize to single patch
                data_i = F.interpolate(
                    data_i,
                    size=(self.patch_size, self.patch_size),
                    mode="bilinear",
                    align_corners=False
                )
            else:
                # Resize the image to the Panopticon pre-training input size
                image_size = 224
                data_i = F.interpolate(
                    data_i,
                    size=(image_size, image_size),
                    mode="bilinear",
                    align_corners=False
                )
            data_list.append(data_i)
        return data_list

    def _create_channel_ids(self, modality: str, batch_size: int) -> torch.Tensor:
        """Create channel IDs for the panopticon model.

        Args:
            total_channels: Total number of channels across all modalities
            batch_size: Batch size

        Returns:
            Channel IDs tensor of shape [1, total_channels] or [batch_size, total_channels]
        """
        # Bands are in the EVAL_TO_HELIOS_S2_BANDS order so we need to use that to pull the information from the yaml files
        if modality == "sentinel2_l2a":
            modality_yaml_name = "sentinel2"
        elif modality == "landsat":
            modality_yaml_name = "landsat8"
        else:
            modality_yaml_name = modality
        with open(f"./helios/evals/panopticon/sensors/{modality_yaml_name}.yaml", "r") as f:
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
        chn_ids = torch.tensor(chn_ids, dtype=torch.float32, device=self.device)
        chn_ids = repeat(chn_ids, "c -> b c", b=batch_size)
        return chn_ids

    def prepare_input(self, masked_helios_sample: MaskedHeliosSample) -> dict[str, torch.Tensor]:
        """Prepare input for the panopticon model from MaskedHeliosSample.

        Args:
            masked_helios_sample: Input MaskedHeliosSample object

        Returns:
            Dictionary with 'imgs' and 'chn_ids' keys for panopticon model
        """
        # Process each modality
        input_data_timesteps = {}
        channel_ids_list = []
        for modality in masked_helios_sample.modalities:
            if modality in ["timestamps", "latlon"]:
                continue  # Skip non-image modalities

            data = getattr(masked_helios_sample, modality)

            print(f"Modality: {modality}, data: shape {data.shape}")
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
            chn_ids = self._create_channel_ids(modality, batch_size)
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

    def forward(self, masked_helios_sample: MaskedHeliosSample, pooling: PoolingType = PoolingType.MEAN) -> torch.Tensor:
        """Forward pass through the panopticon model.

        Args:
            masked_helios_sample: Input MaskedHeliosSample object

        Returns:
            Model embeddings
        """
        # Prepare input
        per_timestep_panopticon_inputs = self.prepare_input(masked_helios_sample)
        # potentially will need to add a flag for segmentation
        output_features = []
        for panopticon_input in per_timestep_panopticon_inputs:
            timestep_output = self.model(panopticon_input)
            print(f"timestep_output shape: {timestep_output.shape}")
            output_features.append(timestep_output.unsqueeze(0))
        # stack in the timestep dimension and take the mean or maybe the max?
        if pooling == PoolingType.MEAN:
            output_features = torch.cat(output_features, dim=0).mean(dim=0)
        elif pooling == PoolingType.MAX:
            output_features = torch.max(torch.cat(output_features, dim=0), dim=0)[0]
        print(f"output_features shape: {output_features}")
        # Do we need this to work for both single pixel and full images
        return output_features

    def get_intermediate_layers(self, masked_helios_sample: MaskedHeliosSample, n: list[int], return_class_token: bool = False) -> list[torch.Tensor]:
        """Get intermediate layers from the panopticon model.

        Args:
            masked_helios_sample: Input MaskedHeliosSample object
            n: List of layer indices to return
            return_class_token: Whether to return the class token

        Returns:
            List of intermediate layers
        """
        # Currently seems to not be working
        print(f" is chunked blocks: {self.model.chunked_blocks}")
        panopticon_input = self.prepare_input(masked_helios_sample)
        embeddings = self.model.get_intermediate_layers(panopticon_input, n, return_class_token)
        return embeddings

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: int | list[int] = 1,  # Layers or n last layers to take
        reshape: bool = False,
        return_class_token: bool = False,
        norm=True,
    ) -> torch.Tensor | tuple[torch.Tensor]:
        if self.model.chunked_blocks:
            outputs = self.model._get_intermediate_layers_chunked(x, n)
        else:
            outputs = self.model._get_intermediate_layers_not_chunked(x, n)
        if norm:
            outputs = [self.model.norm(out) for out in outputs]
        print(f"outputs: {outputs}")
        class_tokens = [out[:, 0] for out in outputs]
        outputs = [out[:, 1 + self.model.num_register_tokens :] for out in outputs]
        if reshape:
            B, _, w, h = x.shape
            outputs = [
                out.reshape(B, w // self.patch_size, h // self.patch_size, -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]
        if return_class_token:
            return tuple(zip(outputs, class_tokens))
        return tuple(outputs)

    # TODO: add a Temporal poolin type option
    def forward_features(self, masked_helios_sample: MaskedHeliosSample, pooling: PoolingType = PoolingType.MEAN) -> torch.Tensor:
        """Forward pass through the panopticon model.

        Args:
            x_dict: Input dictionary with 'imgs' and 'chn_ids' keys

        Returns:
        """
        # supports multi-timestep input single timestep output
        per_timestep_panopticon_inputs = self.prepare_input(masked_helios_sample)
        output_features = []
        for panopticon_input in per_timestep_panopticon_inputs:
            timestep_output = self.model.forward_features(panopticon_input)["x_norm_patchtokens"]
            num_tokens = timestep_output.shape[1]
            height = int(math.sqrt(num_tokens))
            timestep_output = rearrange(timestep_output, "b (h w) d -> b h w d", h=height, w=height)
            output_features.append(timestep_output.unsqueeze(0))
        # stack in the timestep dimension and take the mean or maybe the max?
        if pooling == PoolingType.MEAN:
            output_features = torch.cat(output_features, dim=0).mean(dim=0)
        elif pooling == PoolingType.MAX:
            output_features = torch.max(torch.cat(output_features, dim=0), dim=0)[0]
        return output_features



    def __call__(self, masked_helios_sample: MaskedHeliosSample) -> torch.Tensor:
        """Make the wrapper callable."""
        return self.forward(masked_helios_sample)


@dataclass
class PanopticonConfig(Config):
    """olmo_core style config for PanopticonWrapper."""
    torchhub_id: str = "panopticon_vitb14"
    patch_size: int = 14
    device: str = "cuda"
