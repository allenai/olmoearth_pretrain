"""Wrapper for Running Evals on Panopticon"""

import logging
import torch
import torch.nn.functional as F
from einops import rearrange
from torchvision import transforms
from torch import nn

from helios.train.masking import MaskedHeliosSample
from helios.data.constants import Modality

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
        # Hack to get around https://discuss.pytorch.org/t/torch-hub-load-gives-httperror-rate-limit-exceeded/124769
        torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
        self.model = torch.hub.load(
            'panopticon-FM/panopticon',
            torchhub_id,
        )
        self.model = self.model.eval()
        self.model = self.model.to(device=self.device)
        logger.info(f"Loaded panopticon model {torchhub_id} on device {self.device}")

    def _process_modality_data(self, data: torch.Tensor) -> torch.Tensor:
        """Process individual modality data.

        Args:
            data: Input tensor of shape [B, H, W, T, C]

        Returns:
            Processed tensor of shape [B, C*T, H, W]
        """
        # Rearrange from "b h w t c -> b (c t) h w" for DinoV2/Panopticon format
        data = rearrange(data, "b h w t c -> b (c t) h w")

        # Get original dimensions
        original_height = data.shape[2]

        # Resize the image based on patch size
        if original_height < 224:
            image_size = 224
            data = F.interpolate(
                data,
                size=(image_size, image_size),
                mode="bilinear",
                align_corners=False
            )
        else:
            # Move to nearest multiple of patch_size
            image_size = ((original_height // self.patch_size) - 1) * self.patch_size
            data = F.interpolate(
                data,
                size=(image_size, image_size),
                mode="bilinear",
                align_corners=False
            )

        return data

    def _create_channel_ids(self, total_channels: int, batch_size: int) -> torch.Tensor:
        """Create channel IDs for the panopticon model.

        Args:
            total_channels: Total number of channels across all modalities
            batch_size: Batch size

        Returns:
            Channel IDs tensor of shape [1, total_channels] or [batch_size, total_channels]
        """
        # Create sequential channel IDs (0, 1, 2, ..., total_channels-1)
        # This is a simple approach - could be enhanced with actual wavelength information
        chn_ids = torch.arange(total_channels, dtype=torch.float32, device=self.device)
        chn_ids = chn_ids.unsqueeze(0)  # Shape: [1, total_channels]

        return chn_ids

    def prepare_input(self, masked_helios_sample: MaskedHeliosSample) -> dict[str, torch.Tensor]:
        """Prepare input for the panopticon model from MaskedHeliosSample.

        Args:
            masked_helios_sample: Input MaskedHeliosSample object

        Returns:
            Dictionary with 'imgs' and 'chn_ids' keys for panopticon model
        """
        # Process each modality
        input_data = []
        total_channels = 0

        for modality in masked_helios_sample.modalities:
            if modality in ["timestamps", "latlon"]:
                continue  # Skip non-image modalities

            data = getattr(masked_helios_sample, modality)
            if data is None:
                continue

            # Process the modality data
            processed_data = self._process_modality_data(data)
            input_data.append(processed_data)
            total_channels += processed_data.shape[1]  # Accumulate channel count

            logger.info(f"Processed {modality}: {processed_data.shape}")

        if not input_data:
            raise ValueError("No valid modalities found for processing")

        # Concatenate all modality data along channel dimension
        concatenated_imgs = torch.cat(input_data, dim=1)
        batch_size = concatenated_imgs.shape[0]

        # Create channel IDs
        chn_ids = self._create_channel_ids(total_channels, batch_size)

        panopticon_input = {
            "imgs": concatenated_imgs,
            "chn_ids": chn_ids,
        }

        logger.info(f"Final input shape - imgs: {concatenated_imgs.shape}, chn_ids: {chn_ids.shape}")

        return panopticon_input

    def forward(self, masked_helios_sample: MaskedHeliosSample) -> torch.Tensor:
        """Forward pass through the panopticon model.

        Args:
            masked_helios_sample: Input MaskedHeliosSample object

        Returns:
            Model embeddings
        """
        # Prepare input
        panopticon_input = self.prepare_input(masked_helios_sample)
        # potentially will need to add a flag for segmentation
        embeddings = self.model(panopticon_input)
        logger.info(f"Model output shape: {embeddings.shape}")
        return embeddings

    def __call__(self, masked_helios_sample: MaskedHeliosSample) -> torch.Tensor:
        """Make the wrapper callable."""
        return self.forward(masked_helios_sample)