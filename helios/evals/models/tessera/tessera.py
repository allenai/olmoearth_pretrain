"""Tessera foundation model https://github.com/ucam-eo/tessera ."""

import logging
import math
import warnings
from dataclasses import dataclass
from typing import Optional

import torch
from einops import rearrange
from olmo_core.config import Config
from torch import nn
from torchvision import transforms

from helios.data.constants import Modality
from helios.nn.flexihelios import PoolingType
from helios.train.masking import MaskedHeliosSample

logger = logging.getLogger(__name__)

# Tessera typically expects Sentinel-2 data normalized to [0,1] or similar
# These values may need adjustment based on the actual Tessera implementation
TESSERA_DEFAULT_MEAN = (0.485, 0.456, 0.406)  # Placeholder values
TESSERA_DEFAULT_STD = (0.229, 0.224, 0.225)  # Placeholder values


def make_tessera_normalize_transform() -> transforms.Normalize:
    """Make normalize transform for Tessera model."""
    normalize = transforms.Normalize(
        mean=TESSERA_DEFAULT_MEAN,
        std=TESSERA_DEFAULT_STD,
    )
    return normalize


def make_tessera_resize_transform(resize_size: int) -> transforms.Resize:
    """Make resize transform for Tessera model."""
    resize = transforms.Resize((resize_size, resize_size), antialias=True)
    return resize


# Tessera expects Sentinel-2 bands - adjust based on actual requirements
# Common Sentinel-2 bands: B02 (Blue), B03 (Green), B04 (Red), B08 (NIR)
# B05, B06, B07, B8A, B11, B12 (SWIR bands)
HELIOS_SENTINEL2_TESSERA_BANDS = [
    Modality.SENTINEL2_L2A.band_order.index(b) 
    for b in ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12"]
    if b in Modality.SENTINEL2_L2A.band_order
]


class Tessera(nn.Module):
    """Wrapper for the Tessera model that can ingest MaskedHeliosSample objects."""

    patch_size: int = 16  # Common patch size, adjust based on actual model
    base_resize: int = 224  # Common input size, adjust based on actual model
    supported_modalities: list[str] = [
        Modality.SENTINEL2_L2A.name,
    ]

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        apply_normalization: bool = True,
        input_size: int = 224,
    ):
        """Initialize the Tessera wrapper.

        Args:
            checkpoint_path: Path to Tessera model checkpoint (optional)
            apply_normalization: Whether to apply normalization to input data
            input_size: Input image size expected by the model
        """
        super().__init__()
        self.apply_normalization = apply_normalization
        self.input_size = input_size
        self.checkpoint_path = checkpoint_path
        
        if self.apply_normalization:
            logger.info("Applying Tessera normalization to input data")
            self.normalize_transform = make_tessera_normalize_transform()
        
        # Initialize model - placeholder for now
        self._load_model(checkpoint_path)

    def _load_model(self, checkpoint_path: Optional[str] = None) -> None:
        """Load the Tessera model."""
        if checkpoint_path is None:
            logger.warning("No checkpoint provided for Tessera model. Using placeholder model.")
            # Create a placeholder model for now
            self.model = self._create_placeholder_model()
        else:
            logger.info(f"Loading Tessera model from {checkpoint_path}")
            try:
                # TODO: Implement actual Tessera model loading
                # This will be updated when the actual Tessera implementation is available
                self.model = self._load_tessera_checkpoint(checkpoint_path)
            except Exception as e:
                logger.warning(f"Failed to load Tessera checkpoint: {e}. Using placeholder.")
                self.model = self._create_placeholder_model()

    def _create_placeholder_model(self) -> nn.Module:
        """Create a placeholder model for testing purposes."""
        # Simple placeholder - just a linear layer that outputs fixed size features
        return nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(len(HELIOS_SENTINEL2_TESSERA_BANDS), 768)  # Common feature size
        )

    def _load_tessera_checkpoint(self, checkpoint_path: str) -> nn.Module:
        """Load actual Tessera model from checkpoint."""
        # TODO: Implement actual loading logic based on Tessera's checkpoint format
        # This is a placeholder that will need to be updated
        raise NotImplementedError("Tessera checkpoint loading not yet implemented")

    def _process_modality_data(
        self,
        data: torch.Tensor,
        modality: str,
    ) -> list[torch.Tensor]:
        """Process individual modality data for Tessera."""
        # Rearrange from "b h w t c -> b (c t) h w" for model format
        t_dim = data.shape[3]
        original_height = data.shape[2]
        
        data_list = []
        for i in range(t_dim):
            data_i = rearrange(data[:, :, :, i, :], "b h w c -> b c h w")
            
            # Select appropriate bands for Tessera
            if modality == "sentinel2_l2a":
                data_i = data_i[:, HELIOS_SENTINEL2_TESSERA_BANDS, :, :]
            
            # Resize if necessary
            if original_height != self.input_size:
                resize_transform = make_tessera_resize_transform(self.input_size)
                data_i = resize_transform(data_i)
            
            # Apply normalization if enabled
            if self.apply_normalization:
                data_i = self.normalize_transform(data_i)
            
            data_list.append(data_i)
        
        return data_list

    def prepare_input(
        self,
        masked_helios_sample: MaskedHeliosSample,
    ) -> list[torch.Tensor]:
        """Prepare input for the Tessera model from MaskedHeliosSample."""
        input_data_timesteps: dict[int, list[torch.Tensor]] = {}
        
        for modality in masked_helios_sample.modalities:
            if modality not in self.supported_modalities:
                logger.warning(
                    f"Skipping modality {modality} as it is not supported by Tessera. "
                    f"Supported modalities: {self.supported_modalities}"
                )
                continue

            data = getattr(masked_helios_sample, modality)
            if data is None:
                continue

            # Process the modality data
            processed_data = self._process_modality_data(data, modality)
            for i, data_i in enumerate(processed_data):
                if i not in input_data_timesteps:
                    input_data_timesteps[i] = []
                input_data_timesteps[i].append(data_i)

        if not input_data_timesteps:
            raise ValueError("No valid modalities found for Tessera processing")
        
        per_timestep_inputs = []
        for i, input_data_i in input_data_timesteps.items():
            # Concatenate all modality data along channel dimension
            concatenated_imgs = torch.cat(input_data_i, dim=1)
            per_timestep_inputs.append(concatenated_imgs)
        
        return per_timestep_inputs

    def forward(
        self,
        masked_helios_sample: MaskedHeliosSample,
        pooling: PoolingType = PoolingType.MEAN,
    ) -> torch.Tensor:
        """Forward pass through Tessera model for classification."""
        # Prepare input
        per_timestep_inputs = self.prepare_input(masked_helios_sample)
        
        # Process each timestep
        output_features = []
        for data in per_timestep_inputs:
            # Forward pass through Tessera model
            timestep_output = self.model(data)
            output_features.append(timestep_output.unsqueeze(0))
        
        # Aggregate across timesteps
        if pooling == PoolingType.MEAN:
            output_features = torch.cat(output_features, dim=0).mean(dim=0)
        elif pooling == PoolingType.MAX:
            output_features = torch.max(torch.cat(output_features, dim=0), dim=0)[0]
        else:
            raise ValueError(f"Unsupported pooling type: {pooling}")
        
        return output_features

    def forward_features(
        self,
        masked_helios_sample: MaskedHeliosSample,
        pooling: PoolingType = PoolingType.MEAN,
    ) -> torch.Tensor:
        """Forward pass through Tessera model for segmentation."""
        # For now, use the same forward pass as classification
        # This may need to be updated based on actual Tessera architecture
        per_timestep_inputs = self.prepare_input(masked_helios_sample)
        
        output_features = []
        for data in per_timestep_inputs:
            # Get feature maps from model
            # TODO: Update this when we have the actual Tessera implementation
            features = self.model(data)
            
            # Reshape for segmentation if needed
            if len(features.shape) == 2:  # (batch, features)
                # Placeholder reshaping - adjust based on actual model output
                batch_size = features.shape[0]
                feat_dim = features.shape[1]
                h = w = int(math.sqrt(self.input_size // self.patch_size))
                features = features.view(batch_size, h, w, feat_dim)
            
            output_features.append(features.unsqueeze(0))
        
        # Aggregate across timesteps
        if pooling == PoolingType.MEAN:
            output_features = torch.cat(output_features, dim=0).mean(dim=0)
        elif pooling == PoolingType.MAX:
            output_features = torch.max(torch.cat(output_features, dim=0), dim=0)[0]
        else:
            raise ValueError(f"Unsupported pooling type: {pooling}")
        
        return output_features

    def __call__(
        self,
        masked_helios_sample: MaskedHeliosSample,
        pooling: PoolingType = PoolingType.MEAN,
    ) -> torch.Tensor:
        """Make the wrapper callable."""
        return self.forward(masked_helios_sample, pooling)


@dataclass
class TesseraConfig(Config):
    """olmo_core style config for Tessera."""

    checkpoint_path: Optional[str] = None
    apply_normalization: bool = True
    input_size: int = 224

    def build(self) -> "Tessera":
        """Build the Tessera model from this config."""
        return Tessera(
            checkpoint_path=self.checkpoint_path,
            apply_normalization=self.apply_normalization,
            input_size=self.input_size,
        )