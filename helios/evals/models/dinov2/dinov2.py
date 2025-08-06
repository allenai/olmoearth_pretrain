"""DINOv2 model https://github.com/facebookresearch/dinov2 ."""

import logging
import math
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from einops import rearrange
from olmo_core.config import Config
from torch import nn
from torchvision import transforms

from helios.data.constants import Modality
from helios.nn.flexihelios import PoolingType
from helios.train.masking import MaskedHeliosSample

logger = logging.getLogger(__name__)
# Use timm's names
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def make_normalize_transform(
    mean: tuple[float, float, float] = IMAGENET_DEFAULT_MEAN,
    std: tuple[float, float, float] = IMAGENET_DEFAULT_STD,
) -> transforms.Normalize:
    """Make a normalize transform."""
    return transforms.Normalize(mean=mean, std=std)


# DinoV2 Expects bands ordered as R, G, B
HELIOS_SENTINEL2_RGB_BANDS = [
    Modality.SENTINEL2_L2A.band_order.index(b) for b in ["B04", "B03", "B02"]
]
HELIOS_LANDSAT_RGB_BANDS = [
    Modality.LANDSAT.band_order.index(b) for b in ["B4", "B3", "B2"]
]


class DINOv2(nn.Module):
    """Wrapper for the dinov2 model that can ingest MaskedHeliosSample objects."""

    patch_size: int = 14
    supported_modalities: list[str] = [
        Modality.SENTINEL2_L2A.name,
        Modality.LANDSAT.name,
    ]

    def __init__(
        self,
        torchhub_id: str = "dinov2_vitb14",
        use_cls_token: bool = False,
        apply_imagenet_normalization: bool = False,
    ):
        """Initialize the dinov2 wrapper.

        Args:
            torchhub_id: The torch hub model ID for dinov2
            use_cls_token: Whether to use the cls token (default False)
            apply_imagenet_normalization: Whether to apply imagenet normalization to the input data (default False)
        """
        super().__init__()
        self.use_cls_token = use_cls_token
        self.apply_imagenet_normalization = apply_imagenet_normalization
        if self.apply_imagenet_normalization:
            logger.warning(
                "Applying imagenet normalization to the input data. Make sure other normalization is not applied."
            )
        # Load the model
        self._load_model(torchhub_id)

    def _load_model(self, torchhub_id: str) -> None:
        """Load the dinov2 model from torch hub."""
        # Hack to get around https://discuss.pytorch.org/t/torch-hub-load-gives-httperror-rate-limit-exceeded/124769
        torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
        for attempt in range(2):
            try:
                self.model = torch.hub.load(
                    "facebookresearch/dinov2",
                    torchhub_id,
                )
                break
            except Exception as e:
                logger.warning(f"Error loading  model: {e}. Retrying in 5 seconds...")
                time.sleep(5)
        else:
            raise RuntimeError(
                f"Failed to load dinov2 model {torchhub_id} after retrying."
            )

    def _process_modality_data(
        self,
        data: torch.Tensor,
        modality: str,
    ) -> list[torch.Tensor]:
        """Process individual modality data."""
        # Rearrange from "b h w t c -> b (c t) h w" for DinoV2/dinov2 format
        t_dim = data.shape[3]

        # Get original dimensions
        original_height = data.shape[2]
        data_list = []
        for i in range(t_dim):
            data_i = rearrange(data[:, :, :, i, :], "b h w c -> b c h w")
            # Subset to RGB channels
            if modality == "sentinel2_l2a":
                data_i = data_i[:, HELIOS_SENTINEL2_RGB_BANDS, :, :]
            elif modality == "landsat":
                data_i = data_i[:, HELIOS_LANDSAT_RGB_BANDS, :, :]

            new_height = self.patch_size if original_height == 1 else 224

            data_i = F.interpolate(
                data_i,
                size=(new_height, new_height),
                mode="bilinear",
                align_corners=False,
            )
            if self.apply_imagenet_normalization:
                # normalize the data
                normalize_transform = make_normalize_transform()
                data_i = normalize_transform(data_i)
            data_list.append(data_i)
        return data_list

    def prepare_input(
        self,
        masked_helios_sample: MaskedHeliosSample,
    ) -> list[torch.Tensor]:
        """Prepare input for the dinov2 model from MaskedHeliosSample."""
        input_data_timesteps: dict[int, list[torch.Tensor]] = {}
        num_modalities = len(masked_helios_sample.modalities)
        for modality in masked_helios_sample.modalities:
            if num_modalities > 1:
                raise ValueError(
                    "DINOv2 does not yet support multiple modalities via multiple forward passes"
                )
            if modality not in self.supported_modalities:
                logger.warning(
                    f"Skipping modality {modality} as it is not in the supported modalities list {self.supported_modalities}"
                )
                continue  # Skip non-rgb modalities

            data = getattr(masked_helios_sample, modality)

            if data is None:
                continue

            # Process the modality data
            processed_data = self._process_modality_data(data, modality)
            for i, data_i in enumerate(processed_data):
                # start the list if it doesn't exist
                if i not in input_data_timesteps:
                    input_data_timesteps[i] = []
                input_data_timesteps[i].append(data_i)
            num_modalities += 1

        if not input_data_timesteps:
            raise ValueError("No valid modalities found for processing")
        per_timestep_inputs = []
        for i, input_data_i in input_data_timesteps.items():
            # Concatenate all modality data along channel dimension
            concatenated_imgs = torch.cat(input_data_i, dim=1)

            per_timestep_inputs.append(concatenated_imgs)
        return per_timestep_inputs

    # pooling type is on the timesteps only right now
    def forward(
        self,
        masked_helios_sample: MaskedHeliosSample,
        pooling: PoolingType = PoolingType.MEAN,
    ) -> torch.Tensor:
        """Forward pass through dinov2 model for classification."""
        # Prepare input
        per_timestep_inputs = self.prepare_input(masked_helios_sample)
        # potentially will need to add a flag for segmentation
        output_features = []
        for data in per_timestep_inputs:
            if self.use_cls_token:
                timestep_output = self.model(data)
            else:
                timestep_output = self.model.forward_features(data)[
                    "x_norm_patchtokens"
                ].mean(dim=1)
            output_features.append(timestep_output.unsqueeze(0))
        # stack in the timestep dimension and take the mean or maybe the max?
        if pooling == PoolingType.MEAN:
            output_features = torch.cat(output_features, dim=0).mean(dim=0)
        elif pooling == PoolingType.MAX:
            output_features = torch.max(torch.cat(output_features, dim=0), dim=0)[0]
        return output_features

    def forward_features(
        self,
        masked_helios_sample: MaskedHeliosSample,
        pooling: PoolingType = PoolingType.MEAN,
    ) -> torch.Tensor:
        """Forward pass through dinov2 model for segmentation."""
        # supports multi-timestep input single timestep output
        per_timestep_dinov2_inputs = self.prepare_input(masked_helios_sample)
        output_features = []
        for dinov2_input in per_timestep_dinov2_inputs:
            timestep_output = self.model.forward_features(dinov2_input)[
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
        masked_helios_sample: MaskedHeliosSample,
        pooling: PoolingType = PoolingType.MEAN,
    ) -> torch.Tensor:
        """Make the wrapper callable."""
        return self.forward(masked_helios_sample, pooling)


@dataclass
class DINOv2Config(Config):
    """olmo_core style config for DINOv2Wrapper."""

    torchhub_id: str = "dinov2_vitb14"
    use_cls_token: bool = False
    apply_imagenet_normalization: bool = False

    def build(self) -> DINOv2:
        """Build the DINOv2 from this config."""
        return DINOv2(
            torchhub_id=self.torchhub_id,
            use_cls_token=self.use_cls_token,
            apply_imagenet_normalization=self.apply_imagenet_normalization,
        )
