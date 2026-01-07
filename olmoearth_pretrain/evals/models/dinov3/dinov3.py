"""DINOv3 model https://github.com/facebookresearch/dinov3 ."""

import logging
import math
import time
from dataclasses import dataclass

import torch
from einops import rearrange
from olmo_core.config import Config
from torch import nn
from torchvision import transforms

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.nn.flexi_vit import PoolingType
from olmoearth_pretrain.train.masking import MaskedOlmoEarthSample

from .constants import MODEL_TO_TORCHHUB_ID_AND_WEIGHTS_URL, REPO_DIR, DinoV3Models

logger = logging.getLogger(__name__)
# Use timm's names
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def make_normalize_transform_web() -> transforms.Normalize:
    """Make normalize transofrm for dinov3 trained on web dataset."""
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return normalize


def make_resize_transform(resize_size: int) -> transforms.Resize:
    """Make resize transform for dinov3."""
    resize = transforms.Resize((resize_size, resize_size), antialias=True)
    return resize


def make_normalize_transform_sat() -> transforms.Normalize:
    """Make normalize transform for dinov3 trained on satellite dataset."""
    normalize = transforms.Normalize(
        mean=(0.430, 0.411, 0.296),
        std=(0.213, 0.156, 0.143),
    )
    return normalize


# DinoV3 Expects bands ordered as R, G, B
HELIOS_SENTINEL2_RGB_BANDS = [
    Modality.SENTINEL2_L2A.band_order.index(b) for b in ["B04", "B03", "B02"]
]
HELIOS_LANDSAT_RGB_BANDS = [
    Modality.LANDSAT.band_order.index(b) for b in ["B4", "B3", "B2"]
]


class DINOv3(nn.Module):
    """Wrapper for the dinov3 model that can ingest MaskedOlmoEarthSample objects."""

    patch_size: int = 16
    base_resize: int = 256
    # TODO: Should be the supported modality names
    supported_modalities: list[str] = [
        Modality.SENTINEL2_L2A.name,
        Modality.LANDSAT.name,
    ]
    supports_multiple_modalities_at_once = False

    def __init__(
        self,
        size: str = DinoV3Models.LARGE_SATELLITE,
        use_cls_token: bool = False,
        apply_normalization: bool = False,
    ):
        """Initialize the dinov3 wrapper.

        Args:
            size: The name that corresponds to the model on torch hub to help find the details for loading the model
            use_cls_token: Whether to use the cls token (default False)
            apply_normalization: Whether to apply imagenet normalization to the input data (default False)
        """
        super().__init__()
        self.use_cls_token = use_cls_token
        self.apply_normalization = apply_normalization
        if self.apply_normalization:
            logger.warning(
                "Applying imagenet normalization to the input data. Make sure other normalization is not applied."
            )
        torchhub_id, weights_url = MODEL_TO_TORCHHUB_ID_AND_WEIGHTS_URL[size]
        # Load the model
        self._load_model(torchhub_id, weights_url)
        if "sat" in size:
            logger.info("Using satellite normalization")
            self.normalize_transform = make_normalize_transform_sat()
        else:
            logger.info("Using web normalization")
            self.normalize_transform = make_normalize_transform_web()

    def _load_model(self, torchhub_id: str, weights_url: str) -> None:
        """Load the dinov3 model from torch hub."""
        # Hack to get around https://discuss.pytorch.org/t/torch-hub-load-gives-httperror-rate-limit-exceeded/124769
        torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
        for attempt in range(2):
            try:
                self.model = torch.hub.load(
                    repo_or_dir=REPO_DIR,  # "facebookresearch/dinov3",
                    model=torchhub_id,
                    source="local",
                    weights=weights_url,
                )
                break
            except Exception as e:
                logger.warning(f"Error loading  model: {e}. Retrying in 5 seconds...")
                time.sleep(5)
        else:
            raise RuntimeError(
                f"Failed to load dinov3 model {torchhub_id} after retrying."
            )

    def _process_modality_data(
        self,
        data: torch.Tensor,
        modality: str,
    ) -> list[torch.Tensor]:
        """Process individual modality data."""
        # Rearrange from "b h w t c -> b (c t) h w" for DinoV2/dinov3 format
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

            # If it is greater than 224 Caleb R comment says don;t resize
            if original_height > self.base_resize:
                new_height = original_height
            elif original_height <= self.base_resize and original_height > 1:
                new_height = self.base_resize
            else:
                new_height = self.patch_size
            resize_transform = make_resize_transform(new_height)
            data_i = resize_transform(data_i)
            if self.apply_normalization:
                # normalize the data
                data_i = self.normalize_transform(data_i)
            data_list.append(data_i)
        return data_list

    def prepare_input(
        self,
        masked_olmoearth_sample: MaskedOlmoEarthSample,
    ) -> list[torch.Tensor]:
        """Prepare input for the dinov3 model from MaskedOlmoEarthSample."""
        input_data_timesteps: dict[int, list[torch.Tensor]] = {}
        num_modalities = len(masked_olmoearth_sample.modalities)
        for modality in masked_olmoearth_sample.modalities:
            if num_modalities > 1:
                raise ValueError(
                    f"DINOv3 does not yet support multiple modalities via multiple forward passes, got {num_modalities} modalities including {[modality for modality in masked_olmoearth_sample.modalities]}"
                )
            if modality not in self.supported_modalities:
                logger.warning(
                    f"Skipping modality {modality} as it is not in the supported modalities list {self.supported_modalities}"
                )
                continue  # Skip non-rgb modalities

            data = getattr(masked_olmoearth_sample, modality)

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
        masked_olmoearth_sample: MaskedOlmoEarthSample,
        pooling: PoolingType = PoolingType.MEAN,
    ) -> torch.Tensor:
        """Forward pass through dinov3 model for classification."""
        # Prepare input
        per_timestep_inputs = self.prepare_input(masked_olmoearth_sample)
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
        masked_olmoearth_sample: MaskedOlmoEarthSample,
        pooling: PoolingType = PoolingType.MEAN,
    ) -> torch.Tensor:
        """Forward pass through dinov3 model for segmentation."""
        # supports multi-timestep input single timestep output
        per_timestep_dinov3_inputs = self.prepare_input(masked_olmoearth_sample)
        output_features = []
        for dinov3_input in per_timestep_dinov3_inputs:
            timestep_output = self.model.forward_features(dinov3_input)[
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
class DINOv3Config(Config):
    """olmo_core style config for DINOv2Wrapper."""

    size: str | DinoV3Models = DinoV3Models.LARGE_SATELLITE
    use_cls_token: bool = False
    apply_normalization: bool = False

    def build(self) -> "DINOv3":
        """Build the DINOv3 from this config."""
        return DINOv3(
            size=self.size,
            use_cls_token=self.use_cls_token,
            apply_normalization=self.apply_normalization,
        )
