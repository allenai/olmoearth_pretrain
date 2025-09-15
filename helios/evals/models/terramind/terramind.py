"""Terramind models."""

from enum import Enum
from typing import Any

import torch
from einops import rearrange
from rslearn.train.transforms.transform import Transform
from terratorch.registry import BACKBONE_REGISTRY


# TerraMind v1 provides two sizes: base and large
class TerramindSize(str, Enum):
    """Size of the Terramind model."""

    BASE = "base"
    LARGE = "large"


# Default patch size for Terramind
PATCH_SIZE = 16

# Modalities supported by Terramind
# S2L1C: Sentinel-2 Level 1C (Top-of-atmosphere reflectance), range: 1000 – 11000 DN
# S2L2A: Sentinel-2 Level 2A (Bottom-of-atmosphere reflectance), range: 1000 – 11000 DN
# S1GRD: Sentinel-1 GRD (Calibrated SAR backscatter), range: -50 – +10 dB
# S1RTC: Sentinel-1 RTC (Radiometrically terrain corrected), range: -50 – +10 dB
# RGB: Processed RGB images based on S2L2A, range: 0-255
# DEM: Digital Elevation Model (Copernicus DEM, 30m), range: -400 – 8800 meters

# More details in the TerraMesh paper: https://arxiv.org/pdf/2504.11172v1
TERRAMIND_MODALITIES = ["S2L1C", "S2L2A", "S1GRD", "S1RTC", "RGB", "DEM"]

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


class Terramind(torch.nn.Module):
    """Terramind backbones."""

    def __init__(
        self,
        model_size: TerramindSize,
        modalities: list[str] = ["S2L2A"],
    ) -> None:
        """Initialize the Terramind model.

        Args:
            model_size: The size of the Terramind model.
            modalities: The modalities to use.
        """
        super().__init__()

        # Check if all modalities are valid
        for modality in modalities:
            if modality not in TERRAMIND_MODALITIES:
                raise ValueError(f"Invalid modality: {modality}")

        if model_size == TerramindSize.BASE:
            self.model = BACKBONE_REGISTRY.build(
                "terramind_v1_base", modalities=modalities, pretrained=True
            )
        elif model_size == TerramindSize.LARGE:
            self.model = BACKBONE_REGISTRY.build(
                "terramind_v1_large", modalities=modalities, pretrained=True
            )
        else:
            raise ValueError(f"Invalid model size: {model_size}")

        self.model_size = model_size
        self.modalities = modalities

    def forward(self, inputs: list[dict[str, Any]]) -> list[torch.Tensor]:
        """Forward pass for the Terramind model.

        Args:
            inputs: input dicts that must include modalities as keys which are defined in the self.modalities list

        Returns:
            List[torch.Tensor]: Single-scale feature tensors from the encoder.
        """
        model_inputs = {}
        for modality in self.modalities:
            # We assume the all the inputs include the same modalities
            if modality not in inputs[0]:
                continue
            cur = torch.stack([inp[modality] for inp in inputs], dim=0)  # (B, C, H, W)
            model_inputs[modality] = cur

        # By default, the patch embeddings are averaged over all modalities to reduce output tokens
        # The output is a list of tensors (B, N, D) from each layer of the transformer
        # We only get the last layer's output
        image_features = self.model(model_inputs)[-1]
        batch_size, num_patches, _ = image_features.shape
        height, width = int(num_patches**0.5), int(num_patches**0.5)
        return [rearrange(image_features, "b (h w) d -> b d h w", h=height, w=width)]

    def get_backbone_channels(self) -> list:
        """Returns the output channels of this model when used as a backbone.

        The output channels is a list of (patch_size, depth) that corresponds
        to the feature maps that the backbone returns.

        Returns:
            the output channels of the backbone as a list of (patch_size, depth) tuples.
        """
        if self.model_size == TerramindSize.BASE:
            depth = 768
        elif self.model_size == TerramindSize.LARGE:
            depth = 1024
        else:
            raise ValueError(f"Invalid model size: {self.model_size}")
        return [(PATCH_SIZE, depth)]


class TerramindNormalize(Transform):
    """Normalize inputs using Terramind normalization.

    It will apply normalization to the modalities that are specified in the model configuration.
    """

    def __init__(self) -> None:
        """Initialize a new TerramindNormalize."""
        super().__init__()

    def apply_image(
        self, image: torch.Tensor, means: list[float], stds: list[float]
    ) -> torch.Tensor:
        """Normalize the specified image with Terramind normalization.

        Args:
            image: the image to normalize.
            means: the means to use for the normalization.
            stds: the standard deviations to use for the normalization.

        Returns:
            The normalized image.
        """
        images = image.float()  # (C, H, W)
        if images.shape[0] % len(means) != 0:
            raise ValueError(
                f"the number of image channels {images.shape[0]} is not multiple of expected number of bands {len(means)}"
            )
        for i in range(images.shape[0]):
            band_idx = i % len(means)
            images[i] = (images[i] - means[band_idx]) / stds[band_idx]
        return images

    def forward(
        self, input_dict: dict[str, Any], target_dict: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Normalize the specified image with Terramind normalization.

        Args:
            input_dict: the input dictionary.
            target_dict: the target dictionary.

        Returns:
            normalized (input_dicts, target_dicts) tuple
        """
        for modality in TERRAMIND_MODALITIES:
            if modality not in input_dict:
                continue
            band_info = PRETRAINED_BANDS[modality]
            means = [band_info[band][0] for band in band_info]
            stds = [band_info[band][1] for band in band_info]
            input_dict[modality] = self.apply_image(
                input_dict[modality],
                means,
                stds,
            )
        return input_dict, target_dict
