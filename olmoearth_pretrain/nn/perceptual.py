"""VGG perceptual (content) loss for the NAIP GAN branch.

Mirrors the perceptual loss used by ESRGAN / Real-ESRGAN (and the satlas
super-resolution recipe): a frozen ImageNet VGG-19 feature extractor compares
generated and real images in feature space at several depths, using the
pre-activation (before-ReLU) conv outputs with the standard layer weights. This
supplies the mid-frequency content/structure gradient that an L1 pixel loss and
a weak adversarial term cannot provide on their own.
"""

import logging
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)

# Indices into ``torchvision`` ``vgg19().features`` of the conv outputs
# *before* their ReLU, matching the ESRGAN layer names.
_VGG19_PRE_RELU_INDICES: dict[str, int] = {
    "conv1_2": 2,
    "conv2_2": 7,
    "conv3_4": 16,
    "conv4_4": 25,
    "conv5_4": 34,
}

# ESRGAN / satlas default per-layer weights.
_DEFAULT_LAYER_WEIGHTS: dict[str, float] = {
    "conv1_2": 0.1,
    "conv2_2": 0.1,
    "conv3_4": 1.0,
    "conv4_4": 1.0,
    "conv5_4": 1.0,
}

# ImageNet normalization statistics (VGG was trained on these).
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class VGGPerceptualLoss(nn.Module):
    """Frozen VGG-19 perceptual loss over pre-ReLU feature maps.

    The module is a pure loss: its parameters are frozen and excluded from every
    optimizer, but gradients still flow through it into whatever produced the
    ``fake`` image. Inputs are RGB images in ``[0, 1]`` and are ImageNet-normalized
    internally.
    """

    def __init__(
        self,
        layer_weights: dict[str, float] | None = None,
        criterion: str = "l1",
    ):
        """Initialize the perceptual loss.

        Args:
            layer_weights: Mapping from VGG-19 layer name (a key of
                ``_VGG19_PRE_RELU_INDICES``) to its weight in the summed loss.
                Defaults to the ESRGAN weights.
            criterion: Feature comparison criterion, ``"l1"`` or ``"l2"``.
        """
        super().__init__()
        from torchvision.models import VGG19_Weights, vgg19

        weights = layer_weights or dict(_DEFAULT_LAYER_WEIGHTS)
        unknown = set(weights) - set(_VGG19_PRE_RELU_INDICES)
        if unknown:
            raise ValueError(
                f"Unknown VGG layer(s) {sorted(unknown)}; valid layers are "
                f"{sorted(_VGG19_PRE_RELU_INDICES)}"
            )
        if criterion not in ("l1", "l2"):
            raise ValueError(f"criterion must be 'l1' or 'l2', got {criterion!r}")
        self.layer_weights = weights
        self.criterion = criterion

        # Keep only the feature layers up to the deepest one we need.
        max_index = max(_VGG19_PRE_RELU_INDICES[name] for name in weights)
        features = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[: max_index + 1]
        self.features = features.eval()
        for param in self.features.parameters():
            param.requires_grad_(False)

        # ``[1, 3, 1, 1]`` normalization buffers so they follow ``.to(device)``.
        self.register_buffer(
            "mean", torch.tensor(_IMAGENET_MEAN).reshape(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "std", torch.tensor(_IMAGENET_STD).reshape(1, 3, 1, 1), persistent=False
        )

        # Map each captured feature-map index to its layer weight.
        self._capture = {
            _VGG19_PRE_RELU_INDICES[name]: weight for name, weight in weights.items()
        }

    def train(self, mode: bool = True) -> "VGGPerceptualLoss":
        """Keep the VGG features frozen in eval mode regardless of parent mode."""
        super().train(mode)
        self.features.eval()
        return self

    def _extract(self, x: Tensor) -> dict[int, Tensor]:
        """Return the pre-ReLU feature maps at the configured indices."""
        x = (x - self.mean) / self.std
        feats: dict[int, Tensor] = {}
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self._capture:
                feats[i] = x
        return feats

    def forward(self, fake: Tensor, real: Tensor) -> Tensor:
        """Compute the weighted perceptual loss between ``fake`` and ``real``.

        Args:
            fake: Generated RGB image ``[N, 3, H, W]`` in ``[0, 1]`` (keeps grad).
            real: Target RGB image ``[N, 3, H, W]`` in ``[0, 1]``.

        Returns:
            Scalar perceptual loss.
        """
        param_dtype = self.mean.dtype
        fake_feats = self._extract(fake.to(param_dtype))
        with torch.no_grad():
            real_feats = self._extract(real.to(param_dtype))
        loss = fake.new_zeros(())
        for index, weight in self._capture.items():
            if self.criterion == "l1":
                term = F.l1_loss(fake_feats[index], real_feats[index])
            else:
                term = F.mse_loss(fake_feats[index], real_feats[index])
            loss = loss + weight * term
        return loss


@dataclass
class VGGPerceptualLossConfig:
    """Configuration for :class:`VGGPerceptualLoss`."""

    layer_weights: dict[str, float] = field(
        default_factory=lambda: dict(_DEFAULT_LAYER_WEIGHTS)
    )
    criterion: str = "l1"

    def build(self) -> VGGPerceptualLoss:
        """Build the perceptual loss module."""
        return VGGPerceptualLoss(
            layer_weights=self.layer_weights,
            criterion=self.criterion,
        )
