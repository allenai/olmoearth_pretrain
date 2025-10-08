"""Transformations for the OlmoEarthSample."""

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch
import torchvision.transforms.v2.functional as F
from class_registry import ClassRegistry
from einops import rearrange
from olmo_core.config import Config
from torch.distributions import Beta

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.data.dataset import OlmoEarthSample
from olmoearth_pretrain.types import ArrayTensor


class Transform(ABC):
    """A transform that can be applied to a OlmoEarthSample."""

    @abstractmethod
    def apply(self, batch: OlmoEarthSample) -> "OlmoEarthSample":
        """Apply the transform to the batch."""
        pass


TRANSFORM_REGISTRY = ClassRegistry[Transform]()


@TRANSFORM_REGISTRY.register("no_transform")
class NoTransform(Transform):
    """No transformation."""

    def apply(self, batch: OlmoEarthSample) -> "OlmoEarthSample":
        """Apply the transform to the batch."""
        return batch


@TRANSFORM_REGISTRY.register("flip_and_rotate")
class FlipAndRotateSpace(Transform):
    """Choose 1 of 8 transformations and apply it to data that is space varying."""

    def __init__(self) -> None:
        """Initialize the FlipAndRotateSpace class."""
        self.transformations = [
            self.no_transform,
            self.rotate_90,
            self.rotate_180,
            self.rotate_270,
            self.hflip,
            self.vflip,
            self.hflip_rotate_90,
            self.vflip_rotate_90,
        ]

    def no_transform(self, x: ArrayTensor) -> ArrayTensor:
        """No transformation."""
        return x

    def rotate_90(self, x: ArrayTensor) -> ArrayTensor:
        """Rotate 90 degrees."""
        return F.rotate(x, 90)

    def rotate_180(self, x: ArrayTensor) -> ArrayTensor:
        """Rotate 180 degrees."""
        return F.rotate(x, 180)

    def rotate_270(self, x: ArrayTensor) -> ArrayTensor:
        """Rotate 270 degrees."""
        return F.rotate(x, 270)

    def hflip(self, x: ArrayTensor) -> ArrayTensor:
        """Horizontal flip."""
        return F.hflip(x)

    def vflip(self, x: ArrayTensor) -> ArrayTensor:
        """Vertical flip."""
        return F.vflip(x)

    def hflip_rotate_90(self, x: ArrayTensor) -> ArrayTensor:
        """Horizontal flip of 90-degree rotated image."""
        return F.hflip(F.rotate(x, 90))

    def vflip_rotate_90(self, x: ArrayTensor) -> ArrayTensor:
        """Vertical flip of 90-degree rotated image."""
        return F.vflip(F.rotate(x, 90))

    def apply(
        self,
        batch: OlmoEarthSample,
    ) -> "OlmoEarthSample":
        """Apply a random transformation to the space varying data."""
        # Choose a random transformation
        transformation = random.choice(self.transformations)
        new_data_dict: dict[str, ArrayTensor] = {}
        for attribute, modality_data in batch.as_dict(ignore_nones=True).items():
            if attribute == "timestamps":
                new_data_dict[attribute] = modality_data
            else:
                modality_spec = Modality.get(attribute)
                # Apply the transformation to the space varying data
                if (
                    modality_spec.is_spacetime_varying
                    or modality_spec.is_space_only_varying
                ):
                    modality_data = rearrange(modality_data, "b h w t c -> b t c h w")
                    modality_data = transformation(modality_data)
                    modality_data = rearrange(modality_data, "b t c h w -> b h w t c")
                new_data_dict[attribute] = modality_data
        # Return the transformed sample
        return OlmoEarthSample(**new_data_dict)


@TRANSFORM_REGISTRY.register("mixup")
class Mixup(Transform):
    """Apply mixup.

    https://arxiv.org/abs/1710.09412

    To run this, use the following kwargs when launching a training job:
    --train_module.transform_config.transform_type=mixup
    --train_module.transform_config.transform_kwargs={"alpha": 1.3}
    """

    def __init__(self, alpha: float) -> None:
        """Apply mixup.

        Args:
            alpha: the alpha value to use when creating the Beta distribution to sample from
        """
        self.alpha = alpha
        self.dist = Beta(torch.tensor([alpha]), torch.tensor([alpha]))

    def apply(self, batch: OlmoEarthSample) -> OlmoEarthSample:
        """Apply mixup."""
        other_microbatch = batch.rotate()

        lam = float(self.dist.sample())
        if lam >= 0.5:
            ts_to_keep = other_microbatch.timestamps
        else:
            ts_to_keep = batch.timestamps
        return batch.scale(1 - lam).add(other_microbatch.scale(lam), ts_to_keep)


@dataclass
class TransformConfig(Config):
    """Configuration for the transform."""

    transform_type: str = "no_transform"
    transform_kwargs: dict[str, Any] = field(default_factory=lambda: {})

    def validate(self) -> None:
        """Validate the configuration."""
        if self.transform_type not in TRANSFORM_REGISTRY:
            raise ValueError(f"Invalid transform type: {self.transform_type}")

    def build(self) -> Transform:
        """Build the transform."""
        self.validate()
        return TRANSFORM_REGISTRY.get_class(self.transform_type)(
            **self.transform_kwargs
        )
