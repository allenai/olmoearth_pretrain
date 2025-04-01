"""Transformations for the HeliosSample."""

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torchvision.transforms.v2.functional as F
from class_registry import ClassRegistry
from einops import rearrange
from olmo_core.config import Config

from helios.data.constants import Modality
from helios.data.dataset import HeliosSample
from helios.types import ArrayTensor


class Transform(ABC):
    """A transform that can be applied to a HeliosSample."""

    @abstractmethod
    def apply(self, batch: HeliosSample) -> "HeliosSample":
        """Apply the transform to the batch."""
        pass


TRANSFORM_REGISTRY = ClassRegistry[Transform]()


@TRANSFORM_REGISTRY.register("no_transform")
class NoTransform(Transform):
    """No transformation."""

    def apply(self, batch: HeliosSample) -> "HeliosSample":
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
        batch: HeliosSample,
    ) -> "HeliosSample":
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
        return HeliosSample(**new_data_dict)


@dataclass
class TransformConfig(Config):
    """Configuration for the transform."""

    transform_type: str = "no_transform"

    def validate(self) -> None:
        """Validate the configuration."""
        if self.transform_type not in TRANSFORM_REGISTRY:
            raise ValueError(f"Invalid transform type: {self.transform_type}")

    def build(self) -> Transform:
        """Build the transform."""
        self.validate()
        return TRANSFORM_REGISTRY.get_class(self.transform_type)()
