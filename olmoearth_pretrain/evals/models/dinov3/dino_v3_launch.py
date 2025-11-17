"""Trying to prototype fitting everything into olmo core."""

import logging

from olmoearth_pretrain.evals.models import DINOv3Config
from olmoearth_pretrain.evals.models.dinov3.dinov3 import DinoV3Models
from olmoearth_pretrain.internal.experiment import (
    CommonComponents,
)
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig

logger = logging.getLogger(__name__)


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """Build the model config for an experiment."""
    model_config = DINOv3Config(
        apply_normalization=True, size=DinoV3Models.LARGE_SATELLITE
    )
    return model_config
