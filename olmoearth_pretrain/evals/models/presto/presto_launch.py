"""Trying to prototype fitting everything into olmo core."""

import logging

from olmoearth_pretrain.evals.models import PrestoConfig
from olmoearth_pretrain.internal.experiment import (
    CommonComponents,
)
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig

logger = logging.getLogger(__name__)


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """Build the model config for an experiment."""
    model_config = PrestoConfig()
    return model_config
