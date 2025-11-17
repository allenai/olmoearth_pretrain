"""Trying to prototype fitting everything into olmo core."""

import logging

from olmoearth_pretrain.evals.models.satlas.satlas import SatlasConfig
from olmoearth_pretrain.internal.experiment import (
    CommonComponents,
)

logger = logging.getLogger(__name__)


def build_model_config(common: CommonComponents) -> SatlasConfig:
    """Build the model config for an experiment."""
    model_config = SatlasConfig()
    return model_config
