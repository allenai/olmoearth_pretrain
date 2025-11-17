"""Trying to prototype fitting everything into olmo core."""

import logging

from olmoearth_pretrain.evals.models.prithviv2.prithviv2 import PrithviV2Config
from olmoearth_pretrain.internal.experiment import (
    CommonComponents,
)

logger = logging.getLogger(__name__)


def build_model_config(common: CommonComponents) -> PrithviV2Config:
    """Build the model config for an experiment."""
    model_config = PrithviV2Config()
    return model_config
