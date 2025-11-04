"""Tessera model launch script for evaluation."""

import logging

from olmoearth_pretrain.evals.models.tessera.tessera import TesseraConfig
from olmoearth_pretrain.internal.experiment import (
    CommonComponents,
)
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig

logger = logging.getLogger(__name__)


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """Build the model config for Tessera evaluation."""
    model_config = TesseraConfig()
    return model_config
