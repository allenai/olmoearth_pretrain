"""Tessera precomputed-embeddings launch script for evaluation.

Unlike the ``tessera`` baseline (which rebuilds the Tessera model and runs
forward passes over raw S1/S2 pixel time series), this baseline reads the
published Tessera embedding product baked into eval dataset stores as the
``tessera`` modality — matching how downstream users consume Tessera.
"""

import logging

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.evals.models.precomputed.precomputed import (
    PrecomputedEmbeddingConfig,
)
from olmoearth_pretrain.internal.experiment import (
    CommonComponents,
)

logger = logging.getLogger(__name__)


def build_model_config(common: CommonComponents) -> PrecomputedEmbeddingConfig:
    """Build the model config for precomputed Tessera evaluation."""
    return PrecomputedEmbeddingConfig(modality=Modality.TESSERA.name)
