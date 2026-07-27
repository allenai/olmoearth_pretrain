"""Tessera v1.1 precomputed-embeddings launch script for evaluation.

Reads the published Tessera v1.1 (cambridge variant) embedding product baked
into eval dataset stores as the ``tessera_v11`` modality. The ``tessera``
modality/baseline holds the older v1 product, so the two dataset versions can
be evaluated side by side.
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
    """Build the model config for precomputed Tessera v1.1 evaluation."""
    return PrecomputedEmbeddingConfig(modality=Modality.TESSERA_V11.name)
