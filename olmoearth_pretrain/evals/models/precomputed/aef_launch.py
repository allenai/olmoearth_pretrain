"""AlphaEarth Foundations (Google Satellite Embeddings) launch script for evaluation.

AlphaEarth is distributed as precomputed embeddings only (no released weights),
so this baseline reads the GSE modality baked into the eval dataset stores.
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
    """Build the model config for AlphaEarth (GSE) evaluation."""
    return PrecomputedEmbeddingConfig(modality=Modality.GSE.name)
