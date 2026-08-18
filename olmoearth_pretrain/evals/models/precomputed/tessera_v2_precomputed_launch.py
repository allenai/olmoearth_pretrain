"""Tessera v2 precomputed-embeddings launch script for evaluation.

Reads Tessera v2 student embeddings baked into eval dataset stores as the
``tessera_v2`` modality. Unlike ``tessera``/``tessera_v11`` (published
products fetched via geotessera), the v2 layer is produced by running the
released v2 inference ourselves — see docs/TesseraV2Inference.md; the student
size used is recorded in the layer's provenance manifest.
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
    """Build the model config for precomputed Tessera v2 evaluation."""
    return PrecomputedEmbeddingConfig(modality=Modality.TESSERA_V2.name)
