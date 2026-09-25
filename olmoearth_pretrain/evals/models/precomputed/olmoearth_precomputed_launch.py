"""OlmoEarth published large-scale embeddings launch script for evaluation.

Reads the ``olmoearth_emb`` modality: embeddings produced by the rslearn_projects
large-scale embedding pipeline and published as a geozarr store, baked into an
eval dataset by scripts/tools/bake_olmoearth_zarr_embeddings.py. Scoring them
under the same tasks as a forward-pass OlmoEarth checkpoint checks that the
production inference pipeline reproduces the evals.
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
    """Build the model config for precomputed OlmoEarth embedding evaluation."""
    return PrecomputedEmbeddingConfig(modality=Modality.OLMOEARTH_EMB.name)
