"""Adaptive Patch Transformers (APT) for remote sensing.

This module implements content-aware adaptive patchification for Vision Transformers,
where complex regions get small patches and homogeneous regions get large patches.
This reduces token count while preserving accuracy.

Reference: "Accelerating Vision Transformers with Adaptive Patch Sizes" (arXiv:2510.18091)
"""

from olmoearth_pretrain.nn.apt.adaptive_patch_embed import (
    AdaptiveMultiModalPatchEmbed,
    AdaptivePatchEmbed,
    ConvDownsample,
)
from olmoearth_pretrain.nn.apt.apt_encoder import APTEncoder
from olmoearth_pretrain.nn.apt.config import (
    APTConfig,
    APTEmbedConfig,
    APTMaskingConfig,
    APTPartitionerConfig,
    APTScorerConfig,
    APTTransformConfig,
    ScorerType,
)
from olmoearth_pretrain.nn.apt.masking import (
    AdaptiveMaskingStrategy,
    APTAwareMaskingConfig,
)
from olmoearth_pretrain.nn.apt.partitioner import (
    PatchDescriptor,
    QuadtreePartitioner,
)
from olmoearth_pretrain.nn.apt.scorers import EntropyScorer, LaplacianScorer, Scorer
from olmoearth_pretrain.nn.apt.utils import (
    compute_position_embeddings,
    expand_tokens_to_base_grid,
    pack_adaptive_tokens,
    pack_with_positions,
    unpack_adaptive_tokens,
    validate_patch_descriptors,
)

__all__ = [
    # Scorers
    "EntropyScorer",
    "LaplacianScorer",
    "Scorer",
    "ScorerType",
    # Partitioner
    "PatchDescriptor",
    "QuadtreePartitioner",
    # Embedding
    "AdaptivePatchEmbed",
    "AdaptiveMultiModalPatchEmbed",
    "ConvDownsample",
    # Encoder
    "APTEncoder",
    # Masking
    "AdaptiveMaskingStrategy",
    "APTAwareMaskingConfig",
    # Utilities
    "pack_adaptive_tokens",
    "unpack_adaptive_tokens",
    "pack_with_positions",
    "expand_tokens_to_base_grid",
    "compute_position_embeddings",
    "validate_patch_descriptors",
    # Config
    "APTConfig",
    "APTScorerConfig",
    "APTPartitionerConfig",
    "APTEmbedConfig",
    "APTTransformConfig",
    "APTMaskingConfig",
]
