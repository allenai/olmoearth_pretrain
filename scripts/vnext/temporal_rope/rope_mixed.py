"""ViT-base v1.1 hidden-projection baseline with RoPE-Mixed.

Variant of scripts/vnext/temporal_rope/rope.py that swaps axial 2D RoPE for the
learnable mixed 2D frequencies from "Rotary Position Embedding for Vision
Transformer" (Heo et al., 2024, https://arxiv.org/abs/2403.13298).

Each attention layer learns its own ``(2, num_heads, head_dim // 2)`` table of
2D frequencies, allowing the RoPE phase to mix row/col directions and encode
diagonal relative offsets that axial RoPE cannot represent.
"""

import logging

from base import (
    build_common_components,
    build_dataloader_config,
    build_dataset_config,
    build_train_module_config,
    build_trainer_config,
    build_visualize_config,
)
from base import (
    build_model_config as build_model_config_base,
)

from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig

logger = logging.getLogger(__name__)

SPATIAL_POS_ENCODING = "rope_mixed"
ROPE_MIXED_BASE = 10.0


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """Build the v1.1 base model with RoPE-Mixed."""
    config = build_model_config_base(common)

    config.encoder_config.spatial_pos_encoding = SPATIAL_POS_ENCODING
    config.encoder_config.rope_mixed_base = ROPE_MIXED_BASE

    config.decoder_config.spatial_pos_encoding = SPATIAL_POS_ENCODING
    config.decoder_config.rope_mixed_base = ROPE_MIXED_BASE

    return config


def run() -> None:
    """Run the experiment."""
    main(
        common_components_builder=build_common_components,
        model_config_builder=build_model_config,
        train_module_config_builder=build_train_module_config,
        dataset_config_builder=build_dataset_config,
        dataloader_config_builder=build_dataloader_config,
        trainer_config_builder=build_trainer_config,
        visualize_config_builder=build_visualize_config,
    )


if __name__ == "__main__":
    run()
