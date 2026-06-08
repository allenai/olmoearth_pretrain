"""ViT-base v1.1 hidden-projection baseline with 2D RoPE.

This is intentionally based on scripts/official/v1_1/base.py, which matches the
hidden1 W&B run d7nfwd1i. The only model change is replacing additive absolute
spatial encodings with attention-level axial 2D RoPE.
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

SPATIAL_POS_ENCODING = "rope"
ROPE_BASE = 10000.0
ROPE_COORDINATE_SCALE = 1.0


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """Build the v1.1 base model with 2D RoPE."""
    config = build_model_config_base(common)

    config.encoder_config.spatial_pos_encoding = SPATIAL_POS_ENCODING
    config.encoder_config.rope_base = ROPE_BASE
    config.encoder_config.rope_coordinate_scale = ROPE_COORDINATE_SCALE

    config.decoder_config.spatial_pos_encoding = SPATIAL_POS_ENCODING
    config.decoder_config.rope_base = ROPE_BASE
    config.decoder_config.rope_coordinate_scale = ROPE_COORDINATE_SCALE

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
