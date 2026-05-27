"""Spatial register bottleneck (Perceiver-style) on top of the hidden1 supervision run.

Combines three changes vs. ``hidden1_supervision``:
  1. 2D RoPE (attention-level) replaces additive absolute spatial encodings.
  2. A Perceiver-style spatial register bottleneck: a fixed ``n x n`` grid of learned
     latents reads the encoded patch tokens (cross-attention), a latent transformer
     self-attends over the grid, and the decoder reads ONLY the register grid.
  3. The supervision head reads the register grid (``register_supervision``) at a LOW
     weight -- the intent is a spatial-salience inductive bias for the registers, not a
     learning signal (prior supervision-weight experiments did not pay off).

All register/RoPE knobs are module constants so they can be swept via CLI overrides, e.g.
``--model.encoder_config.register_grid_size=6 --model.encoder_config.register_dim=256``.
"""

import logging

from hidden1_supervision import (
    build_common_components,
    build_dataloader_config,
    build_dataset_config,
    build_train_module_config,
    build_trainer_config,
    build_visualize_config,
)
from hidden1_supervision import (
    build_model_config as build_supervision_model_config,
)

from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig

logger = logging.getLogger(__name__)

# --- 2D RoPE ---------------------------------------------------------------------
SPATIAL_POS_ENCODING = "rope"
ROPE_BASE = 10000.0
ROPE_COORDINATE_SCALE = 1.0

# --- Register bottleneck (sweepable) ---------------------------------------------
REGISTER_GRID_SIZE = 8  # n x n registers (fixed; decoupled from the patch grid)
REGISTER_DIM: int | None = None  # None -> encoder embedding_size // 2
REGISTER_READ_DEPTH = 1  # cross-attention read blocks
REGISTER_LATENT_DEPTH = 4  # latent-transformer self-attention blocks over the grid
# None -> match the encoder's head count (12). register_dim must then be a multiple of 48
# so head_dim = register_dim / 12 stays divisible by 4 (required by 2D RoPE).
REGISTER_NUM_HEADS: int | None = None

# --- Supervision ------------------------------------------------------------------
# Scales every supervision modality weight down: a spatial-salience nudge, not a signal.
SUPERVISION_WEIGHT_MULTIPLIER = 0.1


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """Build the supervision model, then add RoPE + the register bottleneck."""
    config = build_supervision_model_config(
        common, weight_multiplier=SUPERVISION_WEIGHT_MULTIPLIER
    )

    register_dim = (
        REGISTER_DIM
        if REGISTER_DIM is not None
        else config.encoder_config.embedding_size // 2
    )

    for sub_config in (config.encoder_config, config.decoder_config):
        sub_config.spatial_pos_encoding = SPATIAL_POS_ENCODING
        sub_config.rope_base = ROPE_BASE
        sub_config.rope_coordinate_scale = ROPE_COORDINATE_SCALE
        sub_config.use_register_bottleneck = True
        sub_config.register_dim = register_dim

    config.encoder_config.register_grid_size = REGISTER_GRID_SIZE
    config.encoder_config.register_read_depth = REGISTER_READ_DEPTH
    config.encoder_config.register_latent_depth = REGISTER_LATENT_DEPTH
    config.encoder_config.register_num_heads = REGISTER_NUM_HEADS

    config.supervision_head_config.register_supervision = True

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
