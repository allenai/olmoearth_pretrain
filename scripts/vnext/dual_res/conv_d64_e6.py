"""Dual-resolution variant of ``base.py``: convolutional pixel branch, every 6 blocks.

Identical to ``conv_d64_e4.py`` except the pixel branch runs after every 6th coarse
block (2 fusion steps, at blocks 6 and 12) instead of every 4th. This is the cheapest
convolutional configuration -- closest to the classic BiSeNet pattern where the
high-resolution branch is shallow and fuses late. See ``conv_d64_e4.py`` and
``olmoearth_pretrain/nn/dual_res_encoder.py`` for the architecture.

Measured ~1.4x a coarse-only model per training step (single-GPU benchmark,
``benchmark_pixel_branch.py`` variant ``conv_d64_every6``).
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
from olmoearth_pretrain.nn.dual_res_model import DualResLatentMIMConfig

logger = logging.getLogger(__name__)

PIXEL_BRANCH_TYPE = "conv"
PIXEL_EMBEDDING_SIZE = 64
PIXEL_NUM_HEADS = 4
PIXEL_MLP_RATIO = 2.0
PIXEL_EVERY_K_BLOCKS = 6
PIXEL_CONV_KERNEL = 3


def build_model_config(common: CommonComponents) -> DualResLatentMIMConfig:
    """Build the model config with the every-6-blocks convolutional pixel branch."""
    config = build_model_config_base(common)
    encoder = config.encoder_config
    encoder.pixel_branch_type = PIXEL_BRANCH_TYPE
    encoder.pixel_embedding_size = PIXEL_EMBEDDING_SIZE
    encoder.pixel_num_heads = PIXEL_NUM_HEADS
    encoder.pixel_mlp_ratio = PIXEL_MLP_RATIO
    encoder.pixel_every_k_blocks = PIXEL_EVERY_K_BLOCKS
    encoder.pixel_conv_kernel = PIXEL_CONV_KERNEL
    config.pixel_recon_num_heads = PIXEL_NUM_HEADS
    config.pixel_recon_mlp_ratio = PIXEL_MLP_RATIO
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
