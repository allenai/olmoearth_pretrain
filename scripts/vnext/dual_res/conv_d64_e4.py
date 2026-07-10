"""Dual-resolution variant of ``base.py`` with a convolutional pixel branch.

Replaces the joint-attention pixel branch with the BiSeNet / MobileViT-style
convolutional detail branch (``pixel_branch_type="conv"``): a shallow high-resolution
branch of DiT-modulated ConvNeXt units (depthwise 3x3 over each ``(timestep, band
set)`` frame + pointwise MLP with expansion 2) at 64 dims, running after every 4th
coarse block (3 fusion steps). Fusion is bidirectional and cheap: coarse -> pixel via per-patch FiLM,
pixel -> coarse via per-patch mean-pool + zero-init linear. Spatial mixing crosses
patch boundaries (masked pixels are zeroed at input so nothing leaks); temporal and
cross-modal reasoning stays in the coarse branch.

Measured ~1.5x a coarse-only model per training step on a typical batch mix (up to
~1.7x on a heavy large-patch mix; single-GPU benchmark, ``benchmark_pixel_branch.py``
variant ``conv_d64_every4``), vs ~8x for the original joint pixel branch.
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
PIXEL_EVERY_K_BLOCKS = 4
PIXEL_CONV_KERNEL = 3


def build_model_config(common: CommonComponents) -> DualResLatentMIMConfig:
    """Build the model config with the convolutional pixel branch."""
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
