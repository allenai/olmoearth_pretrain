"""Dual-resolution variant of ``base.py`` with a cross-attention-only pixel branch.

Replaces the joint-attention pixel branch with Perceiver-style pixel tokens
(``pixel_branch_type="perceiver"``): pixels never mix with each other. Each pixel step
FiLM-conditions the unit's pixels on its coarse token and applies a pointwise MLP; the
coarse token attention-pools over its unit's pixels (a cheap pixel-width read) to pull
fine detail back up. The pixel branch is a steered high-resolution "memory" that
preserves and sharpens local evidence; ALL reasoning (spatial, temporal, cross-modal)
stays in the coarse branch. Runs after every 4th coarse block (3 fusion steps).

This is the ablation point for whether spatial mixing in the fine branch matters at
all (compare against ``conv_d64_e4.py`` / ``window_d64_e4.py``).

Measured ~1.4x a coarse-only model per training step on a typical batch mix (up to
~1.6x on a heavy large-patch mix; single-GPU benchmark, ``benchmark_pixel_branch.py``
variant ``perceiver_d64_every4``), vs ~8x for the original joint pixel branch.
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

PIXEL_BRANCH_TYPE = "perceiver"
PIXEL_EMBEDDING_SIZE = 64
PIXEL_NUM_HEADS = 4
PIXEL_MLP_RATIO = 2.0
PIXEL_EVERY_K_BLOCKS = 4
PIXEL_COARSE_READ_INTERVAL = 1


def build_model_config(common: CommonComponents) -> DualResLatentMIMConfig:
    """Build the model config with the Perceiver-style pixel branch."""
    config = build_model_config_base(common)
    encoder = config.encoder_config
    encoder.pixel_branch_type = PIXEL_BRANCH_TYPE
    encoder.pixel_embedding_size = PIXEL_EMBEDDING_SIZE
    encoder.pixel_num_heads = PIXEL_NUM_HEADS
    encoder.pixel_mlp_ratio = PIXEL_MLP_RATIO
    encoder.pixel_every_k_blocks = PIXEL_EVERY_K_BLOCKS
    encoder.pixel_coarse_read_interval = PIXEL_COARSE_READ_INTERVAL
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
