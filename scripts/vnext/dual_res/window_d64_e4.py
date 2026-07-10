"""Dual-resolution variant of ``base.py`` with a patch-window attention pixel branch.

Replaces the joint-attention pixel branch with per-unit window attention
(``pixel_branch_type="window"``): each window is exactly one unit's ``P**2`` pixels
plus the unit's coarse token projected down to 64 dims as a per-window register
(ViTDet/Swin window attention + a register/CLS). Sequence length is at most 65, there
is no padding and no attention mask, and the coarse fusion is bidirectional for free:
pixels read the register, and the register's update is projected back up (zero-init)
onto the coarse token. Runs after every 4th coarse block; pixels never attend across
patches, timesteps or modalities (the coarse branch does that reasoning).

Measured ~1.45x a coarse-only model per training step on a typical batch mix (up to
~1.6x on a heavy large-patch mix; single-GPU benchmark, ``benchmark_pixel_branch.py``
variant ``window_d64_every4``), vs ~8x for the original joint pixel branch.
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

PIXEL_BRANCH_TYPE = "window"
PIXEL_EMBEDDING_SIZE = 64
PIXEL_NUM_HEADS = 4
PIXEL_MLP_RATIO = 2.0
PIXEL_EVERY_K_BLOCKS = 4


def build_model_config(common: CommonComponents) -> DualResLatentMIMConfig:
    """Build the model config with the window-attention pixel branch."""
    config = build_model_config_base(common)
    encoder = config.encoder_config
    encoder.pixel_branch_type = PIXEL_BRANCH_TYPE
    encoder.pixel_embedding_size = PIXEL_EMBEDDING_SIZE
    encoder.pixel_num_heads = PIXEL_NUM_HEADS
    encoder.pixel_mlp_ratio = PIXEL_MLP_RATIO
    encoder.pixel_every_k_blocks = PIXEL_EVERY_K_BLOCKS
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
