"""Dual-resolution variant of ``base.py`` with a 96-dim per-pixel branch.

Identical to ``base.py`` in every respect except the per-pixel branch embedding size,
which is 96 instead of 128 (the coarse encoder/decoder stays at 512-dim). Use this to
measure the quality/speed trade-off of an even lighter pixel branch. 96 is divisible by
``pixel_num_heads`` (4), as the encoder config requires.
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

PIXEL_EMBEDDING_SIZE = 96
# 3 heads -> head_dim 32 (the canonical per-head width) for a 96-dim pixel branch;
# 96 % 3 == 0, as the encoder config requires.
PIXEL_NUM_HEADS = 3


def build_model_config(common: CommonComponents) -> DualResLatentMIMConfig:
    """Build the dual-resolution model config with a 96-dim per-pixel branch."""
    config = build_model_config_base(common)
    # Shrink only the per-pixel branch; the coarse encoder/decoder stay at 512-dim. The
    # reconstruction decoder reads the encoder's pixel embedding size at build time, so
    # this single override propagates to it too.
    config.encoder_config.pixel_embedding_size = PIXEL_EMBEDDING_SIZE
    config.encoder_config.pixel_num_heads = PIXEL_NUM_HEADS
    config.pixel_recon_num_heads = PIXEL_NUM_HEADS
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
