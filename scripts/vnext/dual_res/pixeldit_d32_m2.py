"""Dual-resolution variant of ``base.py``: PixelDiT pathway, Dp=32, 2 PiT blocks.

Identical to ``pixeldit_d16_m4.py`` except the pixel embedding size is 32 (double the
paper-proportioned 16) and the pathway is 2 PiT blocks deep instead of 4 -- a
wider-but-shallower point of the same design. See that script and
``olmoearth_pretrain/nn/dual_res_encoder.py`` for the architecture.

Measured ~1.45x a coarse-only model per training step on a typical batch mix, with
the (cross-modality) pixel reconstruction loss active at depth 1
(single-GPU benchmark, ``benchmark_pixel_branch.py`` variant ``pixeldit_d32_m2``).
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

PIXEL_BRANCH_TYPE = "pixeldit"
PIXEL_EMBEDDING_SIZE = 32
# Only used by the pixel reconstruction decoder (the pathway's attention runs at the
# coarse width); must divide PIXEL_EMBEDDING_SIZE.
PIXEL_NUM_HEADS = 2
PIXEL_MLP_RATIO = 4.0
PIXEL_DIT_DEPTH = 2
# One cross-attention block in the (cross-modality) pixel reconstruction decoder
# -- depth 2 measurably slows the step now that the decoder does real work.
PIXEL_RECON_DEPTH = 1


def build_model_config(common: CommonComponents) -> DualResLatentMIMConfig:
    """Build the model config with the wider, shallower PixelDiT pathway."""
    config = build_model_config_base(common)
    encoder = config.encoder_config
    encoder.pixel_branch_type = PIXEL_BRANCH_TYPE
    encoder.pixel_embedding_size = PIXEL_EMBEDDING_SIZE
    encoder.pixel_num_heads = PIXEL_NUM_HEADS
    encoder.pixel_mlp_ratio = PIXEL_MLP_RATIO
    encoder.pixel_dit_depth = PIXEL_DIT_DEPTH
    config.pixel_recon_num_heads = PIXEL_NUM_HEADS
    config.pixel_recon_depth = PIXEL_RECON_DEPTH
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
