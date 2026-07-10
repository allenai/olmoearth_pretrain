"""Dual-resolution variant of ``base.py`` with a PixelDiT-style pixel pathway.

Replaces the interleaved pixel branch with a post-trunk pathway following PixelDiT
(https://arxiv.org/abs/2511.20645, ``pixel_branch_type="pixeldit"``): the coarse
trunk runs unmodified to completion, then M=4 tiny PiT blocks (Dp=16) refine the
pixel tokens, conditioned on the final coarse tokens via **pixel-wise AdaLN** (every
within-patch offset gets its own scale/shift/gate, unlike the per-patch FiLM
broadcast of the other variants) with global context via **token compaction**
attention (each unit's pixels compact to one coarse-width token that attends over the
trunk's packed sequence with the same RoPE, then expands back).

Unlike the paper, whose pixel pathway directly emits the diffusion output, the final
pixel tokens are re-aggregated per unit through a zero-initialized learned compaction
and added residually to the output tokens -- so the latent-MIM loss and all in-loop
evals consume pixel-refined tokens, while the pixel reconstruction decoder and map
probe consume the raw pixel features as usual.

Measured ~1.45x a coarse-only model per training step on a typical batch mix
(single-GPU benchmark, ``benchmark_pixel_branch.py`` variant ``pixeldit_d16_m4``),
vs ~8x for the original joint pixel branch.
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
PIXEL_EMBEDDING_SIZE = 16
# Only used by the pixel reconstruction decoder (the pathway's attention runs at the
# coarse width); must divide PIXEL_EMBEDDING_SIZE.
PIXEL_NUM_HEADS = 2
PIXEL_MLP_RATIO = 4.0
PIXEL_DIT_DEPTH = 4


def build_model_config(common: CommonComponents) -> DualResLatentMIMConfig:
    """Build the model config with the PixelDiT-style pixel pathway."""
    config = build_model_config_base(common)
    encoder = config.encoder_config
    encoder.pixel_branch_type = PIXEL_BRANCH_TYPE
    encoder.pixel_embedding_size = PIXEL_EMBEDDING_SIZE
    encoder.pixel_num_heads = PIXEL_NUM_HEADS
    encoder.pixel_mlp_ratio = PIXEL_MLP_RATIO
    encoder.pixel_dit_depth = PIXEL_DIT_DEPTH
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
