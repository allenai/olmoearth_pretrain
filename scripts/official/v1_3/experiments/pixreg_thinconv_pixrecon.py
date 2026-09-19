"""``pixreg_pixrecon`` + a thin convolutional pixel branch that INITIALIZES the registers.

Identical to ``pixreg_pixrecon.py`` (pixel-resolution d128 registers, ps 1..4, hw_p <=
24, per-cell map supervision at base 0.1, time-conditioned S2 L2A / S1 reconstruction
at 0.05 each) plus ``pixel_branch_type="thinconv"`` (``nn/pixel_branch.py``): a
standalone stack of 4 ConvNeXt-style units (depthwise 3x3 conv + pointwise MLP, width
128) on the DENSE pixel grid of the time-series inputs, run once and independent of the
coarse trunk. Its final per-pixel features -- mean-pooled over the ONLINE
``(timestep, band set, modality)`` units at each pixel -- pass through a zero-init
linear projection and are ADDED to the cloned register latent before the first read.
Non-ONLINE pixels are zeroed before the first convolution (leakage guard), and the
zero-init handoff makes this model EXACTLY ``pixreg_pixrecon`` at step 0.

WHY: in ``pixreg_pixrecon`` every pixel register starts as the same learned latent and
can only tell its neighbours apart through its RoPE phase on a read of the COARSE
trunk tokens; at ps=4 whatever sub-patch detail survives is what the trunk token
still linearly encodes. Fine-grained pressure there comes purely from the losses
(per-cell map heads, per-pixel recon). This arm is the "detail-CARRYING" counterpart:
each register is handed a cheap, translation-equivariant summary of its own pixel's
visible inputs, so the reads and the recon head start from pixel-specific content
rather than having to infer it. In the original program (v1.2 stack) the
``thinconv`` and ``convbranch`` arms were within noise of ``pixrecon`` alone at 360k;
this re-asks the question on v1.3 with the recon heads present in both arms, so the
delta is the branch alone.

COST: the branch adds ~4 x (128-d depthwise conv + 128->512->128 MLP) over up to
``96 x 96 x T x band_sets`` pixel frames with gradient checkpointing; the original
thinconv arm ran at ~0.9x the throughput of pixrecon.

Everything else (sampler, microbatch, evals at 40k steps, W&B project) is imported
from ``pixreg_pixrecon``.
"""

import logging
import sys
from pathlib import Path

# Sibling arms import the pixreg_pixrecon builders from this directory and the
# release recipe from the directory above.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from base import (  # noqa: E402
    build_common_components,
    build_dataset_config,
    build_visualize_config,
)
from pixreg_pixrecon import (  # noqa: E402
    apply_pixel_reconstruction,
    build_dataloader_config,
    build_pixreg_model_config,
    build_train_module_config,
)
from pixreg_pixrecon import (
    build_trainer_config as _pixreg_build_trainer_config,  # noqa: E402
)

from olmoearth_pretrain.internal.experiment import CommonComponents, main  # noqa: E402
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig  # noqa: E402

logger = logging.getLogger(__name__)

MODULE_PATH = "scripts/official/v1_3/experiments/pixreg_thinconv_pixrecon.py"

# --- pixel branch (the original program's thinconv arm, at Dp=128) ---------------------
PIXEL_BRANCH_TYPE = "thinconv"
PIXEL_EMBEDDING_SIZE = 128
PIXEL_THIN_DEPTH = 4
PIXEL_CONV_KERNEL = 3
PIXEL_MLP_RATIO = 4.0


def apply_pixel_branch(config: LatentMIMConfig) -> LatentMIMConfig:
    """Attach the thin conv pixel branch, in place.

    Requires the pixel-resolution register grid (``apply_pixel_registers`` first): the
    branch's only consumer is the register initialization. The zero-init handoff makes
    the model equal the branch-free ``pixreg_pixrecon`` at initialization.
    """
    encoder_config = config.encoder_config
    assert encoder_config.register_pixel_grid, (
        "apply_pixel_branch requires the pixel-resolution register grid"
    )
    encoder_config.pixel_branch_type = PIXEL_BRANCH_TYPE
    encoder_config.pixel_embedding_size = PIXEL_EMBEDDING_SIZE
    encoder_config.pixel_thin_depth = PIXEL_THIN_DEPTH
    encoder_config.pixel_conv_kernel = PIXEL_CONV_KERNEL
    encoder_config.pixel_mlp_ratio = PIXEL_MLP_RATIO
    return config


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """pixreg_pixrecon + thinconv pixel branch initializing the register grid."""
    config = build_pixreg_model_config(common)
    return apply_pixel_reconstruction(apply_pixel_branch(config))


def build_trainer_config(common: CommonComponents):
    """pixreg_pixrecon's trainer, with the eval job re-importing THIS module."""
    return _pixreg_build_trainer_config(common, module_path=MODULE_PATH)


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
