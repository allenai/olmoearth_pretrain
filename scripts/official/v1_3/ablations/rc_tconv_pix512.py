"""``rc_pix512`` + a thin conv branch that INITIALIZES the per-pixel Perceiver latents.

Everything is ``rc_pix512.py`` (the RC with per-pixel random-stride Perceiver latents
under a 512 budget, one latent per pixel at eval) plus the ``"thinconv"`` pixel branch
of ``pixreg_thinconv_pixrecon`` (``origin/favyen/20260917-pixreg-v1_3``; here
``olmoearth_pretrain/nn/pixel_branch.py``): 4 ConvNeXt-style units (depthwise 3x3 conv
+ pointwise MLP, width 128) over each ``(timestep, band set)`` frame of the time-series
inputs, run once, independent of the trunk. Its final features -- mean-pooled over the
ONLINE ``(timestep, band set, modality)`` units at each cell, then a zero-init linear
projection -- are ADDED to the cloned latent before the first read.

The one change from the pixreg branch: the convolutions run at the resolution of the
LATENT grid, not the pixel grid. Each forward pass first draws its latent stride ``s``
(as ``rc_pix512`` does), averages the ONLINE pixels of every ``s x s`` cell, and
convolves the ``(H / s, W / s)`` frames, so cell ``(i, j)`` initializes latent
``(i, j)``. At eval ``s = 1`` and this is the pixel-resolution branch.

Leakage guard: cells with no ONLINE pixel are zeroed before the first convolution and
the register-init pool is ONLINE-only, so masked band sets and timesteps contribute
nothing. The zero-init handoff makes step 0 exactly ``rc_pix512``.

W&B project ``20260921_perceiver_shapes``; trained as ``v1_3_rc_tconv_pix512``.
Siblings: ``rc_tconv_mnorm_pix512`` (mask-normalized convs), ``rc_tconv_mr5075_pix512``
(within-band-set encode ratio U[0.5, 0.75]), ``rc_tconv_hp10_pix512`` (high-pass
reconstruction) and its control ``rc_tconv_fullrecon10_pix512``, ``rc_tconv_st_pix512``
(space + time convs) and ``rc_tconv_t_pix512`` (time convs only).
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from base import (  # noqa: E402
    build_common_components,
    build_dataloader_config,
    build_dataset_config,
    build_train_module_config,
    build_visualize_config,
)
from pure_perceiver_mix import build_mix_trainer_config  # noqa: E402
from rc_pix512 import build_model_config as _rc_pix512_build_model_config  # noqa: E402

from olmoearth_pretrain.internal.experiment import CommonComponents, main  # noqa: E402
from olmoearth_pretrain.nn.flexi_vit import PerceiverConfig  # noqa: E402
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig  # noqa: E402

logger = logging.getLogger(__name__)

MODULE_PATH = "scripts/official/v1_3/ablations/rc_tconv_pix512.py"

# --- pixel branch (the pixreg thinconv arm, at Dp=128) ---------------------------------
PIXEL_BRANCH_TYPE = "thinconv"
PIXEL_BRANCH_DIM = 128
PIXEL_BRANCH_DEPTH = 4
PIXEL_BRANCH_KERNEL = 3
PIXEL_BRANCH_MLP_RATIO = 4.0


def apply_thin_conv(
    config: LatentMIMConfig,
    mask_normalized: bool = False,
    mixing: str | None = None,
    register_pool: str | None = None,
    time_kernel: int | None = None,
) -> LatentMIMConfig:
    """Attach the thin conv branch to the Perceiver's latent init, in place.

    ``mixing`` / ``register_pool`` / ``time_kernel`` are set only when given, so the
    arms that leave them out keep their configs (and checkpoints) unchanged.
    """
    perceiver = config.encoder_config.perceiver_config
    assert isinstance(perceiver, PerceiverConfig) and perceiver.pixel_latents
    perceiver.pixel_branch_type = PIXEL_BRANCH_TYPE
    perceiver.pixel_branch_dim = PIXEL_BRANCH_DIM
    perceiver.pixel_branch_depth = PIXEL_BRANCH_DEPTH
    perceiver.pixel_branch_kernel = PIXEL_BRANCH_KERNEL
    perceiver.pixel_branch_mlp_ratio = PIXEL_BRANCH_MLP_RATIO
    if mask_normalized:
        perceiver.pixel_branch_mask_normalized = True
    if mixing is not None:
        perceiver.pixel_branch_mixing = mixing
    if register_pool is not None:
        perceiver.pixel_branch_register_pool = register_pool
    if time_kernel is not None:
        perceiver.pixel_branch_time_kernel = time_kernel
    return config


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """rc_pix512's model + the thin conv latent init."""
    return apply_thin_conv(_rc_pix512_build_model_config(common))


def build_trainer_config(common: CommonComponents):
    """rc_pix512's evals (incl. ps4 per pixel), re-importing THIS module."""
    return build_mix_trainer_config(common, MODULE_PATH, ps4_student_evals=True)


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
