"""The v1.3 RC with per-pixel Perceiver latents: random latent stride under a 512 budget.

``base.py`` unchanged (12 ViT blocks, the 4-layer register Perceiver, supervision, the
``[128, 64]`` student, sampler, train module) except for the register grid: instead of
one latent per patch, the Perceiver lays its latents at a stride ``s`` that divides the
batch's patch size, so a patch-size-4 batch can carry 1, 4 or 16 latents per patch.
In training ``s`` is drawn per forward pass, uniformly over the divisors whose latent
count fits ``MAX_LATENTS`` (the patch stride is always allowed, so this is the RC
whenever no finer stride fits); at eval ``s`` = 1, one latent per pixel. With ``s`` =
patch size the model is exactly the RC (checked to float rounding on the RC encoder).

This is the pixel-latent convention of ``pix512_*`` and the joint random-stride arms,
on the RC's own backbone: the question is whether the RC can be run at a coarse token
patch size and still produce per-pixel embeddings, without a new backbone.

Compute (MACs, 16x16 / 12 timesteps / S1+S2+L8, per-pixel latents at eval): ps1
2,449 G (= RC), ps2 329 G (RC 316 G at 8x8), ps4 74.2 G (RC 60.5 G at 4x4), ps8
27.7 G. The 13-task protocol costs 238 G with patch-stride latents at eval, 5,298 G
with one latent per pixel.

IN-LOOP EVALS: as ``pix512_*`` (``pure_perceiver_mix.build_mix_trainer_config``):
student AEF + PASTIS at patch size 1 (d128 / d64), m-eurosat + pastis on the d768
registers, and the d128 student at patch size 4 per pixel on AEF + PASTIS.

W&B project ``20260921_perceiver_shapes``; trained as ``v1_3_rc_pix512``.
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
from base import build_model_config as _base_build_model_config  # noqa: E402
from pure_perceiver_mix import build_mix_trainer_config  # noqa: E402

from olmoearth_pretrain.internal.experiment import CommonComponents, main  # noqa: E402
from olmoearth_pretrain.nn.flexi_vit import PerceiverConfig  # noqa: E402
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig  # noqa: E402

logger = logging.getLogger(__name__)

MODULE_PATH = "scripts/official/v1_3/ablations/rc_pix512.py"

# Per-pixel latents: random stride under this budget in training (uniform), stride 1 at eval.
MAX_LATENTS = 512


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """base.py's model with per-pixel, random-stride Perceiver latents."""
    config = _base_build_model_config(common)
    perceiver = config.encoder_config.perceiver_config
    assert isinstance(perceiver, PerceiverConfig)
    perceiver.pixel_latents = True
    perceiver.random_latent_stride = True
    perceiver.max_latents = MAX_LATENTS
    perceiver.eval_latent_stride = 1
    return config


def build_trainer_config(common: CommonComponents):
    """The pix512 arms' evals (incl. ps4 per pixel), re-importing THIS module."""
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
