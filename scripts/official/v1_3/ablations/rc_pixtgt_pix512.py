"""``rc_pix512`` with PIXEL-resolution MIM targets, one random pixel per token footprint.

Everything is ``rc_pix512.py`` (the RC, ``base.py``, with per-pixel random-stride
Perceiver latents under a 512 budget, one latent per pixel at eval) except the MIM
targets. Each training step, every token cell (the ``p x p`` pixel footprint of one
token) draws ONE pixel uniformly at random, shared across that cell's timesteps, band
sets and modalities. Every masked token of the cell keeps its decoder query, but the
query's 2D-RoPE coordinate moves to the drawn pixel's center (the decoder's only
spatial signal; it lands exactly on that pixel's latent at stride 1), and its target is
the frozen projection of that single pixel (the projection-only target at
``patch_size=1``) instead of the whole patch. See ``olmoearth_pretrain/nn/pixel_targets.py``.

One target and one decoder query per masked token, exactly as in the RC, so the
decoder cost does not change; at patch size 1 nothing changes at all.

WHY: with per-pixel latents the decoder still reads them at patch resolution (queries at
patch coordinates, patch-level targets), so the MIM loss never asks a latent to differ
from its neighbours inside a patch. Pixel targets put that pressure on the MIM objective
itself, without the p**2 query blow-up of decoding every pixel.

W&B project ``20260921_perceiver_shapes``; trained as ``v1_3_rc_pixtgt_pix512`` (the
``pixtgt`` token sits before ``pix512`` so the run name is not ``v1_3_rc_pix512`` plus a
suffix). The 2-layer sibling is ``rc_ld2_pixtgt_pix512.py``.
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
    build_visualize_config,
)
from base import (
    build_train_module_config as _base_build_train_module_config,  # noqa: E402
)
from pure_perceiver_mix import build_mix_trainer_config  # noqa: E402
from rc_pix512 import build_model_config  # noqa: E402

from olmoearth_pretrain.internal.experiment import CommonComponents, main  # noqa: E402
from olmoearth_pretrain.train.train_module.latent_mim import (  # noqa: E402
    LatentMIMTrainModuleConfig,
)

logger = logging.getLogger(__name__)

MODULE_PATH = "scripts/official/v1_3/ablations/rc_pixtgt_pix512.py"


def build_train_module_config(common: CommonComponents) -> LatentMIMTrainModuleConfig:
    """base.py's train module with pixel-resolution (one pixel per cell) MIM targets."""
    config = _base_build_train_module_config(common)
    config.pixel_targets = True
    return config


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
