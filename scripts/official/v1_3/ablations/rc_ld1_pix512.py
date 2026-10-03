"""``rc_pix512`` with 1 Perceiver ``[read -> self-attend]`` layer instead of the RC's 4.

Everything else is ``rc_pix512.py``: the RC (``base.py``) with per-pixel, random-stride
Perceiver latents under a 512 budget, one latent per pixel at eval. ``rc_ld2_pix512``
scored the same as 4 layers at ps1 AND at ps4 per pixel through 160k (2026-10-03), so
this asks how far the depth can go: a single read, one latent self-attention block,
then the output. 61 G at ps4 per pixel (vs 66 G / 74 G at depth 2 / 4) and 178 G on the
13-task protocol with patch-stride latents (vs 198 G / 238 G).

W&B project ``20260921_perceiver_shapes``; trained as ``v1_3_rc_ld1_pix512`` (the depth
token sits before ``pix512`` so the run name is not ``v1_3_rc_pix512`` plus a suffix).
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
from rc_pix512 import build_model_config as _rc_pix512_model_config  # noqa: E402

from olmoearth_pretrain.internal.experiment import CommonComponents, main  # noqa: E402
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig  # noqa: E402

logger = logging.getLogger(__name__)

MODULE_PATH = "scripts/official/v1_3/ablations/rc_ld1_pix512.py"

# [read -> self-attend] pairs in the register Perceiver (the RC has 4).
REGISTER_LATENT_DEPTH = 1


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """rc_pix512's model with a 1-layer Perceiver."""
    config = _rc_pix512_model_config(common)
    perceiver = config.encoder_config.perceiver_config
    assert perceiver is not None
    perceiver.latent_depth = REGISTER_LATENT_DEPTH
    return config


def build_trainer_config(common: CommonComponents):
    """rc_pix512's evals plus ps2 on PASTIS + 2 AEF tasks, re-importing THIS module."""
    return build_mix_trainer_config(
        common, MODULE_PATH, ps4_student_evals=True, ps2_student_evals=True
    )


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
