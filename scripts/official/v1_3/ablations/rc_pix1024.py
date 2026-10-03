"""``rc_pix512`` with a 1024-latent budget instead of 512.

Everything is ``rc_pix512.py`` (the RC with per-pixel, random-stride Perceiver
latents, one latent per pixel at eval) except ``max_latents = 1024``. In training the
stride is still drawn uniformly over the divisors of the patch size whose latent count
fits the budget, so doubling it lets finer strides fit on larger grids and patch
sizes: training sees fine strides more often, closer to the stride-1 latents used at
eval.

COST: the latent self-attention and the reads scale with the latent count, so steps
that draw a finer stride cost more than in ``rc_pix512``, and peak memory rises.

W&B project ``20260921_perceiver_shapes``; trained as ``v1_3_rc_pix1024``.
Compare against ``rc_pix512``.
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

MODULE_PATH = "scripts/official/v1_3/ablations/rc_pix1024.py"

MAX_LATENTS = 1024


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """rc_pix512's model with a 1024-latent budget."""
    config = _rc_pix512_build_model_config(common)
    perceiver = config.encoder_config.perceiver_config
    assert isinstance(perceiver, PerceiverConfig)
    perceiver.max_latents = MAX_LATENTS
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
