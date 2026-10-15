"""``rc_ld1_pixtgt_pix512`` (the v1.3 release) plus a sub-token phase penalty.

At ps2 / latent ps1 the release's latents carry a content-independent offset per
pixel phase inside a token (even vs odd pixel; columns about twice rows): one-pixel
stripes in similarity maps. This run adds ``latent_phase_penalty = 1.0`` times the
share of the register grid's variance explained by the per-phase batch means
(``olmoearth_pretrain/nn/latent_phase.py``), applied on every batch whose drawn latent
stride is finer than the patch size. Nothing else changes, so it is a one-flag A/B
against ``v1_3_rc_ld1_pixtgt_pix512``.

W&B project ``20260921_perceiver_shapes``; trained as ``v1_3_rc_ld1_pixtgt_phase1_pix512``.
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
from pure_perceiver_mix import build_mix_trainer_config  # noqa: E402
from rc_ld1_pix512 import build_model_config  # noqa: E402
from rc_pixtgt_pix512 import (  # noqa: E402
    build_train_module_config as _pixtgt_build_train_module_config,
)

from olmoearth_pretrain.internal.experiment import CommonComponents, main  # noqa: E402
from olmoearth_pretrain.train.train_module.latent_mim import (  # noqa: E402
    LatentMIMTrainModuleConfig,
)

logger = logging.getLogger(__name__)

MODULE_PATH = "scripts/official/v1_3/ablations/rc_ld1_pixtgt_phase1_pix512.py"
LATENT_PHASE_PENALTY = 1.0


def build_train_module_config(common: CommonComponents) -> LatentMIMTrainModuleConfig:
    """The release's train module with the sub-token phase penalty."""
    config = _pixtgt_build_train_module_config(common)
    config.latent_phase_penalty = LATENT_PHASE_PENALTY
    return config


def build_trainer_config(common: CommonComponents):
    """rc_ld1_pixtgt_pix512's evals, re-importing THIS module."""
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
