"""``rc_ld1_pixtgt_pix512`` (the v1.3 RC) with the map supervision heads deleted.

The direct-supervision ablation for the v1.3 report: the paper claims direct supervision
on the Perceiver's output tokens keeps them spatially local and fine-grained, and the
earlier ``nosup`` arm measured that on the pre-pix512 RC. This repeats it on the shipped
configuration.

THE CHANGE, AND ONLY IT: ``supervision_head_config = None``. The teacher then trains on
the PixLatentMIM Lite loss alone and the student on the distillation losses alone;
everything else (1-layer Perceiver, per-pixel latents under a 512 budget, shared
one-pixel-per-cell MIM targets, student, schedule, in-loop evals) is
``rc_ld1_pixtgt_pix512.py``, so the pair is a one-flag contrast.

W&B project ``20260921_perceiver_shapes``; trained as ``v1_3_rc_ld1_nosup_pixtgt_pix512``
(no existing run name is a prefix of it).
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
from rc_ld1_pix512 import build_model_config as _rc_ld1_model_config  # noqa: E402
from rc_pixtgt_pix512 import build_train_module_config  # noqa: E402

from olmoearth_pretrain.internal.experiment import CommonComponents, main  # noqa: E402
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig  # noqa: E402

logger = logging.getLogger(__name__)

MODULE_PATH = "scripts/official/v1_3/ablations/rc_ld1_nosup_pixtgt_pix512.py"


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """rc_ld1_pixtgt_pix512's model with the map supervision heads removed."""
    config = _rc_ld1_model_config(common)
    config.supervision_head_config = None
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
