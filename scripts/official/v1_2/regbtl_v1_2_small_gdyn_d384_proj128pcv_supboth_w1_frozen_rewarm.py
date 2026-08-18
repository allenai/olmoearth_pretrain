"""SMALL frozen d384 teacher, supboth student continued at a RE-WARMED LR.

Sibling of ``regbtl_v1_2_small_gdyn_d384_proj128pcv_sup768_w1_frozen_rewarm``;
see ``regbtl_v1_2_frozen_student_common`` for the freeze and its motivation.

Why extend this one too. It is the best-retaining arm anywhere measured -- 90.8%
of its teacher on aligned PASTIS S2 and 91.2% on S1+S2, against 89.4% for the
best d768-teacher arm, and 97.8% averaged over the AEF suite. Retention is the
quantity the frozen-teacher runs exist to move, so the arm that already retains
best is the natural place to ask how much further it goes when the target stops
moving. It was also still climbing at 640k when the parent hit its 300-epoch
wall (peak at the final eval, +0.20 mIoU/100k), which its sup768 sibling was not.

The supboth caveat, and the one config difference from the sibling. supboth hangs
per-prefix supervision heads off the STUDENT
(``LatentMIM.projection_supervision_heads``). Those are student parameters, but
proj_common's STUDENT_PARAM_GLOBS does not match them, so the default freeze
would pin them while the projection they read from kept training -- the
supervision term would degrade into a stale-head regulariser pulling against the
student. ``extra_trainable_globs`` adds them back.

That extra trainable surface is also a confound worth naming: this arm changes
two things against its sup768 sibling (supervision placement AND a slightly
larger trainable set), so read the pair as two candidates rather than a clean
A/B of supervision placement.

Ceiling, stated plainly: like every small arm this one cannot win on PASTIS. Its
teacher is 0.6301 there, so even best-observed retention lands at ~0.575 against
the aligned baseline's 0.5853. Its case is the 26-task suite, where it wins 13/26
at -0.73 pts overall and +0.62 excluding the ethiopia family, from a backbone
costing roughly a quarter of base.

LR: 2e-4, this backbone's peak, not the d768 arms' 1e-4.
"""

import logging

from base import build_trainer_config as _base_build_trainer_config
from olmo_core.train.common import Duration
from regbtl_v1_2_faster_common import build_faster_train_module_config
from regbtl_v1_2_frozen_student_common import (
    REWARM_WARMUP_STEPS,
    STUDENT_SUPERVISION_HEAD_GLOB,
    freeze_all_but_student,
)
from regbtl_v1_2_gdyn_d768_il_pdproj_noic_lsa_1fwd import (
    build_common_components,
    build_dataset_config,
    build_visualize_config,
)
from regbtl_v1_2_newsampling_common import apply_microbatch
from regbtl_v1_2_proj_common import add_proj_loop_eval_beaker_job
from regbtl_v1_2_small_gdyn_d384_proj128pcv_supboth_w1_newsampling_psuniform import (
    build_dataloader_config,
    build_model_config,
)

from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.train.train_module.latent_mim import LatentMIMTrainModuleConfig

logger = logging.getLogger(__name__)

MODULE_PATH = (
    "scripts/official/v1_2/"
    "regbtl_v1_2_small_gdyn_d384_proj128pcv_supboth_w1_frozen_rewarm.py"
)

STUDENT_LR = 2e-4
EXTENSION = Duration.epochs(50)


def build_train_module_config(common: CommonComponents) -> LatentMIMTrainModuleConfig:
    """Parent's train module with everything but the student (and its heads) frozen."""
    return freeze_all_but_student(
        apply_microbatch(build_faster_train_module_config(common)),
        student_lr=STUDENT_LR,
        warmup_steps=REWARM_WARMUP_STEPS,
        extra_trainable_globs=[STUDENT_SUPERVISION_HEAD_GLOB],
    )


def build_trainer_config(common: CommonComponents):
    """Parent trainer, shortened, with a tighter eval cadence (see the d768 arm)."""
    config = add_proj_loop_eval_beaker_job(
        _base_build_trainer_config(common),
        MODULE_PATH,
        embedding_eval_interval_steps=10000,
    )
    config.max_duration = EXTENSION
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
