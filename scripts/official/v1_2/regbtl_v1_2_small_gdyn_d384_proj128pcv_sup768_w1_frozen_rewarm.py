"""SMALL frozen d384 teacher, student continued at a RE-WARMED LR.

The small-backbone twin of
``regbtl_v1_2_gdyn_d768_proj128lin_sup768_w1_frozen_rewarm``; see
``regbtl_v1_2_frozen_student_common`` for the freeze and its motivation.

Why this arm is worth the GPUs. On PASTIS alone the small student looks ordinary
(0.5605, fourth), but across the full 26-task aligned suite it wins 16/26 tasks --
more than any other d128 run -- at -0.41 pts overall and +0.76 excluding the
ethiopia family. It does that from a ViT-Small backbone, i.e. roughly a quarter
of the base encoder's cost, which is the whole point of the small arms: if the
student's ceiling is set by distillation quality rather than teacher capacity,
the cheap backbone should keep most of the quality.

There is headroom above the baseline to aim at: this run's own d384 teacher
scores +1.78 pts over the aligned d128 baseline while its d128 student sits at
-0.41, so the student-teacher gap is ~2.2 pts and closing even half of it puts
the shipped 128-d embedding clearly ahead. (Note this is the AGGREGATE picture.
On PASTIS specifically the small teacher is only 0.6253, so even best-observed
retention lands at ~0.570 against the baseline's 0.5853 -- the small arms cannot
win on PASTIS no matter how well the student tracks. Their case rests on the
suite, not on that one task.)

LR: 2e-4, this backbone's peak from the size sweep's best-LR search -- NOT the
1e-4 the d768 arms use. The cosine floor it ended at was 2e-5.

Only the re-warmed arm runs for the small backbone. The d768 pair
(``_frozen_rewarm`` / ``_frozen_floor``) already separates the LR change from the
freeze, so a second pair here would buy a duplicate answer; if this arm moves,
that pair says which mechanism moved it.

Note sup768 means no supervision heads on the student, so the freeze helper's
student globs cover every trainable parameter. A supboth arm would need its
student supervision heads added to STUDENT_PARAM_GLOBS, or they would sit frozen
while the projection trained.
"""

import logging

from base import build_trainer_config as _base_build_trainer_config
from olmo_core.train.common import Duration
from regbtl_v1_2_faster_common import build_faster_train_module_config
from regbtl_v1_2_frozen_student_common import (
    REWARM_WARMUP_STEPS,
    freeze_all_but_student,
)
from regbtl_v1_2_gdyn_d768_il_pdproj_noic_lsa_1fwd import (
    build_common_components,
    build_dataset_config,
    build_visualize_config,
)
from regbtl_v1_2_newsampling_common import apply_microbatch
from regbtl_v1_2_proj_common import add_proj_loop_eval_beaker_job
from regbtl_v1_2_small_gdyn_d384_proj128pcv_sup768_w1_newsampling_psuniform import (
    build_dataloader_config,
    build_model_config,
)

from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.train.train_module.latent_mim import LatentMIMTrainModuleConfig

logger = logging.getLogger(__name__)

MODULE_PATH = (
    "scripts/official/v1_2/"
    "regbtl_v1_2_small_gdyn_d384_proj128pcv_sup768_w1_frozen_rewarm.py"
)

# The small preset's peak LR (regbtl_v1_2_proj_common.SMALL_LEARNING_RATE), not
# the d768 arms' 1e-4. apply_small_learning_rate() is deliberately NOT called:
# the shared LR is zeroed by the freeze, so the only LR that matters is the
# student group's, set here.
STUDENT_LR = 2e-4
EXTENSION = Duration.epochs(50)


def build_train_module_config(common: CommonComponents) -> LatentMIMTrainModuleConfig:
    """Parent's train module with everything but the student frozen."""
    return freeze_all_but_student(
        apply_microbatch(build_faster_train_module_config(common)),
        student_lr=STUDENT_LR,
        warmup_steps=REWARM_WARMUP_STEPS,
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
