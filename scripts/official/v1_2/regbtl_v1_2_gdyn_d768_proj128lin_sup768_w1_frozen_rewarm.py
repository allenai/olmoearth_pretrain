"""Frozen d768 teacher, student continued at a RE-WARMED LR (arm: frozen_rewarm).

Continues ``lin_sup768_w1`` from step667200 with everything except the detached
student frozen -- see ``regbtl_v1_2_frozen_student_common`` for why the teacher is
not worth training further and how the freeze is implemented.

This is the RE-WARMED half of a pair. The student ended the parent run at 1e-5
(the ``CosWithWarmup`` alpha_f=0.1 floor); here it goes back to its original 1e-4
peak and holds. Rationale: the LR schedule that produced that floor was chosen
for an encoder converging on a stationary objective, and the student's situation
is now the opposite of the one a small LR protects against -- the target has
stopped moving, so the job is to CONVERGE rather than to track, and 1e-5 makes
that slow enough to be untestable in a short extension.

The sibling ``_frozen_floor`` holds 1e-5. Running both keeps the LR change
attributable instead of folded into the freeze, exactly as the ``flatlr`` pair did
for the schedule question. That precedent is also the reason not to assume the LR
is the treatment: ``flatlr_w1`` held its student's LR up for 240k extra steps and
gained nothing (peak 0.5512 @200k, still 0.5512 @440k). Stationarity is the
hypothesis under test; LR is the supporting change.

Bar to beat: native d128 = 0.5853 and the parent itself = 0.5812, both on aligned
PASTIS ps=1 S2 from the 20260806 sweep.
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
from regbtl_v1_2_gdyn_d768_proj128lin_sup768_w1_newsampling_psuniform import (
    build_dataloader_config,
    build_model_config,
)
from regbtl_v1_2_newsampling_common import apply_microbatch
from regbtl_v1_2_proj_common import add_proj_loop_eval_beaker_job

from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.train.train_module.latent_mim import LatentMIMTrainModuleConfig

logger = logging.getLogger(__name__)

MODULE_PATH = (
    "scripts/official/v1_2/regbtl_v1_2_gdyn_d768_proj128lin_sup768_w1_frozen_rewarm.py"
)

STUDENT_LR = 1e-4
# Extension budget, NOT a new total: the launch passes
# --trainer.load_trainer_state=False, so the step counter restarts at 0 and this
# is how far past the parent's 665k we go. ~111k steps, about a day at the
# observed ~120k steps/day.
EXTENSION = Duration.epochs(50)


def build_train_module_config(common: CommonComponents) -> LatentMIMTrainModuleConfig:
    """Parent's train module with everything but the student frozen."""
    return freeze_all_but_student(
        apply_microbatch(build_faster_train_module_config(common)),
        student_lr=STUDENT_LR,
        warmup_steps=REWARM_WARMUP_STEPS,
    )


def build_trainer_config(common: CommonComponents):
    """Parent trainer, shortened, with a tighter eval cadence.

    10k rather than the parent's 40k: the run is a quarter the length and the
    question is the SHAPE of the student's trajectory in the first tens of
    thousands of steps, not its endpoint.
    """
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
