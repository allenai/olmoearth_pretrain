"""Frozen d768 teacher, supboth perceiver student, DECAYED (floor) LR.

The arm the frozen-teacher results actually point at. See
``regbtl_v1_2_frozen_student_common`` for the freeze itself.

What the first four frozen runs established, in order:

* The DISTILLATION objective is saturated. The lin students cannot improve it at
  all (flat to 0.4 sigma over 30k steps; weights rotate ~6 degrees with drift
  growing as N^0.31, slower than a random walk, so a restoring force is pulling
  them back). The pcv students can still descend on it -- but on
  ``small_pcv_sup768``, which has no supervision on the student, 20k steps of
  descent moved the frozen probe by +0.03 mIoU, i.e. nothing.
* SUPERVISION is not saturated. ``small_pcv_supboth``, identical apart from
  carrying supervision heads on the student, rose on three independent probe
  series over the same window: S2 proj128 0.5530 -> 0.5561, S2 proj64 0.4907 ->
  0.4947, S1+S2 proj128 0.5613 -> 0.5624. Its frozen teacher returned
  bit-identical numbers across four evals, so the probe is deterministic at 1e-4
  and those moves are real, not scatter. It is now +0.39 above where its parent
  finished, and rising, while its sup768 sibling sits 0.21 BELOW its own parent.

So the live gradient is the supervision term -- the one part of the student's
objective that is not scale-invariant, and the one that forces cell (i, j) to
encode what is at (i, j). What the small supboth arm lacks is a teacher worth
tracking: at 0.6301 on aligned PASTIS it cannot reach the native d128 baseline's
0.5853 even at the best retention ever measured.

This arm supplies that: the same pcv + supboth recipe on the d768 teacher
(0.6393 in-loop against the small arms' 0.6109). It is the only combination with
both a gradient that still moves and a ceiling above the baseline.

Like the small supboth arm, the student's supervision heads live at
``LatentMIM.projection_supervision_heads`` and are NOT matched by
STUDENT_PARAM_GLOBS, so they are added explicitly -- otherwise the freeze would
pin them while the projection they read from kept training, turning the very
term this arm is built around into a stale-head regulariser.
"""

import logging

from base import build_trainer_config as _base_build_trainer_config
from olmo_core.train.common import Duration
from regbtl_v1_2_faster_common import build_faster_train_module_config
from regbtl_v1_2_frozen_student_common import (
    STUDENT_SUPERVISION_HEAD_GLOB,
    freeze_all_but_student,
)
from regbtl_v1_2_gdyn_d768_il_pdproj_noic_lsa_1fwd import (
    build_common_components,
    build_dataset_config,
    build_visualize_config,
)
from regbtl_v1_2_gdyn_d768_proj128pcv_supboth_w1_newsampling_psuniform import (
    build_dataloader_config,
    build_model_config,
)
from regbtl_v1_2_newsampling_common import apply_microbatch
from regbtl_v1_2_proj_common import add_proj_loop_eval_beaker_job

from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.train.train_module.latent_mim import LatentMIMTrainModuleConfig

logger = logging.getLogger(__name__)

MODULE_PATH = (
    "scripts/official/v1_2/regbtl_v1_2_gdyn_d768_proj128pcv_supboth_w1_frozen_floor.py"
)

STUDENT_LR = 1e-5
# ConstantWithWarmup divides by warmup_steps, so 1 means "no warmup".
NO_WARMUP_STEPS = 1
EXTENSION = Duration.epochs(50)


def build_train_module_config(common: CommonComponents) -> LatentMIMTrainModuleConfig:
    """Parent's train module with everything but the student and its heads frozen."""
    return freeze_all_but_student(
        apply_microbatch(build_faster_train_module_config(common)),
        student_lr=STUDENT_LR,
        warmup_steps=NO_WARMUP_STEPS,
        extra_trainable_globs=[STUDENT_SUPERVISION_HEAD_GLOB],
    )


def build_trainer_config(common: CommonComponents):
    """Parent trainer, shortened, with a tighter eval cadence."""
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
