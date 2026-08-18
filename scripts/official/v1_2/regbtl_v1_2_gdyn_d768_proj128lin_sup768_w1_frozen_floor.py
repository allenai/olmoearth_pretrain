"""Frozen d768 teacher, student continued at the DECAYED LR (arm: frozen_floor).

The control half of the frozen-teacher pair -- see
``regbtl_v1_2_frozen_student_common`` for the freeze and
``..._frozen_rewarm`` for the motivation.

This arm changes exactly one thing about the parent run: the teacher stops
training. The student stays at 1e-5, the ``CosWithWarmup`` alpha_f=0.1 floor it
reached at 665k, so any gain here is attributable to STATIONARITY alone rather
than to the larger step the sibling takes. That matters because the parent's
+0.71 mIoU/100k trailing slope was measured at this LR: if the artifact story is
right and target motion was the limit, this arm should accelerate past that slope
without touching the LR at all.

Warmup is 1 step, i.e. none -- ``ConstantWithWarmup`` divides by warmup_steps so
0 is not allowed, and the student is already at this LR so there is nothing to
ramp into. Fresh Adam moments (``--trainer.load_optim_state=False``, required
because adding the student param group changes the optimizer's group structure)
are the only transient.
"""

import logging

from base import build_trainer_config as _base_build_trainer_config
from olmo_core.train.common import Duration
from regbtl_v1_2_faster_common import build_faster_train_module_config
from regbtl_v1_2_frozen_student_common import freeze_all_but_student
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
    "scripts/official/v1_2/regbtl_v1_2_gdyn_d768_proj128lin_sup768_w1_frozen_floor.py"
)

# Where the parent's cosine schedule left the student: 10% of the 1e-4 peak.
STUDENT_LR = 1e-5
NO_WARMUP_STEPS = 1
EXTENSION = Duration.epochs(50)


def build_train_module_config(common: CommonComponents) -> LatentMIMTrainModuleConfig:
    """Parent's train module with everything but the student frozen."""
    return freeze_all_but_student(
        apply_microbatch(build_faster_train_module_config(common)),
        student_lr=STUDENT_LR,
        warmup_steps=NO_WARMUP_STEPS,
    )


def build_trainer_config(common: CommonComponents):
    """Parent trainer, shortened, with a tighter eval cadence (see the sibling)."""
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
