"""Frozen d768 teacher, PERCEIVER student continued at a RE-WARMED LR.

The pcv counterpart of
``regbtl_v1_2_gdyn_d768_proj128lin_sup768_w1_frozen_rewarm``; see
``regbtl_v1_2_frozen_student_common`` for the freeze and its motivation.

Why the student ARCHITECTURE turned out to matter. Under a frozen teacher the
two lin arms are provably converged -- their distillation loss is flat to within
0.4 sigma over 30k steps, and their weights rotate only ~6 degrees with drift
growing as N^0.31, slower than a random walk, which means a restoring force is
pulling them back. That is what a small nearly-convex fit does once it is done,
and after 665k parent steps it was done.

The perceiver students behave differently: both small_pcv arms are still
DESCENDING against their frozen teachers at -4.6 and -7.6 sigma. A deep
non-convex bottleneck has capacity the linear map does not, so it keeps finding
improvements the fixed target still admits.

What those arms cannot do is win. Their d384 teachers score 0.6253 / 0.6301 on
aligned PASTIS, so even the best retention ever observed (91.2%) lands ~0.575
against the native d128 baseline's 0.5853. This arm pairs the architecture that
is still learning with a teacher that is strong enough for it to matter: 0.6534
teacher, 0.5698 student, 87.2% retention -- and 90.0% of THAT teacher clears the
baseline outright.

Paired with a sibling on the other LR, exactly as the lin arms were, so the LR
change stays attributable to the freeze rather than folded into it.

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
from regbtl_v1_2_gdyn_d768_proj128pcv_sup768_w1_newsampling_psuniform import (
    build_dataloader_config,
    build_model_config,
)
from regbtl_v1_2_newsampling_common import apply_microbatch
from regbtl_v1_2_proj_common import add_proj_loop_eval_beaker_job

from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.train.train_module.latent_mim import LatentMIMTrainModuleConfig

logger = logging.getLogger(__name__)

MODULE_PATH = (
    "scripts/official/v1_2/regbtl_v1_2_gdyn_d768_proj128pcv_sup768_w1_frozen_rewarm.py"
)

STUDENT_LR = 1e-4
EXTENSION = Duration.epochs(50)


def build_train_module_config(common: CommonComponents) -> LatentMIMTrainModuleConfig:
    """Parent's train module with everything but the student frozen."""
    return freeze_all_but_student(
        apply_microbatch(build_faster_train_module_config(common)),
        student_lr=STUDENT_LR,
        warmup_steps=REWARM_WARMUP_STEPS,
    )


def build_trainer_config(common: CommonComponents):
    """Parent trainer, shortened, with a tighter eval cadence (see the lin arm)."""
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
