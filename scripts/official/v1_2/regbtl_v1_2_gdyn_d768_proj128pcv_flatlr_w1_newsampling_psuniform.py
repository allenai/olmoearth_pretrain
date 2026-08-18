"""d768 teacher + detached pcv student on a FLAT student LR (arm: flatlr).

One of five single-change arms on ``proj128pcv_sup768_w1_newsamp_psuniform``, which
is the shared baseline and is already in flight -- no arm re-runs it.

What this tests. On the baseline the student plateaus near 54 mIoU while the teacher
climbs past 63, and ``projection/distill_cosine_d128`` RISES monotonically (0.004 at
26k -> 0.072 at 356k, ~17x): the student falls further behind a target that is still
improving. Meanwhile it sits in the encoder's param group and inherits its
``CosWithWarmup(alpha_f=0.1)``, so its LR is cut 10x over the run. Decaying to a
floor is right for an encoder converging on a stationary objective and wrong for a
head chasing a live teacher.

This arm gives the student its own param group at the SAME peak LR, on
``ConstantWithWarmup`` whose warmup is copied from the encoder's rather than
restated. Mirroring warmup matters: while the encoder's LR ramps, the teacher
representation moves fastest, and a student at full LR from step 0 would chase its
most unstable target at its largest step. The two schedules are identical through
step 8000 and separate only as the decay bites -- so this isolates schedule SHAPE,
with peak LR held fixed.

Note the distillation WEIGHTS are deliberately not the knob here: AdamW is invariant
to a global rescaling of the loss and the student's parameters see gradient from the
distillation terms alone, so scaling both is a no-op. Only their ratio, or the LR,
does anything -- which is why this arm moves the LR and the gram arms move the ratio.

If every arm plateaus together, the limit is the 128-dim budget or the moving
teacher itself, and the next experiment is offline distillation from a FROZEN
finished teacher -- far cheaper, since it needs no encoder backward.
"""

import logging

from base import build_trainer_config as _base_build_trainer_config
from regbtl_v1_2_faster_common import build_faster_train_module_config
from regbtl_v1_2_gdyn_d768_il_pdproj_noic_lsa_1fwd import (
    build_common_components,
    build_dataset_config,
    build_visualize_config,
)
from regbtl_v1_2_gdyn_d768_il_pdproj_noic_lsa_1fwd import (
    build_dataloader_config as _base_build_dataloader_config,
)
from regbtl_v1_2_newsampling_common import (
    apply_microbatch,
    apply_new_sampling,
    apply_uniform_patch_sizes,
)
from regbtl_v1_2_proj_common import (
    STUDENT_ARMS,
    add_proj_loop_eval_beaker_job,
    apply_arm,
    build_arm_model_config,
)

from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig
from olmoearth_pretrain.train.train_module.latent_mim import LatentMIMTrainModuleConfig

logger = logging.getLogger(__name__)

ARM = STUDENT_ARMS["flatlr"]
MODULE_PATH = (
    "scripts/official/v1_2/"
    "regbtl_v1_2_gdyn_d768_proj128pcv_flatlr_w1_newsampling_psuniform.py"
)


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """d768 teacher + detached perceiver student, with this arm's supervision."""
    return build_arm_model_config(common, ARM)


def build_dataloader_config(common: CommonComponents):
    """Newsampling dataloader at uniform patch sizes."""
    return apply_uniform_patch_sizes(
        apply_new_sampling(_base_build_dataloader_config(common))
    )


def build_train_module_config(common: CommonComponents) -> LatentMIMTrainModuleConfig:
    """1fwd + fused AdamW at the newsampling microbatch, plus this arm's knobs."""
    return apply_arm(apply_microbatch(build_faster_train_module_config(common)), ARM)


def build_trainer_config(common: CommonComponents):
    """Base trainer + in-loop evals on the d768, 128d and 64d heads."""
    return add_proj_loop_eval_beaker_job(
        _base_build_trainer_config(common),
        MODULE_PATH,
        embedding_eval_interval_steps=40000,
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
