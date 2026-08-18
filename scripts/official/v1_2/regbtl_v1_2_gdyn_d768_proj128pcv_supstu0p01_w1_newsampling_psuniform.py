"""d768 teacher + detached pcv student, supervision head at 0.01x (arm: supstu0p01).

One of five single-change arms on ``proj128pcv_sup768_w1_newsamp_psuniform``, which
is the shared baseline and is already in flight -- no arm re-runs it.

The far tail of the supervision-weight curve, and the point never measured. Adding a
student supervision head at the teacher's weight cost 2-5 mIoU; 0.1x was roughly
neutral; 0 (the baseline) was best of the three. That curve is monotone in the
student head's weight, so 0.01x asks whether the neutrality at 0.1x is the start of a
plateau or still trending -- i.e. whether a small head buys anything at all, or
whether zero really is the optimum.

As with the 0.1x arm, the teacher stays at w1 and only the student's head weight
moves (``projection_supervision_weight_scale``), which the earlier ``supboth`` runs
could not do because both heads shared one weight.
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

ARM = STUDENT_ARMS["supstu0p01"]
MODULE_PATH = (
    "scripts/official/v1_2/"
    "regbtl_v1_2_gdyn_d768_proj128pcv_supstu0p01_w1_newsampling_psuniform.py"
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
