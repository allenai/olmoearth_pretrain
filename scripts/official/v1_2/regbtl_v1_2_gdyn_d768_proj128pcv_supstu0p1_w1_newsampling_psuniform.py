"""d768 teacher + detached pcv student, supervision head at 0.1x (arm: supstu0p1).

One of five single-change arms on ``proj128pcv_sup768_w1_newsamp_psuniform``, which
is the shared baseline and is already in flight -- no arm re-runs it.

What this tests. Supervising the student head at the teacher's own weight cost
2-5 mIoU on the projected PASTIS probes (``supboth_w1`` vs ``sup768_w1``: -2.1 to
-6.4 for pcv), while 0.1x was roughly neutral. The optimum over {0, 0.1, 1.0} sat at
ZERO -- but those points came from runs whose TEACHER weight moved with the
student's, so the tail was never cleanly measured.

Here the teacher stays at w1 and only the student's head weight varies, via
``projection_supervision_weight_scale``. Without that field the two are built from
one supervision_head_config and ``supervision_source="both"`` forces them equal,
which is exactly what made the earlier evidence ambiguous.

The zero point is the baseline run (no student head), so no extra arm is needed for
it. There is deliberately no scale-1.0 arm: ``supboth_w1`` already measured it and
it is clearly worse. The cost is no internal calibration point -- if both supervision
arms come back neutral, confirm the heads are actually attached via the
``supervision_projection_d{128,64}/<modality>`` losses, which are only logged when a
head exists, rather than assuming the weight simply did not matter.
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

ARM = STUDENT_ARMS["supstu0p1"]
MODULE_PATH = (
    "scripts/official/v1_2/"
    "regbtl_v1_2_gdyn_d768_proj128pcv_supstu0p1_w1_newsampling_psuniform.py"
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
