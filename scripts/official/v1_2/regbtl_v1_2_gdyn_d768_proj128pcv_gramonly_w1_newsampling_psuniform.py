"""d768 teacher + detached pcv student, purely within-scene Gram (arm: gramonly).

One of five single-change arms on ``proj128pcv_sup768_w1_newsamp_psuniform``, which
is the shared baseline and is already in flight -- no arm re-runs it.

The far endpoint of the Gram mix axis: the flat (cross-scene) relational term is
switched OFF entirely and the whole relational budget goes to the block-diagonal,
within-scene term. With the baseline at 1.0/0.0 and ``gramwithin`` at 0.5/0.5, this
arm completes three points on that axis, so the effect can be read as a shape rather
than inferred from a single guessed weight.

Why the endpoint is worth its own run. The hypothesis is specifically that
segmentation depends on within-scene discrimination and that those pairs were too
rare in the flat Gram (~1/64 of them) to shape the representation. 0.0/1.0 tests that
claim directly, with nothing else contributing. It is also the arm most likely to
separate early if the hypothesis is right, and most likely to regress if
cross-scene structure is doing real work.

The risk it carries, stated plainly: dropping cross-scene pairs removes the signal
that ties different locations to each other, which is what retrieval and the global
classification tasks would use. The in-loop evals here are PASTIS segmentation and
fifty_cities, so a regression of that kind would be largely invisible -- if this arm
wins, its 128d embedding should be checked on the classification suite before it is
treated as the recipe.
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

ARM = STUDENT_ARMS["gramonly"]
MODULE_PATH = (
    "scripts/official/v1_2/"
    "regbtl_v1_2_gdyn_d768_proj128pcv_gramonly_w1_newsampling_psuniform.py"
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
