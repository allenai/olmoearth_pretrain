"""d768 teacher + detached pcv student, 50/50 Gram mix (arm: gramwithin).

One of five single-change arms on ``proj128pcv_sup768_w1_newsamp_psuniform``, which
is the shared baseline and is already in flight -- no arm re-runs it.

What this tests. The relational (Gram) term is built over the FLATTENED ``[B * N]``
register grid, so pairs mix scenes freely: at a microbatch of 64, only ~1/64 of its
4.2M pairs relate two cells of the SAME scene -- roughly 65k. The other 98% are
cross-scene, and mostly easy near-orthogonal negatives. But a segmentation probe
discriminates WITHIN a scene, so the pairs the headline metric actually depends on
are a rounding error in the loss that is supposed to teach them.

The block-diagonal form computes the same relational MSE per scene, so every pair is
within-scene -- and it is cheaper per pair by a factor of B, since m blocks of k
cells cost O(m * k^2) against O((m * k)^2) for one flat matrix over the same cells.
At equal pair count it buys ~64x more of the pairs that matter.

This arm splits the relational budget evenly (0.5 flat / 0.5 within), holding the
TOTAL at the 1.0 the baseline uses, so it changes the MIX rather than the amount.
Paired with the ``gramonly`` arm (0.0 / 1.0) and the baseline (1.0 / 0.0), it gives
three points on the mix axis rather than one guess.

Caveat when reading the result: ``distill_gram_d128`` and
``distill_gram_within_d128`` are NOT on a common scale. The flat term is dominated
by pairs that are near-orthogonal in both teacher and student and so easy to match
(hence the ~0.001 values on the baseline), while within-scene pairs carry real
structure. So a 0.5/0.5 weighting need not be 50/50 in gradient terms -- the logged
ratio in the first few thousand steps is what tells you where this arm actually sat.
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

ARM = STUDENT_ARMS["gramwithin"]
MODULE_PATH = (
    "scripts/official/v1_2/"
    "regbtl_v1_2_gdyn_d768_proj128pcv_gramwithin_w1_newsampling_psuniform.py"
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
