"""d768 teacher + detached linear [128, 64] student, sup768 w1, STUDENT-only uniformity.

``regbtl_v1_2_gdyn_d768_proj128lin_sup768_w1_newsampling_psuniform`` with AlphaEarth's
batch-uniformity term (S2.2.4) applied to the STUDENT alone:
``projection_uniformity_weight=0.1``, no ``register_unit_norm``, no register-side
uniformity. The teacher (encoder + primary bottleneck) trains byte-identically to the
base run -- the term reads the detached student's output only, so nothing here can
move the ceiling the student inherits.

WHY student-only, rather than the register-side ``sphere_unif0p1`` recipe: the
student IS the shipped 128-d embedding, and the spread it needs is in its own space
(the train module grew ``projection_uniformity_weight`` for exactly this). Putting
the sphere on the teacher instead would change the best-known w1 encoder -- an
unvalidated perturbation of the one thing this lineage is prized for. The uniformity
term normalizes internally, so it is well-defined on the norm-less lin student; it
spreads DIRECTIONS only. The student output is deliberately NOT unit-normed at
training time (a projection-side norm would also change what the Gram distillation
term sees); AEF-parity unit norm, if wanted, is a serving/eval-time L2 -- measured a
null for the probes in the embedding-norm arms.

Weight 0.1 matches the register-side ``sphere_unif0p1`` arm, for comparability.

In-loop evals are the early-read year-aligned set on BOTH heads (teacher + proj128
student), including the aeftrial_* balanced-trial metrics -- see
``set_proj_earlyread_loop_evals``.

A/B partner: ``regbtl_v1_2_gdyn_d768_proj128lin_sup768_w1_newsampling_psuniform``.
One axis of the ndvi x student-uniformity 2x2; the combined cell is
``..._sup768_ndvi_w1_newsampling_psuniform_stuunif0p1``.
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
    SUPERVISION_BASE_WEIGHT_W1,
    build_proj_model_config,
    set_proj_earlyread_loop_evals,
)

from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig
from olmoearth_pretrain.train.train_module.latent_mim import LatentMIMTrainModuleConfig

logger = logging.getLogger(__name__)

# Matches the register-side sphere_unif0p1 arm's weight, an order of magnitude below
# the supervision weight, so the term nudges the student rather than dominating the
# distillation losses.
PROJECTION_UNIFORMITY_WEIGHT = 0.1
MODULE_PATH = (
    "scripts/official/v1_2/"
    "regbtl_v1_2_gdyn_d768_proj128lin_sup768_w1_newsampling_psuniform_stuunif0p1.py"
)


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """d768 teacher + detached linear student, supervision on the d768 registers."""
    return build_proj_model_config(
        common,
        base_weight=SUPERVISION_BASE_WEIGHT_W1,
        projection_type="linear",
        supervision_source="registers",
    )


def build_dataloader_config(common: CommonComponents):
    """Newsampling dataloader at uniform patch sizes."""
    return apply_uniform_patch_sizes(
        apply_new_sampling(_base_build_dataloader_config(common))
    )


def build_train_module_config(common: CommonComponents) -> LatentMIMTrainModuleConfig:
    """1fwd + fused AdamW train module + the student-only uniformity term."""
    config = apply_microbatch(build_faster_train_module_config(common))
    config.projection_uniformity_weight = PROJECTION_UNIFORMITY_WEIGHT
    return config


def build_trainer_config(common: CommonComponents):
    """Base trainer + the year-aligned early-read evals on both heads."""
    return set_proj_earlyread_loop_evals(
        _base_build_trainer_config(common), MODULE_PATH
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
