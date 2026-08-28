"""supstu0p1 with the DISTILLATION LOSS REMOVED: the student trains on supervision alone.

One-flag-pair mirror of
``regbtl_v1_2_gdyn_d768_proj128lin_supstu0p1_w1_newsampling_psuniform`` -- same d768
teacher, same lin student at [128, 64], same w1 register supervision, same student
supervision heads at 0.1x, same newsampling psuniform data -- with BOTH distillation
terms switched off (``projection_distill_cosine_weight = 0``,
``projection_distill_gram_weight = 0``; the within-scene Gram term is already 0 in this
lineage). Nothing else changes, so the contrast against the parent is the distillation
objective and nothing else.

WHAT THIS ASKS. Every arm of this family has trained the detached student by
distillation (per-prefix cosine through a learned back-projection + a Gram/relational
term), with supervision as an ADD-ON at 0.1x. The 2x3 gram x head matrix is currently
testing how much of that objective's shape matters; this arm tests whether the
objective is load-bearing AT ALL, by deleting it and leaving the student with only the
signal that reaches it directly -- its own per-prefix supervision heads. The teacher is
untouched either way (the registers are detached before the student sees them, and the
student's loss never reaches the encoder), so this is a pure student-side ablation:
d768 register cells are the same in both runs, only the map into 128 dims differs.

Two ways it can land, both informative. If the student holds up, the shipped embedding
never needed a teacher to imitate and the whole distillation apparatus -- back-projection
heads, Gram matrices, the d64 prefix's duplicate terms -- is machinery we can drop.
If it collapses, we get the first measurement of what distillation is actually worth on
the served width, which the gram sweeps could not give: sixteen gram-SCOPE arms were a
null and ``_gram0`` only removes one of the two terms, so no run has ever scored a
student with no teacher signal at all.

THE 0.1x IS KEPT DELIBERATELY. Supervision is now the student's only loss, so the
temptation is to raise it back to 1.0 -- but under decoupled AdamW the per-parameter
update is essentially invariant to a constant loss scale, and holding the parent's
value keeps this a one-thing-changed comparison rather than two.

The back-projection heads are still BUILT (they are part of the encoder config) and
simply receive no gradient; DDP runs with ``find_unused_parameters=True``, so they ride
along as dead weight in the checkpoint and are discarded at inference as always.

In-loop evals: the parent's year-aligned early-read set on both heads (student tasks
first) -- ``set_proj_earlyread_loop_evals`` -- so the curves overlay the parent's
directly. ``projection/distill_*`` metrics are absent here BY CONSTRUCTION: the terms
are not computed, so their disappearance from W&B is the flag working, not a bug.
"""

import logging

from base import build_trainer_config as _base_build_trainer_config
from regbtl_v1_2_gdyn_d768_il_pdproj_noic_lsa_1fwd import (
    build_common_components,
    build_dataset_config,
    build_visualize_config,
)
from regbtl_v1_2_gdyn_d768_proj128lin_supstu0p1_w1_newsampling_psuniform import (
    build_dataloader_config,
    build_model_config,
)
from regbtl_v1_2_gdyn_d768_proj128lin_supstu0p1_w1_newsampling_psuniform import (
    build_train_module_config as _base_build_train_module_config,
)
from regbtl_v1_2_proj_common import set_proj_earlyread_loop_evals

from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.train.train_module.latent_mim import LatentMIMTrainModuleConfig

logger = logging.getLogger(__name__)

MODULE_PATH = (
    "scripts/official/v1_2/"
    "regbtl_v1_2_gdyn_d768_proj128lin_supstu0p1_w1_newsampling_psuniform_nodistill.py"
)


def build_train_module_config(common: CommonComponents) -> LatentMIMTrainModuleConfig:
    """The parent's train module with both distillation terms zeroed."""
    config = _base_build_train_module_config(common)
    config.projection_distill_cosine_weight = 0.0
    config.projection_distill_gram_weight = 0.0
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
