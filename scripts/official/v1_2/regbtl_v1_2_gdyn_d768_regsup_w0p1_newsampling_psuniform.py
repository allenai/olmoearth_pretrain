"""d768 (full-width) regsup at w0p1 with the decorrelated sampler at UNIFORM patch sizes.

The full-width register bottleneck on the newsampling recipe: does the sampling gain seen
at d128 extend to the full-sized register grid? ``register_dim=768`` equals the encoder
width, so the wideread builder's ``register_attn_dim`` assignment is a no-op and this is
the plain d768 frontier (it still sets ``projection_only_target``, matching
``regbtl_v1_2_gdyn_d768_il_pdproj_noic_lsa_regsup``).

A/B partners:
* ``regbtl_v1_2_gdyn_d768_il_pdproj_noic_lsa_regsup`` family (OLD sampling) at d768 --
  isolates the sampler at full width. Those runs predate the ps=1 PASTIS evals, so the
  ps=1 axis can only be read against the d128 arms.
* ``regbtl_v1_2_gdyn_d128_wideread_regsup_w0p1_newsampling_psuniform`` -- same sampler and
  weight at d128, isolating register width.
* ``..._d768_regsup_w1_newsampling_psuniform`` -- same run at supervision weight 1.0.

WHY UNIFORM PATCH SIZES (not the committed newsampling ``PATCH_SIZE_PROBS``): the 4-point
P(ps=1) sweep at d128 showed the 0.40 ps=1 oversampling buys nothing on the frozen ps=1
PASTIS probes (flat from 0.125 to 0.70, worse at 1.00) while costing 0.01-0.03 across the
ps=4 evals. The d768 old-sampling baselines are far ahead of every d128 arm precisely on
those ps=4 evals (e.g. pastis 0.576 vs 0.512, fifty_cities 0.637 vs 0.560 at 280k), so
using the oversampled distribution here would handicap the comparison on its most
informative axis. Uniform delivers the same ps=1 gain without that cost.
"""

import logging

from base import build_trainer_config as _base_build_trainer_config
from regbtl_v1_2_common import add_loop_eval_beaker_job
from regbtl_v1_2_faster_common import (
    build_faster_train_module_config,
    build_wideread_regbtl_model_config,
)
from regbtl_v1_2_gdyn_d768_il_pdproj_noic_lsa_1fwd import (
    build_common_components,
    build_dataset_config,
    build_visualize_config,
)
from regbtl_v1_2_gdyn_d768_il_pdproj_noic_lsa_1fwd import (
    build_dataloader_config as _base_build_dataloader_config,
)
from regbtl_v1_2_newsampling_common import (
    SUPERVISION_BASE_WEIGHT,
    apply_microbatch,
    apply_new_sampling,
    apply_uniform_patch_sizes,
)
from regbtl_v1_2_regsup_common import add_register_supervision

from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig
from olmoearth_pretrain.train.train_module.latent_mim import LatentMIMTrainModuleConfig

logger = logging.getLogger(__name__)

REGISTER_DIM = 768
# rank_microbatch_size stays at the newsampling module's 64. The OOMs in this recipe's
# history were at token_budget 6144 and BEFORE the broadcastable key-mask fix; micro 64 @
# 3072 is proven (d128 newsamp_v6 ran to 540k), and the d768 old-sampling arms ran to 660k
# at micro 64 / budget 2250. The extra memory at d768 is confined to the bottleneck's
# register stream -- the encoder pass over the budget-capped patch tokens dominates and is
# identical to d128 -- so there should be real headroom here. If a run does OOM, drop to
# 32: microbatch affects ONLY memory, not tokens/step, the loss, or the LR schedule.
MODULE_PATH = (
    "scripts/official/v1_2/regbtl_v1_2_gdyn_d768_regsup_w0p1_newsampling_psuniform.py"
)


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """d768 frontier + register-grid supervision at w0p1 (base_weight 0.1)."""
    config = build_wideread_regbtl_model_config(
        common, latent_self_attn=True, register_dim=REGISTER_DIM
    )
    return add_register_supervision(
        config, include_latlon=False, base_weight=SUPERVISION_BASE_WEIGHT
    )


def build_dataloader_config(common: CommonComponents):
    """Newsampling dataloader at uniform patch sizes."""
    return apply_uniform_patch_sizes(
        apply_new_sampling(_base_build_dataloader_config(common))
    )


def build_train_module_config(common: CommonComponents) -> LatentMIMTrainModuleConfig:
    """1fwd + fused AdamW train module at the newsampling microbatch size."""
    return apply_microbatch(build_faster_train_module_config(common))


def build_trainer_config(common: CommonComponents):
    """Base trainer config + fifty_cities evals routed through a Beaker job."""
    return add_loop_eval_beaker_job(_base_build_trainer_config(common), MODULE_PATH)


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
