"""d768 (full-width) regsup at w1 with the decorrelated sampler at UNIFORM patch sizes.

Identical to ``regbtl_v1_2_gdyn_d768_regsup_w0p1_newsampling_psuniform`` except the
supervision ``base_weight`` is 1.0 instead of 0.1 (100x the committed 0.01, 10x w0p1).

WHY w1 AT FULL WIDTH: on the OLD-sampling d768 arms the supervision weight trend is
split rather than uniformly favourable, so this is worth settling rather than assuming.
At 660k, w1 wins on the spatial/segmentation probes -- pastis 0.5881 vs 0.5748 (w0p1) vs
0.5707 (w0.01), fifty_cities_s2 0.6523 vs 0.6451 vs 0.6416 -- but loses clearly on the
classification probes: eurosat 0.8750 vs 0.9160, so2sat 0.6237 vs 0.6602, geo_ecosystem
0.1958 vs 0.2279. w0p1 is best on 5 of 7 tasks; w1 is best on the 2 that most resemble
the frozen ps=1 PASTIS deployment target. Running both under the new sampler tells us
whether the wider register grid plus full-year shapes changes that tradeoff, or whether
w1's classification cost is intrinsic.

See the w0p1 sibling for the register-width and uniform-patch-size rationale.
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
    apply_new_sampling,
    apply_uniform_patch_sizes,
)
from regbtl_v1_2_regsup_common import add_register_supervision

from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig
from olmoearth_pretrain.train.train_module.latent_mim import LatentMIMTrainModuleConfig

logger = logging.getLogger(__name__)

REGISTER_DIM = 768
# w1: base_weight 1.0 -- 10x the w0p1 arms, 100x the committed SUPERVISION_WEIGHT (0.01).
# Scaled by TASK_TYPE_WEIGHTS per arm inside regbtl_v1_2_regsup_common, same as w0p1.
SUPERVISION_BASE_WEIGHT_W1 = 1.0
# See the w0p1 sibling: 32 rather than 64 for memory headroom at 6x the register width.
# Microbatch size affects only memory, not tokens/step, the loss, or the LR schedule.
RANK_MICROBATCH_SIZE = 32
MODULE_PATH = (
    "scripts/official/v1_2/regbtl_v1_2_gdyn_d768_regsup_w1_newsampling_psuniform.py"
)


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """d768 frontier + register-grid supervision at w1 (base_weight 1.0)."""
    config = build_wideread_regbtl_model_config(
        common, latent_self_attn=True, register_dim=REGISTER_DIM
    )
    return add_register_supervision(
        config, include_latlon=False, base_weight=SUPERVISION_BASE_WEIGHT_W1
    )


def build_dataloader_config(common: CommonComponents):
    """Newsampling dataloader at uniform patch sizes."""
    return apply_uniform_patch_sizes(
        apply_new_sampling(_base_build_dataloader_config(common))
    )


def build_train_module_config(common: CommonComponents) -> LatentMIMTrainModuleConfig:
    """1fwd + fused AdamW train module at the reduced d768 microbatch size."""
    config = build_faster_train_module_config(common)
    config.rank_microbatch_size = RANK_MICROBATCH_SIZE
    return config


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
