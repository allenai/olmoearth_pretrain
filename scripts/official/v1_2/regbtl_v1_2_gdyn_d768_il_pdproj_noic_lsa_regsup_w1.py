"""v1.2 register bottleneck noic_lsa d768 + register-grid supervision at 100x weight.

Identical to ``regbtl_v1_2_gdyn_d768_il_pdproj_noic_lsa_regsup.py`` except the
supervision base weight is raised 100x (0.01 -> 1.0); the per-task balancing
(``TASK_TYPE_WEIGHTS``) is unchanged, so effective weights become regression 1.0,
classification/BCE 0.1. Upper end of the supervision-weight sweep (nudge -> full
learning signal). Trains at the base.py default 300 epochs.

In-loop evals run as separate Beaker jobs (run_as_beaker_job=True) and include the
fifty_cities random-split S2 + S1+S2 probes.
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
    build_dataloader_config,
    build_dataset_config,
    build_visualize_config,
)
from regbtl_v1_2_regsup_common import add_register_supervision

from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig
from olmoearth_pretrain.train.train_module.latent_mim import LatentMIMTrainModuleConfig

logger = logging.getLogger(__name__)

REGISTER_DIM = 768
# 100x the regsup_common SUPERVISION_WEIGHT nudge (0.01 -> 1.0).
SUPERVISION_BASE_WEIGHT = 1.0
MODULE_PATH = (
    "scripts/official/v1_2/regbtl_v1_2_gdyn_d768_il_pdproj_noic_lsa_regsup_w1.py"
)


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """d768 frontier + register-grid supervision at the 100x (1.0) base weight."""
    config = build_wideread_regbtl_model_config(
        common, latent_self_attn=True, register_dim=REGISTER_DIM
    )
    return add_register_supervision(
        config, include_latlon=False, base_weight=SUPERVISION_BASE_WEIGHT
    )


def build_train_module_config(common: CommonComponents) -> LatentMIMTrainModuleConfig:
    """1fwd + fused AdamW + replicated DP + bf16 autocast."""
    return build_faster_train_module_config(common)


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
