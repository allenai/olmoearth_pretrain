"""v1.2 register bottleneck noic_lsa d768 + register supervision + latlon at 10x weight.

Identical to ``regbtl_v1_2_gdyn_d768_il_pdproj_noic_lsa_regsup_latlon.py`` except the
supervision base weight is raised 10x (0.01 -> 0.1); the per-task balancing
(``TASK_TYPE_WEIGHTS``) is unchanged, so effective weights become regression 0.1
(incl. latlon), classification/BCE 0.01. Location stays a supervised TARGET, never a
model input. Trains at the base.py default 300 epochs.

In-loop evals run as separate Beaker jobs (run_as_beaker_job=True) and include the
fifty_cities random-split S2 + S1+S2 probes.
"""

import logging

from base import build_trainer_config as _base_build_trainer_config
from regbtl_v1_2_common import add_loop_eval_beaker_job
from regbtl_v1_2_faster_common import build_wideread_regbtl_model_config
from regbtl_v1_2_gdyn_d768_il_pdproj_noic_lsa_1fwd import (
    build_common_components,
    build_visualize_config,
)
from regbtl_v1_2_regsup_common import (
    add_register_supervision,
    build_latlon_dataloader_config,
    build_latlon_dataset_config,
    build_latlon_train_module_config,
)

from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig

logger = logging.getLogger(__name__)

# Eval entrypoints (all_evals.py / checkpoint_sweep_evals.py) look up this exact
# name on the module to rebuild the train module when loading the checkpoint.
build_train_module_config = build_latlon_train_module_config

REGISTER_DIM = 768
# 10x the regsup_common SUPERVISION_WEIGHT nudge (0.01 -> 0.1).
SUPERVISION_BASE_WEIGHT = 0.1
MODULE_PATH = "scripts/official/v1_2/regbtl_v1_2_gdyn_d768_il_pdproj_noic_lsa_regsup_latlon_w0p1.py"


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """d768 frontier + register supervision incl. latlon head, at the 10x (0.1) weight."""
    config = build_wideread_regbtl_model_config(
        common, latent_self_attn=True, register_dim=REGISTER_DIM
    )
    return add_register_supervision(
        config, include_latlon=True, base_weight=SUPERVISION_BASE_WEIGHT
    )


def build_trainer_config(common: CommonComponents):
    """Base trainer config + fifty_cities evals routed through a Beaker job."""
    return add_loop_eval_beaker_job(_base_build_trainer_config(common), MODULE_PATH)


def run() -> None:
    """Run the experiment."""
    main(
        common_components_builder=build_common_components,
        model_config_builder=build_model_config,
        train_module_config_builder=build_latlon_train_module_config,
        dataset_config_builder=build_latlon_dataset_config,
        dataloader_config_builder=build_latlon_dataloader_config,
        trainer_config_builder=build_trainer_config,
        visualize_config_builder=build_visualize_config,
    )


if __name__ == "__main__":
    run()
