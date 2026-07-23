"""Single-forward-pass noic_lsa run, but with a FUSED AdamW optimizer.

Identical to ``regbtl_v1_2_gdyn_d768_il_pdproj_noic_lsa_1fwd`` (one forward pass per batch,
plain non-contrastive train module) except the AdamW optimizer runs its fused kernel
(``fused=True``) instead of the base builder's ``fused=False``. This is purely a speed
optimization -- the fused kernel computes the same update -- so it stacks on top of the
single-forward-pass speedup without changing the training math.

All other model, loss, masking, scheduler, EMA, dataloader (single masked view), and
trainer settings are inherited verbatim from the 1fwd script.

In-loop evals run as separate Beaker jobs (run_as_beaker_job=True) and include the
fifty_cities random-split S2 + S1+S2 probes.
"""

import logging

from base import build_trainer_config as _base_build_trainer_config
from regbtl_v1_2_common import add_loop_eval_beaker_job
from regbtl_v1_2_gdyn_d768_il_pdproj_noic_lsa_1fwd import (
    build_common_components,
    build_dataloader_config,
    build_dataset_config,
    build_model_config,
    build_visualize_config,
)
from regbtl_v1_2_gdyn_d768_il_pdproj_noic_lsa_1fwd import (
    build_train_module_config as _build_1fwd_train_module_config,
)

from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.train.train_module.latent_mim import LatentMIMTrainModuleConfig

logger = logging.getLogger(__name__)

MODULE_PATH = (
    "scripts/official/v1_2/regbtl_v1_2_gdyn_d768_il_pdproj_noic_lsa_1fwd_fusedadamw.py"
)


def build_train_module_config(common: CommonComponents) -> LatentMIMTrainModuleConfig:
    """1fwd train module config, but with the fused AdamW kernel enabled."""
    config = _build_1fwd_train_module_config(common)
    config.optim_config.fused = True
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
