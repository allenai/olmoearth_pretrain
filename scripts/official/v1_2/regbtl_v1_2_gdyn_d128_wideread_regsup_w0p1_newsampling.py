"""d128 wideread + register supervision (w0p1) with the decorrelated shape sampler.

Same recipe as ``regbtl_v1_2_gdyn_d128_il_pdproj_noic_lsa_wideread_regsup`` at the
w0p1 supervision weight, but the dataloader samples the timestep count independently
of the spatial grid (biased toward the full sequence), enforces a token floor, and
oversamples the ps=1 deployment resolution. See ``regbtl_v1_2_newsampling_common``
for the exact knobs. rank_microbatch_size is halved to fit the larger token budget.
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
)
from regbtl_v1_2_regsup_common import add_register_supervision

from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig
from olmoearth_pretrain.train.train_module.latent_mim import LatentMIMTrainModuleConfig

logger = logging.getLogger(__name__)

REGISTER_DIM = 128
MODULE_PATH = (
    "scripts/official/v1_2/regbtl_v1_2_gdyn_d128_wideread_regsup_w0p1_newsampling.py"
)


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """d128 wideread + register-grid supervision at w0p1 (base_weight 0.1)."""
    config = build_wideread_regbtl_model_config(
        common, latent_self_attn=True, register_dim=REGISTER_DIM
    )
    return add_register_supervision(
        config, include_latlon=False, base_weight=SUPERVISION_BASE_WEIGHT
    )


def build_dataloader_config(common: CommonComponents):
    """Base 1fwd dataloader with the decorrelated shape sampler applied."""
    return apply_new_sampling(_base_build_dataloader_config(common))


def build_train_module_config(common: CommonComponents) -> LatentMIMTrainModuleConfig:
    """1fwd + fused AdamW train module with a halved rank microbatch size."""
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
