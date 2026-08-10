"""d128 wideread + register supervision + latlon (w0p1) with the decorrelated sampler.

Same recipe as ``regbtl_v1_2_gdyn_d128_il_pdproj_noic_lsa_wideread_regsup_latlon`` at
the w0p1 supervision weight (this is the current best embedding model), but the
dataloader samples the timestep count independently of the spatial grid (biased
toward the full sequence), enforces a token floor, and oversamples the ps=1
deployment resolution. See ``regbtl_v1_2_newsampling_common`` for the exact knobs.
rank_microbatch_size is halved to fit the larger token budget.
"""

import logging

from base import build_trainer_config as _base_build_trainer_config
from regbtl_v1_2_common import add_loop_eval_beaker_job
from regbtl_v1_2_faster_common import build_wideread_regbtl_model_config
from regbtl_v1_2_gdyn_d768_il_pdproj_noic_lsa_1fwd import (
    build_common_components,
    build_visualize_config,
)
from regbtl_v1_2_newsampling_common import (
    SUPERVISION_BASE_WEIGHT,
    apply_microbatch,
    apply_new_sampling,
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

REGISTER_DIM = 128
MODULE_PATH = "scripts/official/v1_2/regbtl_v1_2_gdyn_d128_wideread_regsup_latlon_w0p1_newsampling.py"


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """d128 wideread + register-grid supervision incl. latlon, at w0p1 (base_weight 0.1)."""
    config = build_wideread_regbtl_model_config(
        common, latent_self_attn=True, register_dim=REGISTER_DIM
    )
    return add_register_supervision(
        config, include_latlon=True, base_weight=SUPERVISION_BASE_WEIGHT
    )


def build_dataloader_config(common: CommonComponents):
    """Latlon-aware dataloader with the decorrelated shape sampler applied."""
    return apply_new_sampling(build_latlon_dataloader_config(common))


def build_train_module_config(common: CommonComponents):
    """Latlon-aware faster train module with a halved rank microbatch size."""
    return apply_microbatch(build_latlon_train_module_config(common))


def build_trainer_config(common: CommonComponents):
    """Base trainer config + fifty_cities evals routed through a Beaker job."""
    return add_loop_eval_beaker_job(_base_build_trainer_config(common), MODULE_PATH)


def run() -> None:
    """Run the experiment."""
    main(
        common_components_builder=build_common_components,
        model_config_builder=build_model_config,
        train_module_config_builder=build_train_module_config,
        dataset_config_builder=build_latlon_dataset_config,
        dataloader_config_builder=build_dataloader_config,
        trainer_config_builder=build_trainer_config,
        visualize_config_builder=build_visualize_config,
    )


if __name__ == "__main__":
    run()
