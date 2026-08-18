"""d128 wideread + regsup (w0p1, newsampling, uniform ps) at temporal_bias 6.

``regbtl_v1_2_gdyn_d128_wideread_regsup_w0p1_newsampling_psuniform`` with the temporal
bias pushed from the committed 2.75 to 6, so the timestep draw is skewed much harder
toward the top of its feasible window. Budget stays at 3072 and every other knob is
unchanged, so this is compute-neutral -- same tokens per step, same wall-clock, same
662,700-step schedule as the baseline.

WHY: tests whether fuller-sequence exposure is worth more than the committed bias
already extracts. Being free (bias costs nothing, unlike token_budget which is linear in
cost) makes this the cheap half of the sweep. Pairs with
``..._psuniform_b1536_tb6`` to give the budget effect AT high bias, which is the
interaction the 2x2 exists to measure -- at a fixed budget the bias only reallocates
tokens from space to time, so its value plausibly depends on how much budget there is to
reallocate.

Part of the budget x bias 2x2; see the directory README for the full grid.
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
    apply_shape_sweep,
    apply_uniform_patch_sizes,
)
from regbtl_v1_2_regsup_common import add_register_supervision

from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig
from olmoearth_pretrain.train.train_module.latent_mim import LatentMIMTrainModuleConfig

logger = logging.getLogger(__name__)

REGISTER_DIM = 128
TOKEN_BUDGET = 3072
TEMPORAL_BIAS = 6.0
MODULE_PATH = (
    "scripts/official/v1_2/"
    "regbtl_v1_2_gdyn_d128_wideread_regsup_w0p1_newsampling_psuniform_tb6.py"
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
    """Uniform-patch-size newsampling dataloader at budget 3072, temporal_bias 6."""
    return apply_shape_sweep(
        apply_uniform_patch_sizes(
            apply_new_sampling(_base_build_dataloader_config(common))
        ),
        token_budget=TOKEN_BUDGET,
        temporal_bias=TEMPORAL_BIAS,
    )


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
