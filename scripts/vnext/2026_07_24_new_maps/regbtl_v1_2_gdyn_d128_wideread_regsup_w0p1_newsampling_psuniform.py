"""New-maps d128 wideread + regsup (w0p1, newsampling, UNIFORM ps), single forward pass.

The official ``scripts/official/v1_2/regbtl_v1_2_gdyn_d128_wideread_regsup_w0p1_newsampling_psuniform``
recipe -- Perceiver register bottleneck at storage width 128 with encoder-width reads,
the faster (1fwd + fused AdamW + ddp/bf16) train module, the decorrelated newsampling
shape sampler at uniform patch sizes, and register-grid supervision at base_weight 0.1 --
run on the NEW maps (GLO30 DSM + Meta Canopy Height) and the new H5 file inherited from
``base.py``. No temporal anchor, no NDVI arm, no latlon arm.

register_dim 128 is the compressed bottleneck (storage width 128, reads at encoder width
768); the d768 twin is the full-width (no-compression) register grid for comparison.
"""

import logging

from base import build_common_components, build_dataset_config, build_visualize_config
from base import build_trainer_config as _base_build_trainer_config
from perceiver_common import (
    SUPERVISION_BASE_WEIGHT,
    add_register_supervision,
    apply_microbatch,
    apply_new_sampling,
    apply_uniform_patch_sizes,
    build_1fwd_dataloader_config,
    build_faster_train_module_config,
    build_wideread_regbtl_model_config,
    route_loop_evals_to_beaker,
)

from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig
from olmoearth_pretrain.train.train_module.latent_mim import LatentMIMTrainModuleConfig

logger = logging.getLogger(__name__)

REGISTER_DIM = 128
MODULE_PATH = (
    "scripts/vnext/2026_07_24_new_maps/"
    "regbtl_v1_2_gdyn_d128_wideread_regsup_w0p1_newsampling_psuniform.py"
)


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """d128 wideread + new-maps register-grid supervision at w0p1 (base_weight 0.1)."""
    config = build_wideread_regbtl_model_config(
        common, latent_self_attn=True, register_dim=REGISTER_DIM
    )
    return add_register_supervision(config, base_weight=SUPERVISION_BASE_WEIGHT)


def build_dataloader_config(common: CommonComponents):
    """Single-view newsampling dataloader with patch-size sampling forced to uniform."""
    return apply_uniform_patch_sizes(
        apply_new_sampling(build_1fwd_dataloader_config(common))
    )


def build_train_module_config(common: CommonComponents) -> LatentMIMTrainModuleConfig:
    """1fwd + fused AdamW + ddp/bf16 train module at the newsampling microbatch size."""
    return apply_microbatch(build_faster_train_module_config(common))


def build_trainer_config(common: CommonComponents):
    """New-maps base trainer, with in-loop evals routed through Beaker jobs."""
    return route_loop_evals_to_beaker(_base_build_trainer_config(common), MODULE_PATH)


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
