"""New-maps d768 wideread + regsup, GLO30 supervised on elevation AND slope.

Exact twin of ``regbtl_v1_2_gdyn_d768_wideread_regsup_w0p1_newsampling_psuniform``
(same DN h5, model, masking, losses, evals) with a SINGLE change to the register-grid
supervision head: the GLO30 DSM target is regressed on **elevation + slope** (bands
0 and 1) instead of elevation alone.

Slope (deg [0, 90)) is a well-behaved continuous target, so it uses the same L1
regression as elevation. The circular ``aspect`` band (deg [0, 360), -1 = flat) is
deliberately left unsupervised: plain L1/MSE is lossy on a wrap-around angle and would
need a sin/cos encoding to be correct.
"""

import logging

from base import build_common_components, build_dataset_config, build_visualize_config
from base import build_trainer_config as _base_build_trainer_config
from perceiver_common import (
    SUPERVISION_BASE_WEIGHT,
    apply_microbatch,
    apply_new_sampling,
    apply_uniform_patch_sizes,
    build_1fwd_dataloader_config,
    build_faster_train_module_config,
    build_supervision_head_config,
    build_wideread_regbtl_model_config,
    route_loop_evals_to_beaker,
)

from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig
from olmoearth_pretrain.nn.supervision_head import (
    SupervisionModalityConfig,
    SupervisionTaskType,
)
from olmoearth_pretrain.train.train_module.latent_mim import LatentMIMTrainModuleConfig

logger = logging.getLogger(__name__)

REGISTER_DIM = 768
MODULE_PATH = (
    "scripts/vnext/2026_07_24_new_maps/"
    "regbtl_v1_2_gdyn_d768_wideread_regsup_w0p1_newsampling_psuniform_glo30_elev_slope.py"
)

# GLO30 band order is [elevation, slope, aspect]; supervise elevation + slope only.
GLO30_ELEV_SLOPE_BAND_INDICES = [0, 1]


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """d768 wideread + regsup, with GLO30 supervised on elevation + slope."""
    config = build_wideread_regbtl_model_config(
        common, latent_self_attn=True, register_dim=REGISTER_DIM
    )
    head = build_supervision_head_config(base_weight=SUPERVISION_BASE_WEIGHT)
    # Swap the GLO30 head from elevation-only to elevation + slope (L1, 2 channels).
    old_glo30 = head.modality_configs["glo30"]
    head.modality_configs["glo30"] = SupervisionModalityConfig(
        task_type=SupervisionTaskType.REGRESSION,
        num_output_channels=len(GLO30_ELEV_SLOPE_BAND_INDICES),
        weight=old_glo30.weight,
        regression_loss_type="l1",
        target_band_indices=GLO30_ELEV_SLOPE_BAND_INDICES,
    )
    config.supervision_head_config = head
    return config


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
