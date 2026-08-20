"""d128 wideread + regsup at base_weight 1.0 (newmaps) on LANDSAT REFLECTANCE h5.

Exact twin of ``regbtl_v1_2_gdyn_d128_wideread_regsup_w0p1_newsampling_psuniform_landsat_refl``
-- same d128 Perceiver register bottleneck reading at encoder width, same 1fwd +
fused AdamW train module, same decorrelated newsampling at uniform patch sizes, same
reflectance h5 and reflectance-scale Landsat norm stats -- with ONE change:

* register-grid supervision runs at ``base_weight=1.0`` instead of 0.1, a 10x
  stronger pull from the six decode-only map modalities (worldcover, glo30,
  openstreetmap_raster, meta_canopy_height, cdl, worldcereal). ``base_weight``
  scales the per-task-type weights, so every supervised modality moves together
  and the latent-MIM objective is unchanged.

The w0p1 twin is the arm to read this against: it isolates how hard the map
supervision should pull on a compressed (d128) register grid, at fixed radiometry.
"""

import logging

from base import build_common_components, build_visualize_config
from base import build_trainer_config as _base_build_trainer_config
from perceiver_common import (
    add_register_supervision,
    apply_microbatch,
    apply_new_sampling,
    apply_uniform_patch_sizes,
    build_1fwd_dataloader_config,
    build_faster_train_module_config,
    build_wideread_regbtl_model_config,
    route_loop_evals_to_beaker,
)

from olmoearth_pretrain.data.dataset import OlmoEarthDatasetConfig
from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig
from olmoearth_pretrain.train.train_module.latent_mim import LatentMIMTrainModuleConfig

logger = logging.getLogger(__name__)

REGISTER_DIM = 128
# The w1 arm: 10x the w0p1 twin's supervision pull. Set here rather than imported
# from perceiver_common's SUPERVISION_BASE_WEIGHT (0.1) because it is the knob
# this script exists to change.
SUPERVISION_BASE_WEIGHT = 1.0
MODULE_PATH = (
    "scripts/vnext/2026_07_24_new_maps/"
    "regbtl_v1_2_gdyn_d128_wideread_regsup_w1_newsampling_psuniform_landsat_refl.py"
)

# New-maps h5 whose Landsat modality is TOA reflectance / brightness temperature.
LANDSAT_REFL_H5_DIR = (
    "/weka/dfive-default/helios/dataset/osm_sampling_landsat_refl/"
    "h5py_data_w_missing_timesteps_zstd_3_128_x_4/"
    "cdl_glo30_landsat_meta_canopy_height_openstreetmap_raster_"
    "sentinel1_sentinel2_l2a_worldcereal_worldcover/1138828"
)
# Reflectance-scale Landsat norm stats (computed.json with only the landsat entry
# replaced); a resource under olmoearth_pretrain/data/norm_configs.
LANDSAT_REFL_NORM_CONFIG = "computed_landsat_reflectance.json"


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """d128 wideread + new-maps register-grid supervision at w1 (base_weight 1.0)."""
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


def build_dataset_config(common: CommonComponents) -> OlmoEarthDatasetConfig:
    """New-maps dataset on the Landsat-reflectance h5 + matching reflectance norm."""
    return OlmoEarthDatasetConfig(
        h5py_dir=LANDSAT_REFL_H5_DIR,
        training_modalities=common.training_modalities,
        computed_norm_config=LANDSAT_REFL_NORM_CONFIG,
    )


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
