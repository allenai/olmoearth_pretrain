"""d768 wideread + regsup (newmaps, Landsat reflectance) with GROUPED band dropout.

Twin of ``regbtl_v1_2_gdyn_d768_wideread_regsup_w0p1_newsampling_psuniform_landsat_refl``
(same Landsat-reflectance h5 + reflectance norm, same model / masking / losses / evals)
with the ONE change being the band-dropout structure:

* instead of per-band random dropout (zero each band independently ~Uniform(0, 0.2)),
  band dropout is now *grouped* -- with probability ``GROUP_BAND_DROPOUT_RATE`` one whole
  resolution group is dropped per sample (S2 10 m / 20 m / 60 m; Landsat with/without the
  panchromatic B8). Within-group bands stay co-present, preserving the fine spectral
  contrast that per-band dropout smears away.

The per-pixel patch-embed MLP stays at the base ``[64]`` (no linear skip): this run
isolates the grouped-vs-per-band dropout change on its own.
"""

import logging

from base import (
    BAND_DROPOUT_GROUPS,
    GROUP_BAND_DROPOUT_RATE,
    build_common_components,
    build_visualize_config,
)
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

from olmoearth_pretrain.data.dataset import OlmoEarthDatasetConfig
from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig
from olmoearth_pretrain.train.train_module.latent_mim import LatentMIMTrainModuleConfig

logger = logging.getLogger(__name__)

REGISTER_DIM = 768
MODULE_PATH = (
    "scripts/vnext/2026_07_24_new_maps/"
    "regbtl_v1_2_gdyn_d768_wideread_regsup_w0p1_newsampling_psuniform_landsat_refl_"
    "groupdrop.py"
)

# New-maps h5 whose Landsat modality is TOA reflectance / brightness temperature.
LANDSAT_REFL_H5_DIR = (
    "/weka/dfive-default/helios/dataset/osm_sampling_landsat_refl/"
    "h5py_data_w_missing_timesteps_zstd_3_128_x_4/"
    "cdl_glo30_landsat_meta_canopy_height_openstreetmap_raster_"
    "sentinel1_sentinel2_l2a_worldcereal_worldcover/1138828"
)
# Reflectance-scale Landsat norm stats.
LANDSAT_REFL_NORM_CONFIG = "computed_landsat_reflectance.json"


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """d768 wideread + regsup, with grouped (resolution-group) band dropout."""
    config = build_wideread_regbtl_model_config(
        common, latent_self_attn=True, register_dim=REGISTER_DIM
    )
    config = add_register_supervision(config, base_weight=SUPERVISION_BASE_WEIGHT)
    # Switch band dropout from per-band to grouped for S2 / Landsat.
    config.encoder_config.band_dropout_groups = BAND_DROPOUT_GROUPS
    config.encoder_config.band_dropout_rate = GROUP_BAND_DROPOUT_RATE
    return config


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
