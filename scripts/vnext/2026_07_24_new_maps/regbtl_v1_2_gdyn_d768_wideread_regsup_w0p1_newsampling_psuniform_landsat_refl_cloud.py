"""Landsat-reflectance d768 recipe + cloud-aware patch discrimination.

Exact twin of
``regbtl_v1_2_gdyn_d768_wideread_regsup_w0p1_newsampling_psuniform_landsat_refl``
(same Landsat-reflectance h5 + reflectance norm, same model / masking / losses /
evals) with a single delta: the patch-discrimination loss skips DECODER (target)
tokens that are mostly cloud, using precomputed OmniCloudMask cloud maps.

Wiring:
  * The dataset is pointed at the cloud sidecar via ``cloud_cache_dir`` (see
    ``olmoearth_pretrain.data.cloud_mask_cache``); ``default_cache_dir`` derives it
    from ``LANDSAT_REFL_H5_DIR`` so it resolves to the reflectance build's own cache
    (``.../osm_sampling_landsat_refl/cloud_masks_omnicloudmask/...``). That build is an
    independently-sampled dataset, so its sidecar MUST be computed against this exact
    h5 -- a DN-build sidecar does NOT align. OCM itself is invariant to the affine
    DN->reflectance conversion, so cloud-mask quality matches the DN case.
  * ``cloud_apply_to="output"`` drops cloudy DECODER tokens (they can't be
    patch-discrimination targets); ``cloud_skip_threshold`` is the per-token cloud
    fraction above which a token is dropped. Both are set on the masking
    ``strategy_config`` for the train module AND the dataloader copies. Override at
    launch with e.g.
    ``--train_module.masking_config.strategy_config.cloud_skip_threshold=0.3``
    (and the matching ``--data_loader.masking_config...``).

Precompute the sidecar first (against LANDSAT_REFL_H5_DIR):
    NUM_SHARDS=8 SHARD_BASE=0 bash scripts/tools/precompute_clouds_8gpu.sh
after setting ``--h5_dir`` to LANDSAT_REFL_H5_DIR (or pass it through the env/CLI).
"""

import logging

from base import build_common_components, build_visualize_config
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

from olmoearth_pretrain.data.cloud_mask_cache import default_cache_dir
from olmoearth_pretrain.data.dataloader import OlmoEarthDataLoaderConfig
from olmoearth_pretrain.data.dataset import OlmoEarthDatasetConfig
from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig
from olmoearth_pretrain.train.train_module.latent_mim import LatentMIMTrainModuleConfig

logger = logging.getLogger(__name__)

REGISTER_DIM = 768
MODULE_PATH = (
    "scripts/vnext/2026_07_24_new_maps/"
    "regbtl_v1_2_gdyn_d768_wideread_regsup_w0p1_newsampling_psuniform_landsat_refl_cloud.py"
)

# New-maps h5 whose Landsat modality is TOA reflectance / brightness temperature.
LANDSAT_REFL_H5_DIR = (
    "/weka/dfive-default/helios/dataset/osm_sampling_landsat_refl/"
    "h5py_data_w_missing_timesteps_zstd_3_128_x_4/"
    "cdl_glo30_landsat_meta_canopy_height_openstreetmap_raster_"
    "sentinel1_sentinel2_l2a_worldcereal_worldcover/1138828"
)
# Reflectance-scale Landsat norm stats (matches the landsat_refl twin).
LANDSAT_REFL_NORM_CONFIG = "computed_landsat_reflectance.json"

# Cloud sidecar for the reflectance build (computed against LANDSAT_REFL_H5_DIR).
CLOUD_CACHE_DIR = default_cache_dir(LANDSAT_REFL_H5_DIR)
# Per-token cloud fraction above which a DECODER token is dropped.
CLOUD_SKIP_THRESHOLD = 0.5
# Drop cloudy tokens from the "output" (DECODER/target) role only.
CLOUD_APPLY_TO = "output"


def _add_cloud_skip(strategy_config: dict) -> None:
    """Add the cloud-skip knobs to a random_time_with_decode strategy_config in place."""
    strategy_config["cloud_skip_threshold"] = CLOUD_SKIP_THRESHOLD
    strategy_config["cloud_apply_to"] = CLOUD_APPLY_TO


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """d768 wideread + new-maps register-grid supervision at w0p1 (base_weight 0.1)."""
    config = build_wideread_regbtl_model_config(
        common, latent_self_attn=True, register_dim=REGISTER_DIM
    )
    return add_register_supervision(config, base_weight=SUPERVISION_BASE_WEIGHT)


def build_dataloader_config(common: CommonComponents) -> OlmoEarthDataLoaderConfig:
    """Single-view newsampling dataloader (psuniform) with the cloud-skip knobs added."""
    config = apply_uniform_patch_sizes(
        apply_new_sampling(build_1fwd_dataloader_config(common))
    )
    _add_cloud_skip(config.masking_config.strategy_config)
    return config


def build_train_module_config(common: CommonComponents) -> LatentMIMTrainModuleConfig:
    """1fwd + fused AdamW train module with the cloud-skip knobs on its masking copy."""
    config = apply_microbatch(build_faster_train_module_config(common))
    _add_cloud_skip(config.masking_config.strategy_config)
    return config


def build_dataset_config(common: CommonComponents) -> OlmoEarthDatasetConfig:
    """Reflectance dataset + matching norm, pointed at the reflectance cloud sidecar."""
    return OlmoEarthDatasetConfig(
        h5py_dir=LANDSAT_REFL_H5_DIR,
        training_modalities=common.training_modalities,
        computed_norm_config=LANDSAT_REFL_NORM_CONFIG,
        cloud_cache_dir=CLOUD_CACHE_DIR,
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
