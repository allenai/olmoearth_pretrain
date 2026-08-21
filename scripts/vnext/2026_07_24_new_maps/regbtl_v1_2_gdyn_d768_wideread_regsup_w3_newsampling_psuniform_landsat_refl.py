"""d768 wideread + regsup at base_weight 3.0 (newmaps) on LANDSAT REFLECTANCE h5.

Exact twin of ``regbtl_v1_2_gdyn_d768_wideread_regsup_w1_newsampling_psuniform_landsat_refl``
with ONE change: register-grid supervision runs at ``base_weight=3.0`` instead of
1.0. ``base_weight`` scales the per-task-type weights, so all six decode-only map
modalities (worldcover, glo30, openstreetmap_raster, meta_canopy_height, cdl,
worldcereal) move together and the latent-MIM objective is untouched.

WHY GO PAST w1: the weight sweep was settled on the OLD sampler, where w0p1 beat w1
on 5 of 7 probes and the losses were concentrated in classification (eurosat 0.9160
vs 0.8750, so2sat 0.6602 vs 0.6237, geo_ecosystem 0.2279 vs 0.1958). Re-reading the
same pair under the CURRENT decorrelated sampler at psuniform (660k, d768 regsup)
inverts that verdict -- w1 now wins 7 of 10, and it wins the large cells:
geo_ecosystem 0.2621 vs 0.2089 (+5.3pp, a sign flip from old sampling), the frozen
ps=1 PASTIS exports +1.8pp (s2) and +1.2pp (s1s2), pastis +1.1pp. so2sat also flips
to a small w1 edge. Only eurosat keeps a real penalty (-2.2pp, narrowed from ~4.1pp),
with mados (-1.5pp) and yemen_crop (-1.4pp) the other two losses. So the classification
cost that justified w0p1 was mostly an artifact of the old sampler, and 0.01 -> 0.1 ->
1.0 is now a mostly-upward trend rather than a tradeoff -- which leaves the top of the
range untested.

WHY 3 AND NOT 10: the supervision loss is added to the latent-MIM loss unweighted
(``loss = loss + sup_loss``) and REGRESSION sits at task-type weight 1.0, so at w1 the
glo30/canopy-height terms already match the self-supervised objective in scale. Above
w1 map prediction becomes the dominant term; at w10 latent-MIM is a rounding error.
w3 is the largest step that still leaves the SSL objective contributing, so it is the
informative third point for locating the turn rather than overshooting past it.

Judge this arm on the segmentation and dense probes (pastis, the ps=1 exports,
fifty_cities) and on eurosat as the known casualty; the AEF year-aligned suite is
all-classification under balanced accuracy and is the axis higher weight is most
likely to cost.
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

REGISTER_DIM = 768
# The w3 arm: 3x the w1 twin, 30x w0p1. Set here rather than imported from
# perceiver_common's SUPERVISION_BASE_WEIGHT (0.1) because it is the knob this
# script exists to change.
SUPERVISION_BASE_WEIGHT = 3.0
MODULE_PATH = (
    "scripts/vnext/2026_07_24_new_maps/"
    "regbtl_v1_2_gdyn_d768_wideread_regsup_w3_newsampling_psuniform_landsat_refl.py"
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
    """d768 wideread + new-maps register-grid supervision at w3 (base_weight 3.0)."""
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
