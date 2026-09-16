"""d768 wideread + regsup w1 (newmaps) + ERA5 monthly tokens as an ENCODER INPUT.

Exact twin of ``regbtl_v1_2_gdyn_d768_wideread_regsup_w1_newsampling_psuniform_landsat_refl``
-- same d768 full-width Perceiver register bottleneck, same 1fwd + fused AdamW train
module, same decorrelated newsampling at uniform patch sizes, same reflectance-scale
Landsat norm stats, same six decode-only map supervision at ``base_weight=1.0`` -- with
ONE change:

* ``era5_10`` is added to ``training_modalities`` as a regular (NON decode-only)
  input modality, so the encoder consumes it and the Perceiver read can fold the
  monthly weather trajectory into the registers as CONDITIONING. It is not
  supervised and not forced decode-only; it is a cheap side input (a single 1x1
  time-only pixel, 12 months x 6 vars) that the register bottleneck summarises.

Rationale (see the 2026-09-15 ERA5 discussion): ERA5_10 is a scene-level temporal
context signal -- not spatial imagery -- so it serves as climate/phenology
conditioning for the crop/vegetation-adjacent supervision (cdl, worldcereal,
meta_canopy_height) and helps disambiguate spectral state (snow, drought, moisture).
It is conditioning, not leakage: the monthly weather trajectory is not trivially
recoverable from a masked S2/S1/Landsat patch. The maps are still predicted from the
spatial patches -- ERA5 rides along.

This arm reads the ERA5-inclusive h5 (built by adding ``era5_10`` to the
``run_h5_conversion`` supported-modality list); everything else is encoder-identical
to the d768 w1 arm, so its evals read directly against that anchor.
"""

import logging

from base import build_common_components as _base_build_common_components
from base import build_trainer_config as _base_build_trainer_config
from base import build_visualize_config
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

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.data.dataset import OlmoEarthDatasetConfig
from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig
from olmoearth_pretrain.train.train_module.latent_mim import LatentMIMTrainModuleConfig

logger = logging.getLogger(__name__)

REGISTER_DIM = 768
SUPERVISION_BASE_WEIGHT = 1.0
MODULE_PATH = (
    "scripts/vnext/2026_07_24_new_maps/"
    "regbtl_v1_2_gdyn_d768_wideread_regsup_w1_newsampling_psuniform_landsat_refl_era5in.py"
)

# ERA5-inclusive new-maps h5: the reflectance h5 rebuilt with ``era5_10`` added to the
# supported-modality list. ``era5_10`` sorts right after ``cdl`` in the auto-derived
# folder name. NOTE: confirm the trailing sample-count segment against the final
# ``h5py_dir`` printed by run_h5_conversion (expected unchanged at 1138828 since era5
# is a non-required time-only modality that does not change which windows pass filters).
ERA5_H5_DIR = (
    "/weka/dfive-default/helios/dataset/osm_sampling_landsat_refl/"
    "h5py_data_w_missing_timesteps_zstd_3_128_x_4/"
    "cdl_era5_10_glo30_landsat_meta_canopy_height_openstreetmap_raster_"
    "sentinel1_sentinel2_l2a_worldcereal_worldcover/1138828"
)
# Reflectance-scale Landsat norm stats; already carries an ``era5_10`` entry so the
# ERA5 input is per-band normalised (critical -- surface-pressure ~9.3e4 vs precip
# ~3e-3 would otherwise be un-comparable).
LANDSAT_REFL_NORM_CONFIG = "computed_landsat_reflectance.json"


def build_common_components(
    script: str, cmd, run_name: str, cluster: str, overrides: list[str]
) -> CommonComponents:
    """New-maps common components with ``era5_10`` appended as an input modality.

    Appending to ``training_modalities`` (and NOT to the masking's
    ``only_decode_modalities``) is what makes ERA5 an encoded input rather than a
    decode-only target: the encoder builds embeddings for it and the bottleneck
    reads it. The base tokenizer handles a non-spatial multitemporal modality
    natively (no per-modality tokenization override needed).
    """
    common = _base_build_common_components(script, cmd, run_name, cluster, overrides)
    if Modality.ERA5_10.name not in common.training_modalities:
        common.training_modalities = [
            *common.training_modalities,
            Modality.ERA5_10.name,
        ]
    return common


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """d768 wideread + new-maps register-grid supervision at w1 (unchanged from anchor)."""
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
    """New-maps dataset on the ERA5-inclusive reflectance h5 + matching norm."""
    return OlmoEarthDatasetConfig(
        h5py_dir=ERA5_H5_DIR,
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
