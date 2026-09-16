"""proj128lin_sup768 (d768 teacher + detached 128/64 student) + ERA5 as ENCODER INPUT.

Twin of ``regbtl_v1_2_gdyn_d768_proj128lin_sup768_w1_newsampling_psuniform_landsat_refl``
-- same d768 teacher bottleneck, same detached per-cell ``Linear(768, 128)`` student
with a self-sufficient 64-dim Matryoshka prefix, same w1 register supervision on the
d768 registers, same distillation (cosine-through-back-projection + Gram), and the same
projected + frozen ps=1 PASTIS evals scored at 768 / 128 / 64 -- with ONE change:

* ``era5_10`` is added to ``training_modalities`` as a regular (NON decode-only) input,
  so the encoder folds the 12-month weather trajectory into the d768 registers as
  CONDITIONING. It is not supervised and not forced decode-only; it is a cheap 1x1
  time-only side input the Perceiver read summarises. The student and every eval are
  otherwise identical to the anchor, so the proj/ps=1 readouts compare directly.

Rationale mirrors the wideread ``_era5in`` arm: ERA5 is scene-level temporal context
(climate/phenology), not spatial imagery, so it conditions the crop/vegetation-adjacent
supervision without leaking the maps (the monthly weather trajectory is not recoverable
from a masked S2/S1/Landsat patch). Reads the ERA5-inclusive reflectance h5.
"""

import logging

from base import build_common_components as _base_build_common_components
from base import build_trainer_config as _base_build_trainer_config
from base import build_visualize_config
from perceiver_common import route_loop_evals_to_beaker
from regbtl_v1_2_gdyn_d768_proj128lin_sup768_w1_newsampling_psuniform_landsat_refl import (
    LANDSAT_REFL_NORM_CONFIG,
    add_pastis_ps1_eval_tasks,
    add_projected_eval_tasks,
    build_dataloader_config,
    build_model_config,
    build_train_module_config,
)

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.data.dataset import OlmoEarthDatasetConfig
from olmoearth_pretrain.internal.experiment import CommonComponents, main

logger = logging.getLogger(__name__)

# Must point at THIS script so the Beaker eval jobs rebuild the ERA5-including model
# (era5_10 in training_modalities changes the encoder's per-modality embeddings).
MODULE_PATH = (
    "scripts/vnext/2026_07_24_new_maps/"
    "regbtl_v1_2_gdyn_d768_proj128lin_sup768_w1_newsampling_psuniform_landsat_refl_era5in.py"
)

# ERA5-inclusive new-maps reflectance h5 (era5_10 added to run_h5_conversion's modality
# list). era5_10 sorts right after cdl in the auto-derived folder name; the trailing
# sample-count segment is unchanged at 1138828 (confirmed against the final h5py_dir).
ERA5_H5_DIR = (
    "/weka/dfive-default/helios/dataset/osm_sampling_landsat_refl/"
    "h5py_data_w_missing_timesteps_zstd_3_128_x_4/"
    "cdl_era5_10_glo30_landsat_meta_canopy_height_openstreetmap_raster_"
    "sentinel1_sentinel2_l2a_worldcereal_worldcover/1138828"
)


def build_common_components(
    script: str, cmd, run_name: str, cluster: str, overrides: list[str]
) -> CommonComponents:
    """proj128lin common components with ``era5_10`` appended as an input modality.

    Appending to ``training_modalities`` (and NOT to the masking's
    ``only_decode_modalities``) is what makes ERA5 an encoded input rather than a
    decode-only target: the encoder builds embeddings for it and the bottleneck reads
    it. The base tokenizer handles a non-spatial multitemporal modality natively.
    """
    common = _base_build_common_components(script, cmd, run_name, cluster, overrides)
    if Modality.ERA5_10.name not in common.training_modalities:
        common.training_modalities = [
            *common.training_modalities,
            Modality.ERA5_10.name,
        ]
    return common


def build_dataset_config(common: CommonComponents) -> OlmoEarthDatasetConfig:
    """proj128lin dataset on the ERA5-inclusive reflectance h5 + matching norm."""
    return OlmoEarthDatasetConfig(
        h5py_dir=ERA5_H5_DIR,
        training_modalities=common.training_modalities,
        computed_norm_config=LANDSAT_REFL_NORM_CONFIG,
    )


def build_trainer_config(common: CommonComponents):
    """proj128lin trainer: student-head evals + ps=1 PASTIS, routed through Beaker.

    Re-derived here (rather than imported from the anchor) so ``MODULE_PATH`` points at
    this ERA5 script; otherwise the eval jobs would rebuild the non-ERA5 anchor model.
    """
    return route_loop_evals_to_beaker(
        add_pastis_ps1_eval_tasks(
            add_projected_eval_tasks(_base_build_trainer_config(common))
        ),
        MODULE_PATH,
    )


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
