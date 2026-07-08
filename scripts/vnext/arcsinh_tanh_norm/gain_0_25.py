"""arcsinh_tanh normalization with tanh_gain=0.25.

Identical to ``base.py`` except the dataset normalization uses ``tanh_gain=0.25``
instead of 1.0. This is the gentlest squash of the variants: the bulk of the
distribution uses almost the full (-1, 1) range and only far outliers saturate
(e.g. +1 std -> tanh(0.25) = 0.24, +2 std -> tanh(0.5) = 0.46, +3 std ->
tanh(0.75) = 0.64).
"""

import logging

from base import (
    build_common_components,
    build_dataloader_config,
    build_model_config,
    build_train_module_config,
    build_trainer_config,
    build_visualize_config,
)

from olmoearth_pretrain.data.dataset import OlmoEarthDatasetConfig
from olmoearth_pretrain.internal.experiment import CommonComponents, main

logger = logging.getLogger(__name__)

TANH_GAIN = 0.25


def build_dataset_config(common: CommonComponents) -> OlmoEarthDatasetConfig:
    """Build the dataset config using arcsinh_tanh normalization with a lower gain."""
    return OlmoEarthDatasetConfig(
        h5py_dir="/weka/dfive-default/helios/dataset/osm_sampling/h5py_data_w_missing_timesteps_zstd_3_128_x_4/cdl_gse_landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcereal_worldcover_worldpop_wri_canopy_height_map/1138828",
        training_modalities=common.training_modalities,
        norm_strategy="arcsinh_tanh",
        tanh_gain=TANH_GAIN,
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
