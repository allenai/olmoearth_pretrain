"""Single bandset pretraining (no S1 drop, random time) with 3000 training windows."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "vnext/single_bandset_band_dropout"))

from base_band_dropout_no_s1_drop_random_time import (
    build_common_components,
    build_dataloader_config,
    build_model_config,
    build_train_module_config,
    build_trainer_config,
    build_visualize_config,
)

from olmoearth_pretrain.data.dataset import OlmoEarthDataset, OlmoEarthDatasetConfig
from olmoearth_pretrain.internal.experiment import CommonComponents, main

logger = logging.getLogger(__name__)


class _LoggingDataset(OlmoEarthDataset):
    def prepare(self) -> None:
        super().prepare()
        logger.info("Loaded %d training windows.", len(self))


class _LoggingDatasetConfig(OlmoEarthDatasetConfig):
    def build(self) -> _LoggingDataset:
        dataset = super().build()
        dataset.__class__ = _LoggingDataset
        return dataset


def build_dataset_config(common: CommonComponents) -> _LoggingDatasetConfig:
    return _LoggingDatasetConfig(
        h5py_dir="/weka/dfive-default/helios/dataset/osm_sampling/h5py_data_w_missing_timesteps_zstd_3_128_x_4/cdl_gse_landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcereal_worldcover_worldpop_wri_canopy_height_map/1138828",
        training_modalities=common.training_modalities,
        dataset_percentage=0.00264,
    )


if __name__ == "__main__":
    main(
        common_components_builder=build_common_components,
        model_config_builder=build_model_config,
        train_module_config_builder=build_train_module_config,
        dataset_config_builder=build_dataset_config,
        dataloader_config_builder=build_dataloader_config,
        trainer_config_builder=build_trainer_config,
        visualize_config_builder=build_visualize_config,
    )
