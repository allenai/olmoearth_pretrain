"""Evaluation for models trained on all data."""

from eval import build_trainer_config
from train import (
    build_dataloader_config,
    build_model_config,
    build_train_module_config,
    build_visualize_config,
    my_build_common_components,
)

from helios.data.concat import HeliosConcatDatasetConfig
from helios.data.dataset import HeliosDatasetConfig
from helios.internal.experiment import (
    CommonComponents,
    main,
)


def build_dataset_config(common: CommonComponents) -> HeliosDatasetConfig:
    """Build the dataset config for an experiment."""
    dataset_configs = [
        # presto
        HeliosDatasetConfig(
            h5py_dir="/weka/dfive-default/helios/dataset/presto/h5py_data_w_missing_timesteps_128_x_4_zstd_3/landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcover/469892",
            training_modalities=common.training_modalities,
        ),
        # osm_sampling
        HeliosDatasetConfig(
            h5py_dir="/weka/dfive-default/helios/dataset/osm_sampling/h5py_data_w_missing_timesteps_128_x_4_zstd_3/landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcover/1141152",
            training_modalities=common.training_modalities,
        ),
        # osmbig
        HeliosDatasetConfig(
            h5py_dir="/weka/dfive-default/helios/dataset/osmbig/h5py_data_w_missing_timesteps_zstd_3_128_x_4/landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcover/1297928",
            training_modalities=common.training_modalities,
        ),
        # presto neighbor
        HeliosDatasetConfig(
            h5py_dir="/weka/dfive-default/helios/dataset/presto_neighbor/h5py_data_w_missing_timesteps_zstd_3_128_x_4/landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcover/3507748",
            training_modalities=common.training_modalities,
        ),
        # worldcover
        HeliosDatasetConfig(
            h5py_dir="/weka/dfive-default/helios/dataset/worldcover_sampling/h5py_data_w_missing_timesteps_zstd_3_128_x_4/landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcover/6370580",
            training_modalities=common.training_modalities,
        ),
    ]
    return HeliosConcatDatasetConfig(dataset_configs=dataset_configs)


if __name__ == "__main__":
    main(
        common_components_builder=my_build_common_components,
        model_config_builder=build_model_config,
        train_module_config_builder=build_train_module_config,
        dataset_config_builder=build_dataset_config,
        dataloader_config_builder=build_dataloader_config,
        trainer_config_builder=build_trainer_config,
        visualize_config_builder=build_visualize_config,
    )
