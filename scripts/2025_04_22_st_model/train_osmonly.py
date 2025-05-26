"""Trying to prototype fitting everything into olmo core."""

from olmo_core.config import DType
from shared import (
    build_dataloader_config,
    build_model_config,
    build_train_module_config,
    build_trainer_config,
    build_visualize_config,
)

from helios.data.concat import HeliosConcatDatasetConfig
from helios.data.dataset import HeliosDatasetConfig
from helios.internal.common import build_common_components
from helios.internal.experiment import CommonComponents, main


def build_dataset_config(common: CommonComponents) -> HeliosDatasetConfig:
    """Build the dataset config for an experiment."""
    # NOTE: Change this directory based on the supported modalities
    dataset_configs = [
        HeliosDatasetConfig(
            h5py_dir="/weka/dfive-default/helios/dataset/osm_sampling/h5py_data_rerun/sentinel1_sentinel2_l2a_worldcover/283204",
            training_modalities=common.training_modalities,
            use_samples_with_missing_supported_modalities=False,
            dtype=DType.float32,
            cache_dir="/helios_cache/osm_sampling",
        ),
    ]
    return HeliosConcatDatasetConfig(dataset_configs=dataset_configs)


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
