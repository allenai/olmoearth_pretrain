"""Script for Debugging Galileo.

These Settings are meant to help you get quick results on a single GPU in minimal time
"""

from galileo_shared import (
    build_common_components_limited_modalities,
    build_dataloader_config,
    build_dataset_config,
    build_model_config,
    build_train_module_config,
    build_trainer_config,
    build_visualize_config,
)

from helios.data.dataloader import HeliosDataLoaderConfig
from helios.internal.experiment import CommonComponents, main
from helios.train.train_module.galileo import GalileoTrainModuleConfig


def my_build_train_module_config(
    common: CommonComponents,
) -> GalileoTrainModuleConfig:
    """Build the train module config for an experiment."""
    train_module_config = build_train_module_config(common)
    train_module_config.rank_microbatch_size = 8
    return train_module_config


def my_build_dataloader_config(common: CommonComponents) -> HeliosDataLoaderConfig:
    """Build the dataloader config for an experiment."""
    dataloader_config = build_dataloader_config(common)
    dataloader_config.token_budget = 6000
    return dataloader_config


if __name__ == "__main__":
    main(
        common_components_builder=build_common_components_limited_modalities,
        model_config_builder=build_model_config,
        train_module_config_builder=my_build_train_module_config,
        dataset_config_builder=build_dataset_config,
        dataloader_config_builder=my_build_dataloader_config,
        trainer_config_builder=build_trainer_config,
        visualize_config_builder=build_visualize_config,
    )
