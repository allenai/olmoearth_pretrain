"""Trying to prototype fitting everything into olmo core."""

from shared import (
    build_dataloader_config,
    build_dataset_config,
    build_model_config,
    build_train_module_config,
    build_trainer_config,
    build_visualize_config,
    my_build_common_components,
)

from helios.data.dataloader import HeliosDataLoaderConfig
from helios.internal.experiment import CommonComponents, main
from helios.nn.latent_mim import LatentMIMConfig


def my_build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """Build the model config for an experiment."""
    model_config = build_model_config(common)
    model_config.encoder_config.min_patch_size = 1
    model_config.encoder_config.max_patch_size = 8
    return model_config


def my_build_dataloader_config(common: CommonComponents) -> HeliosDataLoaderConfig:
    """Build the dataloader config for an experiment."""
    dataloader_config = build_dataloader_config(common)
    dataloader_config.min_patch_size = 1
    dataloader_config.max_patch_size = 8
    return dataloader_config


if __name__ == "__main__":
    main(
        common_components_builder=my_build_common_components,
        model_config_builder=my_build_model_config,
        train_module_config_builder=build_train_module_config,
        dataset_config_builder=build_dataset_config,
        dataloader_config_builder=my_build_dataloader_config,
        trainer_config_builder=build_trainer_config,
        visualize_config_builder=build_visualize_config,
    )
