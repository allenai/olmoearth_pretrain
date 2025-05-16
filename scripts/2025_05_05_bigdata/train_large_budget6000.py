"""Trying to prototype fitting everything into olmo core."""

from shared import (
    build_dataloader_config,
    build_dataset_config,
    build_model_config,
    build_train_module_config,
    build_trainer_config,
    build_visualize_config,
)

from helios.data.dataloader import HeliosDataLoaderConfig
from helios.internal.common import build_common_components
from helios.internal.experiment import CommonComponents, main
from helios.nn.latent_mim import LatentMIMConfig
from helios.train.train_module.latent_mim import LatentMIMTrainModuleConfig


def my_build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """Build the model config for an experiment."""
    model_config = build_model_config(common)
    model_config.encoder_config.embedding_size = 1024
    model_config.encoder_config.depth = 24
    model_config.encoder_config.num_heads = 16
    model_config.decoder_config.encoder_embedding_size = 1024
    return model_config


def my_build_dataloader_config(common: CommonComponents) -> HeliosDataLoaderConfig:
    """Build the dataloader config for an experiment."""
    dataloader_config = build_dataloader_config(common)
    dataloader_config.token_budget = 6000
    # Got NaN with the default seed, try a different one.
    dataloader_config.seed = 12345
    return dataloader_config


def my_build_train_module_config(
    common: CommonComponents,
) -> LatentMIMTrainModuleConfig:
    """Build the train module config for an experiment."""
    train_module_config = build_train_module_config(common)
    train_module_config.rank_microbatch_size = 8
    return train_module_config


if __name__ == "__main__":
    main(
        common_components_builder=build_common_components,
        model_config_builder=my_build_model_config,
        train_module_config_builder=my_build_train_module_config,
        dataset_config_builder=build_dataset_config,
        dataloader_config_builder=my_build_dataloader_config,
        trainer_config_builder=build_trainer_config,
        visualize_config_builder=build_visualize_config,
    )
