"""Trying to prototype fitting everything into olmo core."""

from shared import (
    build_dataloader_config,
    build_dataset_config,
    build_model_config,
    build_train_module_config,
    build_trainer_config,
    build_visualize_config,
)

from helios.internal.common import build_common_components
from helios.internal.experiment import CommonComponents, main
from helios.nn.latent_mim import LatentMIMConfig


def my_build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """Build the model config for an experiment."""
    model_config = build_model_config(common)
    model_config.encoder_config.embedding_size = 1024
    model_config.encoder_config.depth = 24
    model_config.encoder_config.num_heads = 16
    model_config.decoder_config.encoder_embedding_size = 1024
    return model_config


if __name__ == "__main__":
    main(
        common_components_builder=build_common_components,
        model_config_builder=my_build_model_config,
        train_module_config_builder=build_train_module_config,
        dataset_config_builder=build_dataset_config,
        dataloader_config_builder=build_dataloader_config,
        trainer_config_builder=build_trainer_config,
        visualize_config_builder=build_visualize_config,
    )
