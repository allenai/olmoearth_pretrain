"""Script for v0 sweep for Contrastive Latent MIM."""

from shared import (
    build_dataloader_config,
    build_dataset_config,
    build_model_config_builder,
    build_train_module_config_builder,
    build_trainer_config,
    build_visualize_config,
)

from helios.internal.common import build_common_components
from helios.internal.experiment import main

if __name__ == "__main__":
    main(
        common_components_builder=build_common_components,
        model_config_builder=build_model_config_builder(
            separate_attention=False, model="contrastive_latentmim"
        ),
        train_module_config_builder=build_train_module_config_builder(
            model="contrastive_latentmim"
        ),
        dataset_config_builder=build_dataset_config,
        dataloader_config_builder=build_dataloader_config,
        trainer_config_builder=build_trainer_config,
        visualize_config_builder=build_visualize_config,
    )
