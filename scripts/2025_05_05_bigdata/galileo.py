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

from helios.internal.experiment import main

if __name__ == "__main__":
    main(
        common_components_builder=build_common_components_limited_modalities,
        model_config_builder=build_model_config,
        train_module_config_builder=build_train_module_config,
        dataset_config_builder=build_dataset_config,
        dataloader_config_builder=build_dataloader_config,
        trainer_config_builder=build_trainer_config,
        visualize_config_builder=build_visualize_config,
    )
