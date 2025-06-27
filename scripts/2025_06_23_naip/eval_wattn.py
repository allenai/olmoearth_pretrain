"""Evaluation for models trained on all data."""

from eval import build_trainer_config
from train import (
    build_dataloader_config,
    build_dataset_config,
    build_train_module_config,
    build_visualize_config,
)
from train_wattn import (
    build_model_config,
    my_build_common_components,
)

from helios.internal.experiment import (
    main,
)

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
