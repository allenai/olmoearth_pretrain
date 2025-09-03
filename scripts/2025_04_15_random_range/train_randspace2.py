"""Launch script."""

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
from helios.train.masking import MaskingConfig
from helios.train.train_module.latent_mim import LatentMIMTrainModuleConfig


def my_build_train_module_config(
    common: CommonComponents,
) -> LatentMIMTrainModuleConfig:
    """Get updated train module with randspace masking."""
    train_module_config = build_train_module_config(common)
    train_module_config.masking_config = MaskingConfig(
        strategy_config={
            "type": "random_space",
            "encode_ratio": 0.1,
            "decode_ratio": 0.85,
        }
    )
    return train_module_config


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
