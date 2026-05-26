"""hidden1 ablation: disable band dropout."""

import logging

from hidden1 import (
    build_common_components,
    build_dataloader_config,
    build_dataset_config,
    build_trainer_config,
    build_visualize_config,
)
from hidden1 import (
    build_model_config as build_model_config_base,
)
from hidden1 import (
    build_train_module_config as build_train_module_config_base,
)

from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig
from olmoearth_pretrain.train.loss import LossConfig
from olmoearth_pretrain.train.train_module.contrastive_latentmim import (
    ContrastiveLatentMIMTrainModuleConfig,
)

logger = logging.getLogger(__name__)


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """Build the model config for an experiment."""
    config = build_model_config_base(common)
    config.encoder_config.random_band_dropout = False
    config.encoder_config.band_dropout_rate = 0.0
    config.encoder_config.band_dropout_modalities = []
    return config


def build_train_module_config(
    common: CommonComponents,
) -> ContrastiveLatentMIMTrainModuleConfig:
    """Build the train module config for an experiment."""
    config = build_train_module_config_base(common)
    config.loss_config = LossConfig(
        loss_config={
            "type": "modality_patch_discrimination_vec",
            "tau": 0.1,
        }
    )
    return config


def run() -> None:
    """Run the experiment."""
    main(
        common_components_builder=build_common_components,
        model_config_builder=build_model_config,
        train_module_config_builder=build_train_module_config,
        dataset_config_builder=build_dataset_config,
        dataloader_config_builder=build_dataloader_config,
        trainer_config_builder=build_trainer_config,
        visualize_config_builder=build_visualize_config,
    )


if __name__ == "__main__":
    run()
