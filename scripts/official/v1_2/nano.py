"""v1.2 nano: ViT-nano encoder + mixed 3D RoPE, decoder depth 4."""

import logging

from base import (
    build_common_components,
    build_dataloader_config,
    build_dataset_config,
    build_size_model_config,
    build_trainer_config,
    build_visualize_config,
)
from base import (
    build_train_module_config as build_train_module_config_base,
)
from olmo_core.optim import AdamWConfig

from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig
from olmoearth_pretrain.train.train_module.contrastive_latentmim import (
    ContrastiveLatentMIMTrainModuleConfig,
)

logger = logging.getLogger(__name__)

LEARNING_RATE = 0.0002
WEIGHT_DECAY = 0.02
# Nano's hidden size is small, so use a small patch-embed hidden width.
PATCH_EMBED_HIDDEN_SIZES = [12]


def build_train_module_config(
    common: CommonComponents,
) -> ContrastiveLatentMIMTrainModuleConfig:
    """Build the train module config for an experiment."""
    config = build_train_module_config_base(common)
    config.optim_config = AdamWConfig(
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        fused=False,
    )
    return config


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """Build the model config for an experiment."""
    return build_size_model_config(common, "nano", PATCH_EMBED_HIDDEN_SIZES)


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
