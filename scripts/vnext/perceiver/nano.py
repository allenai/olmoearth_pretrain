"""Set-Latent Perceiver nano preset (small SLP for quick iteration)."""

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
from olmoearth_pretrain.nn.set_latent_perceiver import SetLatentPerceiverConfig
from olmoearth_pretrain.train.train_module.set_latent_perceiver import (
    SetLatentPerceiverTrainModuleConfig,
)

logger = logging.getLogger(__name__)


def build_train_module_config(
    common: CommonComponents,
) -> SetLatentPerceiverTrainModuleConfig:
    """Build the nano train module config (larger LR, bigger microbatch)."""
    config = build_train_module_config_base(common)
    config.optim_config = AdamWConfig(lr=0.0002, weight_decay=0.02, fused=False)
    config.rank_microbatch_size = 64
    return config


def build_model_config(common: CommonComponents) -> SetLatentPerceiverConfig:
    """Build the nano SLP model config."""
    return build_size_model_config(common, "nano")


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
