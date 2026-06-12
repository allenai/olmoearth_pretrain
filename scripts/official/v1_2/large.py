"""v1.2 large: ViT-large encoder + mixed 3D RoPE, decoder depth 4.

Uses the v1.1 base train-module config unchanged (lr 1e-4, wd 0.02).
"""

import logging

from base import (
    PATCH_EMBED_HIDDEN_SIZES,
    build_common_components,
    build_dataloader_config,
    build_dataset_config,
    build_size_model_config,
    build_train_module_config,
    build_trainer_config,
    build_visualize_config,
)

from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig

logger = logging.getLogger(__name__)


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """Build the model config for an experiment."""
    return build_size_model_config(
        common, "large_shallow_decoder", PATCH_EMBED_HIDDEN_SIZES
    )


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
