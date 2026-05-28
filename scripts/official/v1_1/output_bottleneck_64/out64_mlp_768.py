"""64-dim bottleneck, packer = 1-hidden-layer MLP (768).

Smallest non-linear packer: Linear(768, 768) -> LN -> GELU -> Linear(768, 64).
"""

from base_out64 import (
    build_common_components,
    build_dataloader_config,
    build_dataset_config,
    build_model_config_with,
    build_train_module_config,
    build_trainer_config,
    build_visualize_config,
)

from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig

OUTPUT_PROJ_HIDDEN_SIZES = [768]


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """64-dim bottleneck with a 1-hidden-layer non-linear packer."""
    return build_model_config_with(
        common, output_proj_hidden_sizes=OUTPUT_PROJ_HIDDEN_SIZES
    )


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
