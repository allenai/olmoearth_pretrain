"""64-dim bottleneck, packer = 2-hidden-layer MLP (1536 -> 768).

Expands to 1536 before compressing: Linear(768, 1536) -> LN -> GELU ->
Linear(1536, 768) -> LN -> GELU -> Linear(768, 64).
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

OUTPUT_PROJ_HIDDEN_SIZES = [1536, 768]


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """64-dim bottleneck with a 2-hidden-layer expand-then-compress packer."""
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
