"""Dual-resolution variant of ``base.py`` restricted to patch sizes 1-4.

Identical to ``base.py`` except the sampled coarse patch size is capped at 4 instead of
8 (the minimum stays 1). This bounds the per-pixel branch's cost, which scales with the
number of pixels per coarse patch (``P**2``): capping ``P`` at 4 keeps the pixel branch
at or below the coarse branch's per-token cost. Both the dataloader (which samples the
patch size) and the encoder's flexi patch embedding (which must support the sampled
range) are updated.
"""

import logging

from base import (
    build_common_components,
    build_dataset_config,
    build_train_module_config,
    build_trainer_config,
    build_visualize_config,
)
from base import (
    build_dataloader_config as build_dataloader_config_base,
)
from base import (
    build_model_config as build_model_config_base,
)

from olmoearth_pretrain.data.dataloader import OlmoEarthDataLoaderConfig
from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.nn.dual_res_model import DualResLatentMIMConfig

logger = logging.getLogger(__name__)

MAX_PATCH_SIZE = 4


def build_dataloader_config(common: CommonComponents) -> OlmoEarthDataLoaderConfig:
    """Build the dataloader config, sampling patch sizes in [1, 4]."""
    config = build_dataloader_config_base(common)
    config.max_patch_size = MAX_PATCH_SIZE
    return config


def build_model_config(common: CommonComponents) -> DualResLatentMIMConfig:
    """Build the model config whose patch embedding supports patch sizes up to 4."""
    config = build_model_config_base(common)
    config.encoder_config.max_patch_size = MAX_PATCH_SIZE
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
