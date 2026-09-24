r"""Launch script: ``open_set_only`` with a non-linear (2-layer MLP) open-set probe.

Identical to ``open_set_only.py`` except the probe: a shared ``Linear(768, 768) ->
GELU`` trunk (first layer shared across all classes and datasets) feeds the linear
classification and regression heads, instead of the heads reading the register grid
directly. Tests whether 3871 classes need a non-linear readout of the registers.

Usage (from the repo root)::

    python scripts/official/v1_3/open_set_only_mlp.py launch open_set_only_mlp \\
        ai2/jupiter --launch.num_gpus=8
"""

import logging

from base import build_visualize_config
from open_set_base import (
    build_common_components,
    build_dataloader_config,
    build_open_set_dataset_config,
    build_train_module_config,
)
from open_set_base import build_model_config as _build_model_config
from open_set_base import build_trainer_config as _build_open_set_trainer_config

from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.nn.open_set_latent_mim import OpenSetLatentMIMConfig

logger = logging.getLogger(__name__)

MODULE_PATH = "scripts/official/v1_3/open_set_only_mlp.py"


def build_model_config(common: CommonComponents) -> OpenSetLatentMIMConfig:
    """v1.3 + open-set probe with a shared MLP trunk."""
    return _build_model_config(common, head_type="mlp")


def build_dataset_config(common: CommonComponents):
    """Open-set supervised dataset only."""
    return build_open_set_dataset_config(common)


def build_trainer_config(common: CommonComponents):
    """Trainer with the in-loop evals pointed at this module."""
    return _build_open_set_trainer_config(common, MODULE_PATH)


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
