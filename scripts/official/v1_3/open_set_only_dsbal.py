r"""Launch script: ``open_set_only`` with per-dataset temperature-balanced loss.

Identical to ``open_set_only.py`` except the supervised loss weighting: each labeled
sample is weighted by its source dataset's size ``n_d ** (tau - 1)`` (tau = 0.5,
normalized so the mean weight under natural sampling is one), which emulates drawing
datasets with probability ``p_d ~ n_d ** tau`` instead of proportionally to size. The
label bank spans ~100 to ~50k samples per dataset, so without this the largest
datasets dominate the probe. Dataset sizes come from the label-bank
``registry.json`` (``num_samples``).

Usage (from the repo root)::

    python scripts/official/v1_3/open_set_only_dsbal.py launch open_set_only_dsbal \\
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

MODULE_PATH = "scripts/official/v1_3/open_set_only_dsbal.py"

BALANCE_TEMPERATURE = 0.5


def build_model_config(common: CommonComponents) -> OpenSetLatentMIMConfig:
    """v1.3 + open-set probe with dataset-temperature-balanced supervised loss."""
    return _build_model_config(
        common,
        dataset_balance="dataset_temperature",
        balance_temperature=BALANCE_TEMPERATURE,
    )


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
