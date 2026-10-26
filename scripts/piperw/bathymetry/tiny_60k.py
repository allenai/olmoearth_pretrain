"""Bathymetry-target ablation, baseline arm: v1.2 tiny for 60k steps."""

import logging
import sys
from collections.abc import Callable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "official" / "v1_2"))

import base  # noqa: E402
from olmo_core.train.common import Duration  # noqa: E402
from olmo_core.train.config import TrainerConfig  # noqa: E402
from tiny import build_model_config, build_train_module_config  # noqa: E402

from olmoearth_pretrain.internal.experiment import CommonComponents, main  # noqa: E402

logger = logging.getLogger(__name__)

MAX_STEPS = 60_000
WANDB_PROJECT = "2026_10_26_bathymetry_target"


def build_trainer_config(common: CommonComponents) -> TrainerConfig:
    """v1.2 trainer, shortened to MAX_STEPS and logged to the ablation project."""
    config = base.build_trainer_config(common)
    config.max_duration = Duration.steps(MAX_STEPS)
    config.callbacks["wandb"].project = WANDB_PROJECT
    return config


def run_with(**builders: Callable) -> None:
    """Run the experiment, replacing any of the default builders."""
    defaults: dict[str, Callable] = dict(
        common_components_builder=base.build_common_components,
        model_config_builder=build_model_config,
        train_module_config_builder=build_train_module_config,
        dataset_config_builder=base.build_dataset_config,
        dataloader_config_builder=base.build_dataloader_config,
        trainer_config_builder=build_trainer_config,
        visualize_config_builder=base.build_visualize_config,
    )
    main(**{**defaults, **builders})


if __name__ == "__main__":
    run_with()
