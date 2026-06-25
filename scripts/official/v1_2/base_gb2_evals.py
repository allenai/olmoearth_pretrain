"""v1.2 base, but the in-loop eval set also includes the GeoBench-2 tasks.

Identical to base.py except that build_trainer_config additionally registers the
gb2 eval tasks (pulled from the canonical eval set) on the downstream-evaluator
callback. base.py is left unchanged. The gb2 tasks use a 20000-step cadence (a
multiple of the permanent save_interval) so run_as_beaker_job evals always find a
persisted, non-ephemeral checkpoint at the eval step.
"""

import dataclasses
import logging

from base import (
    build_common_components,
    build_dataloader_config,
    build_dataset_config,
    build_model_config,
    build_train_module_config,
    build_visualize_config,
)
from base import (
    build_trainer_config as build_trainer_config_base,
)
from olmo_core.train.common import Duration
from olmo_core.train.config import TrainerConfig

from olmoearth_pretrain.internal.all_evals import EVAL_TASKS as ALL_EVAL_TASKS
from olmoearth_pretrain.internal.experiment import CommonComponents, main

logger = logging.getLogger(__name__)

GB2_EVAL_INTERVAL = Duration.steps(20000)


def build_trainer_config(common: CommonComponents) -> TrainerConfig:
    """base.py trainer config + the GeoBench-2 tasks added to the eval set."""
    config = build_trainer_config_base(common)
    downstream_evaluator = config.callbacks["downstream_evaluator"]
    for name, task in ALL_EVAL_TASKS.items():
        if name.startswith("gb2_"):
            downstream_evaluator.tasks[name] = dataclasses.replace(
                task, eval_interval=GB2_EVAL_INTERVAL
            )
    return config


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
