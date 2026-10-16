"""Perceiver with a designated class latent (CLS) — variant of perceiver_base.

Same as ``perceiver_base.py`` plus:
- ``num_class_latents=1``: a position-less learned class latent whose
  projection replaces the pooled instance embedding for the InfoNCE loss.
- Classification evals (m-eurosat, m_so2sat) consume the class embedding
  directly via ``PoolingType.CLS``. Set here in code because the task key
  ``m-eurosat`` contains a dash, which the dotted CLI override parser
  mangles to ``m_eurosat`` (creating a new, invalid task entry).
"""

import logging

from olmo_core.train.config import TrainerConfig
from perceiver_base import (
    build_common_components,
    build_dataloader_config,
    build_dataset_config,
    build_train_module_config,
    build_visualize_config,
)
from perceiver_base import (
    build_model_config as build_model_config_base,
)
from perceiver_base import (
    build_trainer_config as build_trainer_config_base,
)

from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig
from olmoearth_pretrain.nn.pooling import PoolingType

logger = logging.getLogger(__name__)

CLS_EVAL_TASKS = ["m-eurosat", "m_so2sat"]


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """Perceiver base model with one class latent."""
    config = build_model_config_base(common)
    config.encoder_config.num_class_latents = 1
    return config


def build_trainer_config(common: CommonComponents) -> TrainerConfig:
    """Perceiver trainer config with CLS pooling for classification evals."""
    trainer_config = build_trainer_config_base(common)
    tasks = trainer_config.callbacks["downstream_evaluator"].tasks
    for task_name in CLS_EVAL_TASKS:
        tasks[task_name].pooling_type = PoolingType.CLS
    return trainer_config


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
