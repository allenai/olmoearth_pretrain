"""v1.2 BASELINE model + centroid-contrastive patch discrimination.

Identical to the official v1.2 base recipe (standard full-self-attention
encoder + cross-attn Predictor) except the token-level loss:
``modality_patch_discrimination_centroid_vec`` — within-sample targets are
mean-centered, grouped by connected components at cosine >= 0.90, and
predictions contrast against group centroids instead of all raw targets.
Calibration on real corpus targets (calibrate_group_threshold.py): raw
target cosines are ~0.99 for most pairs (dominant per-sample mean
direction); centered at theta=0.90 gives healthy group structure (S2 ~822
groups/sample, 15x compression; maps 7-130x) with ~0% degenerate samples.
"""

import logging

from base import (
    build_common_components,
    build_dataloader_config,
    build_dataset_config,
    build_model_config,
    build_visualize_config,
)
from base import (
    build_train_module_config as build_train_module_config_base,
)
from base import (
    build_trainer_config as build_trainer_config_base,
)
from olmo_core.train.config import TrainerConfig

from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.train.loss import LossConfig
from olmoearth_pretrain.train.train_module.contrastive_latentmim import (
    ContrastiveLatentMIMTrainModuleConfig,
)

logger = logging.getLogger(__name__)

WANDB_PROJECT = "2026_07_13_perceiver"

GROUP_THRESHOLD = 0.90


def build_train_module_config(
    common: CommonComponents,
) -> ContrastiveLatentMIMTrainModuleConfig:
    """v1.2 train module with the centroid-contrastive token loss."""
    config = build_train_module_config_base(common)
    config.loss_config = LossConfig(
        loss_config={
            "type": "modality_patch_discrimination_centroid_vec",
            "tau": 0.1,
            "group_threshold": GROUP_THRESHOLD,
            "center_targets": True,
        }
    )
    return config


def build_trainer_config(common: CommonComponents) -> TrainerConfig:
    """v1.2 trainer config in the perceiver-comparison wandb project."""
    trainer_config = build_trainer_config_base(common)
    trainer_config.callbacks["wandb"].project = WANDB_PROJECT
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
