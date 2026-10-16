"""Centroid loss, RAW-POSITIVE variant, on the v1.2 baseline.

Same as ``v1_2_centroid.py`` except ``positive_mode="raw"``: each
prediction's positive is its own (centered, normalized) target — keeping
sub-threshold detail in the positive direction — while negatives remain
the OTHER groups' centroids (compressed, false-negative-free).
"""

import logging

from base import build_visualize_config
from v1_2_centroid import (
    GROUP_THRESHOLD,
    build_common_components,
    build_dataloader_config,
    build_dataset_config,
    build_model_config,
    build_trainer_config,
)
from v1_2_centroid import (
    build_train_module_config as build_train_module_config_centroid,
)

from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.train.loss import LossConfig
from olmoearth_pretrain.train.train_module.contrastive_latentmim import (
    ContrastiveLatentMIMTrainModuleConfig,
)

logger = logging.getLogger(__name__)


def build_train_module_config(
    common: CommonComponents,
) -> ContrastiveLatentMIMTrainModuleConfig:
    """Centroid train module with raw positives."""
    config = build_train_module_config_centroid(common)
    config.loss_config = LossConfig(
        loss_config={
            "type": "modality_patch_discrimination_centroid_vec",
            "tau": 0.1,
            "group_threshold": GROUP_THRESHOLD,
            "center_targets": True,
            "positive_mode": "raw",
        }
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
