"""hidden1 + supervision with the InfoNCE contrastive loss disabled.

Tests whether supervision on the map modalities can substitute for the
token-contrastive signal entirely. If pretrain_*_geo_sentinel2_l2a probes
hold and downstream evals (m-eurosat / pastis) stay flat, the masked-
negatives contrastive on map modalities was at best neutral.
"""

import logging

from hidden1 import (
    build_common_components,
    build_dataloader_config,
    build_dataset_config,
    build_trainer_config,
    build_visualize_config,
)
from hidden1 import build_train_module_config as _base_build_train_module_config
from hidden1_supervision import build_model_config

from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.train.train_module.contrastive_latentmim import (
    ContrastiveLatentMIMTrainModuleConfig,
)

logger = logging.getLogger(__name__)


def build_train_module_config(
    common: CommonComponents,
) -> ContrastiveLatentMIMTrainModuleConfig:
    """Build the train module config with contrastive disabled."""
    cfg = _base_build_train_module_config(common)
    cfg.contrastive_config = None
    return cfg


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
