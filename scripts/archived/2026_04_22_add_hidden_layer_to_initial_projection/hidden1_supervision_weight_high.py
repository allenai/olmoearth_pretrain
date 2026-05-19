"""hidden1 + supervision with supervision loss weights scaled by 10.0.

Investigates whether heavy supervision crowds out the contrastive signal on
radiometric modalities (m-eurosat / pastis evals would drop while
pretrain_*_geo_sentinel2_l2a probes climb).
"""

import logging
from functools import partial

from hidden1 import (
    build_common_components,
    build_dataloader_config,
    build_dataset_config,
    build_train_module_config,
    build_trainer_config,
    build_visualize_config,
)
from hidden1_supervision import build_model_config as _build_model_config

from olmoearth_pretrain.internal.experiment import main

logger = logging.getLogger(__name__)

WEIGHT_MULTIPLIER = 10.0

build_model_config = partial(_build_model_config, weight_multiplier=WEIGHT_MULTIPLIER)


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
