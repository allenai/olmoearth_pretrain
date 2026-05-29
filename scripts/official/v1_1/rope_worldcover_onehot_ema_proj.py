"""ViT-base v1.1 RoPE + one-hot WorldCover, with EMA on the target projection layer.

Identical to scripts/official/v1_1/rope_worldcover_onehot.py (2D RoPE + one-hot
WorldCover on top of the hidden1 baseline) except that the target encoder is updated
with an exponential moving average of the online encoder instead of being frozen at
init (``ema_decay=(1.0, 1.0)`` in base.py).

The target encoder is only ever used to build the prediction targets, and with
``token_exit_cfg=0`` for every modality its tokens are just the initial pixel -> token
projection (see olmoearth_pretrain/nn/st_model.py: "exited tokens are just the linear
projection"). So EMA-updating the whole target encoder is, for target construction,
equivalent to enabling EMA on the projection layer alone -- the deeper transformer
blocks never contribute to a target.
"""

import logging

import base
import rope_worldcover_onehot

from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.train.train_module.contrastive_latentmim import (
    ContrastiveLatentMIMTrainModuleConfig,
)

logger = logging.getLogger(__name__)

# EMA schedule applied to the target encoder (effectively just the projection layer
# used to build targets, see module docstring). base.py freezes it with (1.0, 1.0).
EMA_DECAY = (0.996, 1.0)


def build_train_module_config(
    common: CommonComponents,
) -> ContrastiveLatentMIMTrainModuleConfig:
    """Build the train module config, enabling EMA on the target projection layer."""
    config = base.build_train_module_config(common)
    config.ema_decay = EMA_DECAY
    return config


def run() -> None:
    """Run the experiment."""
    main(
        common_components_builder=rope_worldcover_onehot.build_common_components,
        model_config_builder=rope_worldcover_onehot.build_model_config,
        train_module_config_builder=build_train_module_config,
        dataset_config_builder=base.build_dataset_config,
        dataloader_config_builder=base.build_dataloader_config,
        trainer_config_builder=base.build_trainer_config,
        visualize_config_builder=base.build_visualize_config,
    )


if __name__ == "__main__":
    run()
