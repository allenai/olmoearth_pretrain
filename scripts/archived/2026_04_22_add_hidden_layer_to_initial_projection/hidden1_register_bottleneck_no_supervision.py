"""No-supervision ablation of the register bottleneck.

Identical to ``hidden1_supervision_register_bottleneck.py`` (2D RoPE + Perceiver-style
register bottleneck) but **drops the supervision head entirely** (`supervision_head_config
= None`), to isolate whether the low-weight register supervision has any effect. The
encoder register bottleneck and the decoder-reads-only-registers path are unchanged; this
is a pure JEPA bottleneck. The `ContrastiveLatentMIM` train module skips supervision when
`model.supervision_head is None`.
"""

import logging

from hidden1_supervision_register_bottleneck import (
    build_common_components,
    build_dataloader_config,
    build_dataset_config,
    build_train_module_config,
    build_trainer_config,
    build_visualize_config,
)
from hidden1_supervision_register_bottleneck import (
    build_model_config as build_model_config_with_supervision,
)

from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig

logger = logging.getLogger(__name__)


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """Register bottleneck with the supervision head removed (pure JEPA bottleneck)."""
    config = build_model_config_with_supervision(common)
    config.supervision_head_config = None
    return config


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
