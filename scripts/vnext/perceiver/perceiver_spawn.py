"""Perceiver with predicted child queries (spawn).

Coarse anchored latents (stride 4 spatial, stride 6 temporal) read the
tokens and process for 4 blocks; then each latent PREDICTS 8 child queries —
content vector + free (t, row, col) coordinate via tanh offsets scaled by
the sample extent — so the model decides where to spend its fine-grained
query budget. Children join the latent set for the reads at blocks 4/8 and
all remaining processing, and serve as dense read-out K/V alongside the
coarse coverage grid. Read-out is spatial-only (sread). Total latent count
~75% of base_2's, so any win is not a budget increase.
"""

import logging

from perceiver_base import (
    build_common_components,
    build_dataloader_config,
    build_dataset_config,
    build_train_module_config,
    build_trainer_config,
    build_visualize_config,
)
from perceiver_base import (
    build_model_config as build_model_config_base,
)

from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig

logger = logging.getLogger(__name__)


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """Perceiver with coarse anchors spawning free-coordinate children."""
    config = build_model_config_base(common)
    config.encoder_config.latent_stride_hw = 4
    config.encoder_config.latent_stride_t = 6
    config.encoder_config.num_reads = 3
    config.encoder_config.spawn_layer = 4
    config.encoder_config.spawn_children = 8
    config.encoder_config.readout_spatial_only = True
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
