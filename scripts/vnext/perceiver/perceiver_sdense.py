"""Perceiver with spatially-dense, temporally-coarse latents (sdense).

Same as ``perceiver_base.py`` except the latent bottleneck is reallocated:
``latent_stride_hw=1`` anchors a latent at EVERY spatial position while
``latent_stride_t=6`` collapses time to at most two anchor slots, and the
dense read-out is spatial-only (sread). Latent count stays in base_2's
range — this tests whether deep per-position residual streams (not more
compute) close the dense-prediction (PASTIS) gap. Reads are interleaved at
blocks 0 and 6 as in base.
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
    """Perceiver base model with per-position latents and coarse time."""
    config = build_model_config_base(common)
    config.encoder_config.latent_stride_hw = 1
    config.encoder_config.latent_stride_t = 6
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
