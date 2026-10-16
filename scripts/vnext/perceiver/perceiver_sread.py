"""Perceiver with a spatial-only dense read-out (one token per location).

Same as ``perceiver_base.py`` except ``readout_spatial_only=True``: the dense
read-out emits ONE fused token per (h, w) — queries carry (t=0, row, col)
coordinates and no month embedding — broadcast across each modality's
timesteps. The pointwise head then distinguishes timesteps only via its month
embedding, so the location token must encode the full seasonal profile.
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
    """Perceiver base model with a spatial-only dense read-out."""
    config = build_model_config_base(common)
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
