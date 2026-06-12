"""ViT-base v1.1 hidden-projection baseline with axial 3D RoPE (t, row, col).

Variant of scripts/vnext/temporal_rope/rope.py that extends 2D axial spatial RoPE
to 3D by also rotating queries/keys along the temporal axis. The slot-index
additive temporal encoding is automatically dropped when ``rope_3d`` is set;
the calendar/month additive encoding is preserved.

The temporal coordinate is days-since-2000 derived from the per-sample
timestamps, so models see real calendar deltas (not slot indices). The
temporal axis has its own configurable base and coordinate scale to handle
the very different magnitude of days vs. patch-grid coordinates.

By default the temporal axis claims 25% of head_dim (matching the existing
1/4 split used by absolute encodings), with the remaining 75% split evenly
between row and col axes.
"""

import logging

from base import (
    build_common_components,
    build_dataloader_config,
    build_dataset_config,
    build_train_module_config,
    build_trainer_config,
    build_visualize_config,
)
from base import (
    build_model_config as build_model_config_base,
)

from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig

logger = logging.getLogger(__name__)

SPATIAL_POS_ENCODING = "rope_3d"
ROPE_BASE = 10000.0
ROPE_COORDINATE_SCALE = 1.0
TEMPORAL_ROPE_DIM_FRAC = 0.25
# Temporal axis sees days. With base=1000 and head_dim_t=16 (ViT-base), the
# lowest-frequency wavelength is ~1000 days, so relative deltas of 30-365 days
# produce meaningful (not microscopic) rotations.
ROPE_TEMPORAL_BASE = 1000.0
# Raw days; override with e.g. 1/30 for months or 24 for hours.
ROPE_TEMPORAL_COORDINATE_SCALE = 1.0


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """Build the v1.1 base model with axial 3D RoPE."""
    config = build_model_config_base(common)

    config.encoder_config.spatial_pos_encoding = SPATIAL_POS_ENCODING
    config.encoder_config.rope_base = ROPE_BASE
    config.encoder_config.rope_coordinate_scale = ROPE_COORDINATE_SCALE
    config.encoder_config.temporal_rope_dim_frac = TEMPORAL_ROPE_DIM_FRAC
    config.encoder_config.rope_temporal_base = ROPE_TEMPORAL_BASE
    config.encoder_config.rope_temporal_coordinate_scale = (
        ROPE_TEMPORAL_COORDINATE_SCALE
    )

    config.decoder_config.spatial_pos_encoding = SPATIAL_POS_ENCODING
    config.decoder_config.rope_base = ROPE_BASE
    config.decoder_config.rope_coordinate_scale = ROPE_COORDINATE_SCALE
    config.decoder_config.temporal_rope_dim_frac = TEMPORAL_ROPE_DIM_FRAC
    config.decoder_config.rope_temporal_base = ROPE_TEMPORAL_BASE
    config.decoder_config.rope_temporal_coordinate_scale = (
        ROPE_TEMPORAL_COORDINATE_SCALE
    )

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
