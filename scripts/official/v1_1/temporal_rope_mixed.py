"""ViT-base v1.1 hidden-projection baseline with mixed 3D RoPE (t, row, col).

Variant of scripts/official/v1_1/rope_mixed.py that extends RoPE-Mixed
(Heo et al. 2024) from 2D to 3D: every per-head per-pair frequency becomes
a learnable 3-vector ``(theta_t, theta_row, theta_col)`` instead of a 2-vector.
The full head_dim is shared across axes (no chunk split). The slot-index
additive temporal encoding is dropped automatically; the calendar/month
additive encoding is preserved.

Temporal coordinate is days-since-2000. ``rope_temporal_coordinate_scale``
multiplies that into whatever unit you want (default raw days; set to 1/30
for months, 24 for hours, etc.). The learnable mixed frequencies then adapt
during training, so there is no separate ``rope_temporal_base`` knob in this
variant — only the input scale matters.
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

SPATIAL_POS_ENCODING = "rope_3d_mixed"
ROPE_MIXED_BASE = 10.0
# Days are large numbers (~7000 in 2020); scale into a more typical range so
# the learnable freqs don't have to fight magnitudes. 1/30 -> roughly months.
ROPE_TEMPORAL_COORDINATE_SCALE = 1.0 / 30.0


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """Build the v1.1 base model with mixed 3D RoPE."""
    config = build_model_config_base(common)

    config.encoder_config.spatial_pos_encoding = SPATIAL_POS_ENCODING
    config.encoder_config.rope_mixed_base = ROPE_MIXED_BASE
    config.encoder_config.rope_temporal_coordinate_scale = (
        ROPE_TEMPORAL_COORDINATE_SCALE
    )

    config.decoder_config.spatial_pos_encoding = SPATIAL_POS_ENCODING
    config.decoder_config.rope_mixed_base = ROPE_MIXED_BASE
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
