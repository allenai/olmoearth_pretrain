"""ViT-base v1.1 with 2D RoPE + separate-path encodings, MINIMAL temporal/latlon.

Identical to ``rope_separate.py`` (RoPE spatial + separate-projection encoding
mode + latlon dropout) EXCEPT the temporal and lat/lon encodings are swapped for
minimal 3-number representations. Tests whether the model can learn from raw,
low-dimensional conditioning signals (projected up by the combine_proj layer)
instead of high-dimensional multi-frequency sinusoidal expansions.

- **Temporal (3 numbers)**: ``[frac_year, sin(2*pi*frac_year), cos(2*pi*frac_year)]``
  where ``frac_year = year + day_of_year/365.25 - 2020``. Channel 0 is linear
  "years since 2020" (absolute time); channels 1-2 are the annual phase (same
  day-of-year maps to the same value across years).
- **Latlon (3 numbers)**: raw unit-sphere ``[x, y, z]`` from (lat, lon). No
  frequency expansion; longitude wrap-around and pole behavior exact.

Everything else matches rope_separate.py: RoPE base 10000 / scale 0.25,
separate encoding mode, channel_encoding_dim 128, latlon_dropout_rate 0.5,
and the v1.1 hidden1 base recipe (single bandset, band dropout, updated
masking/loss, beta2=0.95).
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

# --- Spatial RoPE (matches rope_base10k_scale0.25 W&B run) ---
SPATIAL_POS_ENCODING = "rope"
ROPE_BASE = 10000.0
ROPE_COORDINATE_SCALE = 0.25

# --- Separate-path encodings, MINIMAL temporal/latlon ---
ENCODING_MODE = "separate"
CHANNEL_ENCODING_DIM = 128  # per-modality, per-bandset learnable channel embed
TEMPORAL_ENCODING_DIM = 3  # [frac_year, sin, cos]
LATLON_ENCODING_DIM = 3  # unit-sphere [x, y, z]
TEMPORAL_ENCODING_TYPE = "simple"
LATLON_ENCODING_TYPE = "simple"
LATLON_DROPOUT_RATE = 0.5  # per-sample bernoulli; rate>=1.0 disables entirely


def _apply(cfg) -> None:
    cfg.spatial_pos_encoding = SPATIAL_POS_ENCODING
    cfg.rope_base = ROPE_BASE
    cfg.rope_coordinate_scale = ROPE_COORDINATE_SCALE
    cfg.encoding_mode = ENCODING_MODE
    cfg.channel_encoding_dim = CHANNEL_ENCODING_DIM
    cfg.temporal_encoding_dim = TEMPORAL_ENCODING_DIM
    cfg.latlon_encoding_dim = LATLON_ENCODING_DIM
    cfg.temporal_encoding_type = TEMPORAL_ENCODING_TYPE
    cfg.latlon_encoding_type = LATLON_ENCODING_TYPE
    cfg.latlon_dropout_rate = LATLON_DROPOUT_RATE


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """Build the v1.1 base model with RoPE + simple separate-path encodings."""
    config = build_model_config_base(common)
    _apply(config.encoder_config)
    _apply(config.decoder_config)
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
