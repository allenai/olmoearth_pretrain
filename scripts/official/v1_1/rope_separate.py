"""ViT-base v1.1 with 2D RoPE + separate-path encodings.

Builds on ``scripts/official/v1_1/base.py`` (the hidden1 v1.1 base) and layers on:

1. **Spatial: axial 2D RoPE** (``spatial_pos_encoding='rope'``, base 10000,
   coordinate_scale 0.25) -- matches the configurable-RoPE setup used by the
   W&B run ``rope_base10k_scale0.25``.

2. **Temporal: static_temporal** (multi-frequency sin/cos of fractional year)
   -- multitemporal modalities pick this up; non-temporal modalities get zero
   in the temporal slot.

3. **Latlon: static_latlon with dropout** (sphere-mapped multi-frequency
   sin/cos of tile-center lat/lon). Per-sample bernoulli dropout at rate 0.5
   during training so the model is robust to the eval-time-with-no-latlon
   distribution; ``rate>=1.0`` would fully disable.

4. **Encoding mode: ``separate``** -- the encoding-side signals
   (channel + static_temporal + static_latlon) are concatenated into a
   ``channel + temporal + latlon = enc_dim`` vector that lives on a parallel
   path from the image patch projection. A single ``Linear(embedding +
   enc_dim, embedding)`` layer combines them. There is no additive mixing of
   encodings into the image projection.

No legacy positional encodings: no learnable time-index, no month embeddings,
no 2D-sincos absolute spatial encoding. RoPE handles spatial position inside
attention; everything else is the encoding-token path.
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

# --- Separate-path encodings ---
ENCODING_MODE = "separate"
CHANNEL_ENCODING_DIM = 128  # per-modality, per-bandset learnable channel embed
TEMPORAL_ENCODING_DIM = 128  # static multi-freq sincos of fractional year
LATLON_ENCODING_DIM = 192  # sphere-mapped multi-freq sincos (div by 6)
LATLON_DROPOUT_RATE = 0.5  # per-sample bernoulli; rate>=1.0 disables entirely


def _apply(cfg) -> None:
    cfg.spatial_pos_encoding = SPATIAL_POS_ENCODING
    cfg.rope_base = ROPE_BASE
    cfg.rope_coordinate_scale = ROPE_COORDINATE_SCALE
    cfg.encoding_mode = ENCODING_MODE
    cfg.channel_encoding_dim = CHANNEL_ENCODING_DIM
    cfg.temporal_encoding_dim = TEMPORAL_ENCODING_DIM
    cfg.latlon_encoding_dim = LATLON_ENCODING_DIM
    cfg.latlon_dropout_rate = LATLON_DROPOUT_RATE


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """Build the v1.1 base model with 2D RoPE + separate-path encodings."""
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
