"""ViT-base v1.1: 3D RoPE (t, row, col) + our separate-path simple temporal encoding.

Builds on henryh's `temporal_rope_mixed.py` (3D RoPE-Mixed over time + space,
temporal coordinate = days-since-2000 scaled to ~months) and adds our
separate-projection simple temporal encoding + batch-level year dropout. No
latlon (tested separately later), no class token, no CLIP — standard v1.1
loss (modality_patch_discrimination_masked_negatives_vec + mean-pool InfoNCE).

Rationale: relative time is handled by 3D temporal RoPE; our simple temporal
encoding adds the ABSOLUTE signal RoPE can't ([frac_year, sin, cos,
year_valid] — years since 2020 + annual phase + a dropout-able validity
flag). encoding_mode='separate' concatenates [channel | simple-temporal] onto
the patch projection (latlon slot disabled). Year dropout (collator, per rank
batch, rate 0.5) zeroes the absolute year -> year_valid=0 so the model is
robust to missing dates at eval (most eval tasks have fabricated dates).
"""

import logging

from base import (
    build_common_components,
    build_dataset_config,
    build_train_module_config,
    build_trainer_config,
    build_visualize_config,
)
from base import (
    build_dataloader_config as build_dataloader_config_base,
)
from base import (
    build_model_config as build_model_config_base,
)

from olmoearth_pretrain.data.dataloader import OlmoEarthDataLoaderConfig
from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig

logger = logging.getLogger(__name__)

# --- 3D RoPE-Mixed (t, row, col), matching henryh trope_mixed_tscale_months ---
SPATIAL_POS_ENCODING = "rope_3d_mixed"
ROPE_MIXED_BASE = 10.0
ROPE_TEMPORAL_COORDINATE_SCALE = 1.0 / 30.0  # days -> ~months

# --- Separate-path encodings: simple temporal only, NO latlon ---
ENCODING_MODE = "separate"
ENCODER_CHANNEL_ENCODING_DIM = 0  # redundant given per-bandset embed biases
DECODER_CHANNEL_ENCODING_DIM = 128  # required: decoder queries share one mask token
TEMPORAL_ENCODING_DIM = 4  # [frac_year, sin, cos, year_valid]
TEMPORAL_ENCODING_TYPE = "simple"
LATLON_ENCODING_DIM = 0  # no latlon this run

# --- Batch-level year dropout (collator, per rank batch). No latlon dropout. ---
YEAR_DROPOUT_RATE = 0.5
LATLON_DROPOUT_RATE = 0.0
METADATA_DROPOUT_VIEW_MODE = "shared"  # no class token => no view-pairing concern


def _apply(cfg) -> None:
    cfg.spatial_pos_encoding = SPATIAL_POS_ENCODING
    cfg.rope_mixed_base = ROPE_MIXED_BASE
    cfg.rope_temporal_coordinate_scale = ROPE_TEMPORAL_COORDINATE_SCALE
    cfg.encoding_mode = ENCODING_MODE
    cfg.temporal_encoding_dim = TEMPORAL_ENCODING_DIM
    cfg.temporal_encoding_type = TEMPORAL_ENCODING_TYPE
    cfg.latlon_encoding_dim = LATLON_ENCODING_DIM
    cfg.latlon_dropout_rate = 0.0  # dropout handled by the collator


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """v1.1 base + 3D RoPE-Mixed + separate-path simple temporal (no latlon/class token)."""
    config = build_model_config_base(common)
    _apply(config.encoder_config)
    _apply(config.decoder_config)
    config.encoder_config.channel_encoding_dim = ENCODER_CHANNEL_ENCODING_DIM
    config.decoder_config.channel_encoding_dim = DECODER_CHANNEL_ENCODING_DIM
    return config


def build_dataloader_config(common: CommonComponents) -> OlmoEarthDataLoaderConfig:
    """v1.1 dataloader + batch-level year dropout (no latlon dropout)."""
    config = build_dataloader_config_base(common)
    config.year_dropout_rate = YEAR_DROPOUT_RATE
    config.latlon_dropout_rate = LATLON_DROPOUT_RATE
    config.metadata_dropout_view_mode = METADATA_DROPOUT_VIEW_MODE
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
