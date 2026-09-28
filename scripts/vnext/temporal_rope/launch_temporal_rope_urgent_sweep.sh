#!/bin/bash
# v1.1 temporal-RoPE urgent sweep (8 GPUs each, ai2/jupiter).
#
# Launches three families head-to-head:
#   1. rope_large.py        -- ViT-large 2D RoPE baseline (the run that failed
#                              earlier because the script wasn't committed).
#   2. temporal_rope.py     -- axial 3D RoPE (t, row, col). Spatial + temporal
#                              base fixed; 2-run sweep over the TEMPORAL scale.
#   3. temporal_rope_mixed  -- learnable mixed 3D RoPE. Mixed base (2D init)
#                              fixed; 2-run sweep over the TEMPORAL scale (no
#                              temporal base knob -- freqs are learned).
#
# Temporal coordinate is days-since-2000, so scale=1.0 -> raw days and
# scale=0.0333 (~1/30) -> months. Axial temporal_base sets the rotation
# wavelength in those units.
set -e

PROJECT="2026_04_22_add_hidden_layer_to_initial_projection"
LAUNCH_ARGS="--launch.num_gpus=8 --launch.priority=urgent --launch.clusters=[ai2/jupiter]"

# Fixed 2D spatial knobs shared across the axial temporal runs.
SPATIAL_AXIAL="\
    --model.encoder_config.rope_base=10000 \
    --model.decoder_config.rope_base=10000 \
    --model.encoder_config.rope_coordinate_scale=1.0 \
    --model.decoder_config.rope_coordinate_scale=1.0"

# Fixed 2D init base shared across the mixed temporal runs.
SPATIAL_MIXED="\
    --model.encoder_config.rope_mixed_base=10 \
    --model.decoder_config.rope_mixed_base=10"

# ---------------------------------------------------------------------------
# 1. ViT-large 2D RoPE baseline.
# ---------------------------------------------------------------------------
python scripts/vnext/temporal_rope/rope_large.py launch large_rope_base10k_scale0.25 ai2/jupiter \
    $LAUNCH_ARGS \
    --trainer.callbacks.wandb.project="$PROJECT" \
    --model.encoder_config.rope_base=10000 \
    --model.decoder_config.rope_base=10000 \
    --model.encoder_config.rope_coordinate_scale=0.25 \
    --model.decoder_config.rope_coordinate_scale=0.25

# ---------------------------------------------------------------------------
# 2. Axial 3D RoPE: fixed temporal_base=1000, sweep temporal scale (days vs months).
# ---------------------------------------------------------------------------
AXIAL_SCRIPT="scripts/vnext/temporal_rope/temporal_rope.py"

python "$AXIAL_SCRIPT" launch trope_axial_tbase1k_tscale_days ai2/jupiter \
    $LAUNCH_ARGS $SPATIAL_AXIAL \
    --trainer.callbacks.wandb.project="$PROJECT" \
    --model.encoder_config.rope_temporal_base=1000 \
    --model.decoder_config.rope_temporal_base=1000 \
    --model.encoder_config.rope_temporal_coordinate_scale=1.0 \
    --model.decoder_config.rope_temporal_coordinate_scale=1.0

python "$AXIAL_SCRIPT" launch trope_axial_tbase1k_tscale_months ai2/jupiter \
    $LAUNCH_ARGS $SPATIAL_AXIAL \
    --trainer.callbacks.wandb.project="$PROJECT" \
    --model.encoder_config.rope_temporal_base=1000 \
    --model.decoder_config.rope_temporal_base=1000 \
    --model.encoder_config.rope_temporal_coordinate_scale=0.0333 \
    --model.decoder_config.rope_temporal_coordinate_scale=0.0333

# ---------------------------------------------------------------------------
# 3. Mixed 3D RoPE: sweep temporal scale only (freqs learned), fixed 2D base.
#    temporal_scale in {1.0 days, 0.0333 months}.
# ---------------------------------------------------------------------------
MIXED_SCRIPT="scripts/vnext/temporal_rope/temporal_rope_mixed.py"

python "$MIXED_SCRIPT" launch trope_mixed_tscale_days ai2/jupiter \
    $LAUNCH_ARGS $SPATIAL_MIXED \
    --trainer.callbacks.wandb.project="$PROJECT" \
    --model.encoder_config.rope_temporal_coordinate_scale=1.0 \
    --model.decoder_config.rope_temporal_coordinate_scale=1.0

python "$MIXED_SCRIPT" launch trope_mixed_tscale_months ai2/jupiter \
    $LAUNCH_ARGS $SPATIAL_MIXED \
    --trainer.callbacks.wandb.project="$PROJECT" \
    --model.encoder_config.rope_temporal_coordinate_scale=0.0333 \
    --model.decoder_config.rope_temporal_coordinate_scale=0.0333
