#!/bin/bash
# v1.1 temporal-RoPE urgent sweep (8 GPUs each, ai2/jupiter).
#
# Launches three families head-to-head:
#   1. rope_large.py        -- ViT-large 2D RoPE baseline (the run that failed
#                              earlier because the script wasn't committed).
#   2. temporal_rope.py     -- axial 3D RoPE (t, row, col). Spatial base/scale
#                              fixed; small sweep over the TEMPORAL base + scale.
#   3. temporal_rope_mixed  -- learnable mixed 3D RoPE. Mixed base (2D init)
#                              fixed; small sweep over the TEMPORAL scale (no
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
python scripts/official/v1_1/rope_large.py launch large_rope_base10k_scale1 ai2/jupiter \
    $LAUNCH_ARGS \
    --trainer.callbacks.wandb.project="$PROJECT" \
    --model.encoder_config.rope_base=10000 \
    --model.decoder_config.rope_base=10000 \
    --model.encoder_config.rope_coordinate_scale=1.0 \
    --model.decoder_config.rope_coordinate_scale=1.0

# ---------------------------------------------------------------------------
# 2. Axial 3D RoPE: sweep temporal base x temporal scale, fixed spatial.
#    temporal_base in {1000, 10000}, temporal_scale in {1.0 days, 0.0333 months}.
# ---------------------------------------------------------------------------
AXIAL_SCRIPT="scripts/official/v1_1/temporal_rope.py"

python "$AXIAL_SCRIPT" launch trope_axial_tbase1k_tscale_days ai2/jupiter \
    $LAUNCH_ARGS $SPATIAL_AXIAL \
    --trainer.callbacks.wandb.project="$PROJECT" \
    --model.encoder_config.rope_temporal_base=1000 \
    --model.decoder_config.rope_temporal_base=1000 \
    --model.encoder_config.rope_temporal_coordinate_scale=1.0 \
    --model.decoder_config.rope_temporal_coordinate_scale=1.0

python "$AXIAL_SCRIPT" launch trope_axial_tbase10k_tscale_days ai2/jupiter \
    $LAUNCH_ARGS $SPATIAL_AXIAL \
    --trainer.callbacks.wandb.project="$PROJECT" \
    --model.encoder_config.rope_temporal_base=10000 \
    --model.decoder_config.rope_temporal_base=10000 \
    --model.encoder_config.rope_temporal_coordinate_scale=1.0 \
    --model.decoder_config.rope_temporal_coordinate_scale=1.0

python "$AXIAL_SCRIPT" launch trope_axial_tbase1k_tscale_months ai2/jupiter \
    $LAUNCH_ARGS $SPATIAL_AXIAL \
    --trainer.callbacks.wandb.project="$PROJECT" \
    --model.encoder_config.rope_temporal_base=1000 \
    --model.decoder_config.rope_temporal_base=1000 \
    --model.encoder_config.rope_temporal_coordinate_scale=0.0333 \
    --model.decoder_config.rope_temporal_coordinate_scale=0.0333

python "$AXIAL_SCRIPT" launch trope_axial_tbase100_tscale_months ai2/jupiter \
    $LAUNCH_ARGS $SPATIAL_AXIAL \
    --trainer.callbacks.wandb.project="$PROJECT" \
    --model.encoder_config.rope_temporal_base=100 \
    --model.decoder_config.rope_temporal_base=100 \
    --model.encoder_config.rope_temporal_coordinate_scale=0.0333 \
    --model.decoder_config.rope_temporal_coordinate_scale=0.0333

# ---------------------------------------------------------------------------
# 3. Mixed 3D RoPE: sweep temporal scale only (freqs learned), fixed 2D base.
#    temporal_scale in {1.0 days, 0.0333 months, 0.00274 years}.
# ---------------------------------------------------------------------------
MIXED_SCRIPT="scripts/official/v1_1/temporal_rope_mixed.py"

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

python "$MIXED_SCRIPT" launch trope_mixed_tscale_years ai2/jupiter \
    $LAUNCH_ARGS $SPATIAL_MIXED \
    --trainer.callbacks.wandb.project="$PROJECT" \
    --model.encoder_config.rope_temporal_coordinate_scale=0.00274 \
    --model.decoder_config.rope_temporal_coordinate_scale=0.00274
