#!/bin/bash
# v1.2 size sweep: nano / tiny / small / large.
#
# v1.2 = v1.1 (hidden patch-embed projection) + the winning mixed 3D RoPE
# config from the temporal-RoPE sweep (run nd3xh7py / trope_mixed_tscale_months
# in 2026_04_22_add_hidden_layer_to_initial_projection): spatial_pos_encoding=
# rope_3d_mixed, rope_mixed_base=10, rope_temporal_coordinate_scale=1/30 (~months).
#
# All four sizes use decoder depth 4 (the *_shallow_decoder presets; nano is
# already depth 4). RoPE/temporal knobs are baked into v1_2/base.py.
set -e

PROJECT="2026_06_12_v1_2_size_sweep"
LAUNCH_ARGS="--launch.num_gpus=8 --launch.priority=high --launch.clusters=[ai2/jupiter,ai2/ceres]"

python scripts/official/v1_2/nano.py launch v1_2_nano ai2/jupiter \
    $LAUNCH_ARGS \
    --trainer.callbacks.wandb.project="$PROJECT"

python scripts/official/v1_2/tiny.py launch v1_2_tiny ai2/jupiter \
    $LAUNCH_ARGS \
    --trainer.callbacks.wandb.project="$PROJECT"

python scripts/official/v1_2/small.py launch v1_2_small ai2/jupiter \
    $LAUNCH_ARGS \
    --trainer.callbacks.wandb.project="$PROJECT"

python scripts/official/v1_2/large.py launch v1_2_large ai2/jupiter \
    $LAUNCH_ARGS \
    --trainer.callbacks.wandb.project="$PROJECT"
