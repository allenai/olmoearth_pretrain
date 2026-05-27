#!/bin/bash
# Spatial register bottleneck sweep (Perceiver-style), vs the hidden1 supervision run.
#
# Fixed for all runs:
#   - RoPE: base 10k, coordinate scale 0.25  (base10k_scale0.25)
#   - register_read_depth=1, register_latent_depth=4, register_num_heads=12 (= encoder)
#   - low-weight register supervision (supervision weight x0.1, set in the script)
#
# Sweep (8 runs): register_grid_size in {16, 32} x register_dim in {192, 288, 528, 768}.
# dims are multiples of 48 so head_dim = dim/12 is divisible by 4 (required by 2D RoPE);
# head_dims are 16, 24, 44, 64 respectively (768 = full model width = no width bottleneck).
#
# Plus 1 ablation: g16 / d528 with NO supervision head, to isolate whether the low-weight
# register supervision has any effect.
set -e

SCRIPT="scripts/archived/2026_04_22_add_hidden_layer_to_initial_projection/hidden1_supervision_register_bottleneck.py"
CLUSTER="ai2/jupiter"
LAUNCH_ARGS="--launch.num_gpus=8 --launch.priority=high --launch.clusters=[ai2/jupiter,ai2/ceres]"
WANDB_PROJECT="--trainer.callbacks.wandb.project=2026_04_22_add_hidden_layer_to_initial_projection"
ROPE="--model.encoder_config.rope_coordinate_scale=0.25 --model.decoder_config.rope_coordinate_scale=0.25"

# for GRID in 16 32; do
#     for DIM in 192 288 528 768; do
#         python "$SCRIPT" launch "regbtl_base10k_scale0.25_g${GRID}_d${DIM}" "$CLUSTER" \
#             $LAUNCH_ARGS \
#             $WANDB_PROJECT \
#             $ROPE \
#             --model.encoder_config.register_grid_size="${GRID}" \
#             --model.encoder_config.register_dim="${DIM}" \
#             --model.decoder_config.register_dim="${DIM}"
#     done
# done

# Ablation: g16 / d528 with NO supervision head (pure JEPA register bottleneck).
NOSUP_SCRIPT="scripts/archived/2026_04_22_add_hidden_layer_to_initial_projection/hidden1_register_bottleneck_no_supervision.py"
python "$NOSUP_SCRIPT" launch "regbtl_base10k_scale0.25_g16_d528_nosup" "$CLUSTER" \
    $LAUNCH_ARGS \
    $WANDB_PROJECT \
    $ROPE \
    --model.encoder_config.register_grid_size=16 \
    --model.encoder_config.register_dim=528 \
    --model.decoder_config.register_dim=528
