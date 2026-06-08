#!/bin/bash
# Small v1.1 ViT-base RoPE-Mixed sweep, head-to-head with the axial RoPE runs in
# https://wandb.ai/eai-ai2/2026_04_22_add_hidden_layer_to_initial_projection
#
# All runs use scripts/official/v1_1/rope_mixed.py, which imports v1_1/base.py
# and only changes the spatial positional encoding fields.
set -e

SCRIPT="scripts/official/v1_1/rope_mixed.py"
PROJECT="2026_04_22_add_hidden_layer_to_initial_projection"
LAUNCH_ARGS="--launch.num_gpus=8 --launch.priority=high --launch.clusters=[ai2/jupiter,ai2/ceres]"

# Primary run: coordinate_scale=0.25, mixed_base=10000.
python "$SCRIPT" launch mixed_base10k_scale0.25 ai2/jupiter \
    $LAUNCH_ARGS \
    --trainer.callbacks.wandb.project="$PROJECT" \
    --model.encoder_config.rope_mixed_base=10000 \
    --model.decoder_config.rope_mixed_base=10000 \
    --model.encoder_config.rope_coordinate_scale=0.25 \
    --model.decoder_config.rope_coordinate_scale=0.25

# Same base, default coordinate scale (compare against the axial scale=1 runs).
python "$SCRIPT" launch mixed_base10k_scale1 ai2/jupiter \
    $LAUNCH_ARGS \
    --trainer.callbacks.wandb.project="$PROJECT" \
    --model.encoder_config.rope_mixed_base=10000 \
    --model.decoder_config.rope_mixed_base=10000 \
    --model.encoder_config.rope_coordinate_scale=1.0 \
    --model.decoder_config.rope_coordinate_scale=1.0

# Paper default base (Heo et al. 2024 used theta=10 for ViT-B RoPE-Mixed).
python "$SCRIPT" launch mixed_base10_scale0.25 ai2/jupiter \
    $LAUNCH_ARGS \
    --trainer.callbacks.wandb.project="$PROJECT" \
    --model.encoder_config.rope_mixed_base=10 \
    --model.decoder_config.rope_mixed_base=10 \
    --model.encoder_config.rope_coordinate_scale=0.25 \
    --model.decoder_config.rope_coordinate_scale=0.25

# Axial-paper vision base (theta=100 recommended in Eq.13 for axial 2D RoPE).
python "$SCRIPT" launch mixed_base100_scale0.25 ai2/jupiter \
    $LAUNCH_ARGS \
    --trainer.callbacks.wandb.project="$PROJECT" \
    --model.encoder_config.rope_mixed_base=100 \
    --model.decoder_config.rope_mixed_base=100 \
    --model.encoder_config.rope_coordinate_scale=0.25 \
    --model.decoder_config.rope_coordinate_scale=0.25
