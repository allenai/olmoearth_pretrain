#!/bin/bash
# Small v1.1 ViT-base RoPE sweep against the hidden1 run:
# https://wandb.ai/eai-ai2/2026_04_22_add_hidden_layer_to_initial_projection/runs/d7nfwd1i
#
# All runs use scripts/vnext/temporal_rope/rope.py, which imports v1_1/base.py and only
# changes spatial positional encoding fields.
set -e

SCRIPT="scripts/vnext/temporal_rope/rope.py"
WANDB_PROJECT="--trainer.callbacks.wandb.project=2026_04_22_add_hidden_layer_to_initial_projection"
LAUNCH_ARGS="--launch.num_gpus=8 --launch.priority=high --launch.clusters=[ai2/jupiter,ai2/ceres]"

python "$SCRIPT" launch rope_base10k_scale1 ai2/jupiter \
    $LAUNCH_ARGS \
    $WANDB_PROJECT

python "$SCRIPT" launch rope_base1k_scale1 ai2/jupiter \
    $LAUNCH_ARGS \
    $WANDB_PROJECT \
    --model.encoder_config.rope_base=1000 \
    --model.decoder_config.rope_base=1000

python "$SCRIPT" launch rope_base100k_scale1 ai2/jupiter \
    $LAUNCH_ARGS \
    $WANDB_PROJECT \
    --model.encoder_config.rope_base=100000 \
    --model.decoder_config.rope_base=100000

python "$SCRIPT" launch rope_base10k_scale0.25 ai2/jupiter \
    $LAUNCH_ARGS \
    $WANDB_PROJECT \
    --model.encoder_config.rope_coordinate_scale=0.25 \
    --model.decoder_config.rope_coordinate_scale=0.25
