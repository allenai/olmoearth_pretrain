#!/bin/bash
# Spatial register bottleneck sweep (Perceiver-style) — INDIVIDUAL launch commands.
#
# This is a MENU, not a run-top-to-bottom script. Run the variable block below ONCE
# (paste it / `source` the header), then run any individual launch command you still
# need — so you can skip runs that have already started.
#
# 16 runs: register_grid_size in {16, 32} x register_dim in {192, 288, 528, 768}, each:
#   - low-weight register supervision:  regbtl_base10k_scale0.25_g{G}_d{D}
#   - no supervision (pure JEPA):       regbtl_base10k_scale0.25_g{G}_d{D}_nosup
# All use RoPE base10k + coordinate scale 0.25; head_dim = dim/12 in {16, 24, 44, 64}.
# register_read_depth=1, register_latent_depth=4, register_num_heads=12 (set in scripts).

SCRIPT="scripts/archived/2026_04_22_add_hidden_layer_to_initial_projection/hidden1_supervision_register_bottleneck.py"
NOSUP_SCRIPT="scripts/archived/2026_04_22_add_hidden_layer_to_initial_projection/hidden1_register_bottleneck_no_supervision.py"
CLUSTER="ai2/jupiter"
LAUNCH_ARGS="--launch.num_gpus=8 --launch.priority=urgent --launch.clusters=[ai2/jupiter,ai2/ceres]"
WANDB_PROJECT="--trainer.callbacks.wandb.project=2026_04_22_add_hidden_layer_to_initial_projection"
ROPE="--model.encoder_config.rope_coordinate_scale=0.25 --model.decoder_config.rope_coordinate_scale=0.25"

# ============================ supervised (8) ============================

python "$SCRIPT" launch "regbtl_base10k_scale0.25_g16_d192" "$CLUSTER" \
    $LAUNCH_ARGS $WANDB_PROJECT $ROPE \
    --model.encoder_config.register_grid_size=16 --model.encoder_config.register_dim=192 --model.decoder_config.register_dim=192

python "$SCRIPT" launch "regbtl_base10k_scale0.25_g16_d288" "$CLUSTER" \
    $LAUNCH_ARGS $WANDB_PROJECT $ROPE \
    --model.encoder_config.register_grid_size=16 --model.encoder_config.register_dim=288 --model.decoder_config.register_dim=288

python "$SCRIPT" launch "regbtl_base10k_scale0.25_g16_d528" "$CLUSTER" \
    $LAUNCH_ARGS $WANDB_PROJECT $ROPE \
    --model.encoder_config.register_grid_size=16 --model.encoder_config.register_dim=528 --model.decoder_config.register_dim=528

python "$SCRIPT" launch "regbtl_base10k_scale0.25_g16_d768" "$CLUSTER" \
    $LAUNCH_ARGS $WANDB_PROJECT $ROPE \
    --model.encoder_config.register_grid_size=16 --model.encoder_config.register_dim=768 --model.decoder_config.register_dim=768

python "$SCRIPT" launch "regbtl_base10k_scale0.25_g32_d192" "$CLUSTER" \
    $LAUNCH_ARGS $WANDB_PROJECT $ROPE \
    --model.encoder_config.register_grid_size=32 --model.encoder_config.register_dim=192 --model.decoder_config.register_dim=192

python "$SCRIPT" launch "regbtl_base10k_scale0.25_g32_d288" "$CLUSTER" \
    $LAUNCH_ARGS $WANDB_PROJECT $ROPE \
    --model.encoder_config.register_grid_size=32 --model.encoder_config.register_dim=288 --model.decoder_config.register_dim=288

python "$SCRIPT" launch "regbtl_base10k_scale0.25_g32_d528" "$CLUSTER" \
    $LAUNCH_ARGS $WANDB_PROJECT $ROPE \
    --model.encoder_config.register_grid_size=32 --model.encoder_config.register_dim=528 --model.decoder_config.register_dim=528

python "$SCRIPT" launch "regbtl_base10k_scale0.25_g32_d768" "$CLUSTER" \
    $LAUNCH_ARGS $WANDB_PROJECT $ROPE \
    --model.encoder_config.register_grid_size=32 --model.encoder_config.register_dim=768 --model.decoder_config.register_dim=768

# ========================= no supervision (8) ==========================

python "$NOSUP_SCRIPT" launch "regbtl_base10k_scale0.25_g16_d192_nosup" "$CLUSTER" \
    $LAUNCH_ARGS $WANDB_PROJECT $ROPE \
    --model.encoder_config.register_grid_size=16 --model.encoder_config.register_dim=192 --model.decoder_config.register_dim=192

python "$NOSUP_SCRIPT" launch "regbtl_base10k_scale0.25_g16_d288_nosup" "$CLUSTER" \
    $LAUNCH_ARGS $WANDB_PROJECT $ROPE \
    --model.encoder_config.register_grid_size=16 --model.encoder_config.register_dim=288 --model.decoder_config.register_dim=288

python "$NOSUP_SCRIPT" launch "regbtl_base10k_scale0.25_g16_d528_nosup" "$CLUSTER" \
    $LAUNCH_ARGS $WANDB_PROJECT $ROPE \
    --model.encoder_config.register_grid_size=16 --model.encoder_config.register_dim=528 --model.decoder_config.register_dim=528

python "$NOSUP_SCRIPT" launch "regbtl_base10k_scale0.25_g16_d768_nosup" "$CLUSTER" \
    $LAUNCH_ARGS $WANDB_PROJECT $ROPE \
    --model.encoder_config.register_grid_size=16 --model.encoder_config.register_dim=768 --model.decoder_config.register_dim=768

python "$NOSUP_SCRIPT" launch "regbtl_base10k_scale0.25_g32_d192_nosup" "$CLUSTER" \
    $LAUNCH_ARGS $WANDB_PROJECT $ROPE \
    --model.encoder_config.register_grid_size=32 --model.encoder_config.register_dim=192 --model.decoder_config.register_dim=192

python "$NOSUP_SCRIPT" launch "regbtl_base10k_scale0.25_g32_d288_nosup" "$CLUSTER" \
    $LAUNCH_ARGS $WANDB_PROJECT $ROPE \
    --model.encoder_config.register_grid_size=32 --model.encoder_config.register_dim=288 --model.decoder_config.register_dim=288

python "$NOSUP_SCRIPT" launch "regbtl_base10k_scale0.25_g32_d528_nosup" "$CLUSTER" \
    $LAUNCH_ARGS $WANDB_PROJECT $ROPE \
    --model.encoder_config.register_grid_size=32 --model.encoder_config.register_dim=528 --model.decoder_config.register_dim=528

python "$NOSUP_SCRIPT" launch "regbtl_base10k_scale0.25_g32_d768_nosup" "$CLUSTER" \
    $LAUNCH_ARGS $WANDB_PROJECT $ROPE \
    --model.encoder_config.register_grid_size=32 --model.encoder_config.register_dim=768 --model.decoder_config.register_dim=768
