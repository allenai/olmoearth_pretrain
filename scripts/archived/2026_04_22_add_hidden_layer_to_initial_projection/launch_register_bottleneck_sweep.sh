#!/bin/bash
# Spatial register bottleneck sweep (Perceiver-style) — INDIVIDUAL launch commands.
#
# This is a MENU, not a run-top-to-bottom script. Run the variable block below ONCE
# (paste it / `source` the header), then run any individual launch command you still
# need — so you can skip runs that have already started.
#
# 24 runs: register_grid_size in {8, 16, 32} x register_dim in {192, 288, 528, 768}, each:
#   - low-weight register supervision:  regbtl_base10k_scale0.25_g{G}_d{D}
#   - no supervision (pure JEPA):       regbtl_base10k_scale0.25_g{G}_d{D}_nosup
# All use RoPE base10k + coordinate scale 0.25; head_dim = dim/12 in {16, 24, 44, 64}.
# Registers per grid: g8 -> 64, g16 -> 256, g32 -> 1024.
# register_read_depth=1, register_latent_depth=4, register_num_heads=12 (set in scripts).
#
# + 2 dynamic-grid runs (register_grid_size=0): a SINGLE learned latent cloned across
#   a grid that MATCHES THE PATCH GRID at forward time (translation-invariant prior, no
#   fixed grid size). d768, sup + nosup. Tagged "gdyn".
# + 4 interleaved-read runs (register_interleave=true): [read -> self] x register_latent_depth
#   instead of one up-front read (Perceiver/DETR/Flamingo). d768, g16 + gdyn, sup + nosup.
#   Tagged "il". See the final two sections below.

SCRIPT="scripts/archived/2026_04_22_add_hidden_layer_to_initial_projection/hidden1_supervision_register_bottleneck.py"
NOSUP_SCRIPT="scripts/archived/2026_04_22_add_hidden_layer_to_initial_projection/hidden1_register_bottleneck_no_supervision.py"
CLUSTER="ai2/jupiter"
LAUNCH_ARGS="--launch.num_gpus=8 --launch.priority=urgent --launch.clusters=[ai2/jupiter,ai2/ceres]"
WANDB_PROJECT="--trainer.callbacks.wandb.project=2026_04_22_add_hidden_layer_to_initial_projection"
ROPE="--model.encoder_config.rope_coordinate_scale=0.25 --model.decoder_config.rope_coordinate_scale=0.25"

# ============================ supervised (12) ===========================

python "$SCRIPT" launch "regbtl_base10k_scale0.25_g8_d192" "$CLUSTER" \
    $LAUNCH_ARGS $WANDB_PROJECT $ROPE \
    --model.encoder_config.register_grid_size=8 --model.encoder_config.register_dim=192 --model.decoder_config.register_dim=192

python "$SCRIPT" launch "regbtl_base10k_scale0.25_g8_d288" "$CLUSTER" \
    $LAUNCH_ARGS $WANDB_PROJECT $ROPE \
    --model.encoder_config.register_grid_size=8 --model.encoder_config.register_dim=288 --model.decoder_config.register_dim=288

python "$SCRIPT" launch "regbtl_base10k_scale0.25_g8_d528" "$CLUSTER" \
    $LAUNCH_ARGS $WANDB_PROJECT $ROPE \
    --model.encoder_config.register_grid_size=8 --model.encoder_config.register_dim=528 --model.decoder_config.register_dim=528

python "$SCRIPT" launch "regbtl_base10k_scale0.25_g8_d768" "$CLUSTER" \
    $LAUNCH_ARGS $WANDB_PROJECT $ROPE \
    --model.encoder_config.register_grid_size=8 --model.encoder_config.register_dim=768 --model.decoder_config.register_dim=768

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

# ========================= no supervision (12) =========================

python "$NOSUP_SCRIPT" launch "regbtl_base10k_scale0.25_g8_d192_nosup" "$CLUSTER" \
    $LAUNCH_ARGS $WANDB_PROJECT $ROPE \
    --model.encoder_config.register_grid_size=8 --model.encoder_config.register_dim=192 --model.decoder_config.register_dim=192

python "$NOSUP_SCRIPT" launch "regbtl_base10k_scale0.25_g8_d288_nosup" "$CLUSTER" \
    $LAUNCH_ARGS $WANDB_PROJECT $ROPE \
    --model.encoder_config.register_grid_size=8 --model.encoder_config.register_dim=288 --model.decoder_config.register_dim=288

python "$NOSUP_SCRIPT" launch "regbtl_base10k_scale0.25_g8_d528_nosup" "$CLUSTER" \
    $LAUNCH_ARGS $WANDB_PROJECT $ROPE \
    --model.encoder_config.register_grid_size=8 --model.encoder_config.register_dim=528 --model.decoder_config.register_dim=528

python "$NOSUP_SCRIPT" launch "regbtl_base10k_scale0.25_g8_d768_nosup" "$CLUSTER" \
    $LAUNCH_ARGS $WANDB_PROJECT $ROPE \
    --model.encoder_config.register_grid_size=8 --model.encoder_config.register_dim=768 --model.decoder_config.register_dim=768

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

# ==================== dynamic grid: single cloned latent (2) ====================
# register_grid_size=0 -> one shared latent cloned across the patch grid (no fixed grid).
# 0 (not null) is the dynamic sentinel so it survives config serialization; the decoder
# has no grid field.

python "$SCRIPT" launch "regbtl_base10k_scale0.25_gdyn_d768" "$CLUSTER" \
    $LAUNCH_ARGS $WANDB_PROJECT $ROPE \
    --model.encoder_config.register_grid_size=0 --model.encoder_config.register_dim=768 --model.decoder_config.register_dim=768

python "$NOSUP_SCRIPT" launch "regbtl_base10k_scale0.25_gdyn_d768_nosup" "$CLUSTER" \
    $LAUNCH_ARGS $WANDB_PROJECT $ROPE \
    --model.encoder_config.register_grid_size=0 --model.encoder_config.register_dim=768 --model.decoder_config.register_dim=768

# =================== interleaved reads: [read -> self] x4 (4) ===================
# register_interleave=true interleaves each cross-attention read with a latent self-
# attention block (Perceiver/DETR/Flamingo) so the registers re-query after each refine,
# instead of the default single up-front read. register_latent_depth=4 -> 4 reads + 4 self.
# d768, on the g16 fixed grid and the gdyn dynamic grid, sup + nosup. Tagged "il".

python "$SCRIPT" launch "regbtl_base10k_scale0.25_g16_d768_il" "$CLUSTER" \
    $LAUNCH_ARGS $WANDB_PROJECT $ROPE \
    --model.encoder_config.register_grid_size=16 --model.encoder_config.register_dim=768 --model.decoder_config.register_dim=768 \
    --model.encoder_config.register_interleave=true

python "$NOSUP_SCRIPT" launch "regbtl_base10k_scale0.25_g16_d768_il_nosup" "$CLUSTER" \
    $LAUNCH_ARGS $WANDB_PROJECT $ROPE \
    --model.encoder_config.register_grid_size=16 --model.encoder_config.register_dim=768 --model.decoder_config.register_dim=768 \
    --model.encoder_config.register_interleave=true

python "$SCRIPT" launch "regbtl_base10k_scale0.25_gdyn_d768_il" "$CLUSTER" \
    $LAUNCH_ARGS $WANDB_PROJECT $ROPE \
    --model.encoder_config.register_grid_size=0 --model.encoder_config.register_dim=768 --model.decoder_config.register_dim=768 \
    --model.encoder_config.register_interleave=true

python "$NOSUP_SCRIPT" launch "regbtl_base10k_scale0.25_gdyn_d768_il_nosup" "$CLUSTER" \
    $LAUNCH_ARGS $WANDB_PROJECT $ROPE \
    --model.encoder_config.register_grid_size=0 --model.encoder_config.register_dim=768 --model.decoder_config.register_dim=768 \
    --model.encoder_config.register_interleave=true

# ============ no instance contrastive loss: gdyn_d768_il, supervised (1) ============
# Identical to regbtl_base10k_scale0.25_gdyn_d768_il above, but drops the instance-level
# InfoNCE contrastive loss (train_module.contrastive_config -> null). The patch
# discrimination (JEPA) loss and low-weight register supervision are unchanged. No code
# change needed: contrastive_config is Optional and the train module no-ops when it's None.

python "$SCRIPT" launch "regbtl_base10k_scale0.25_gdyn_d768_il_noic" "$CLUSTER" \
    $LAUNCH_ARGS $WANDB_PROJECT $ROPE \
    --model.encoder_config.register_grid_size=0 --model.encoder_config.register_dim=768 --model.decoder_config.register_dim=768 \
    --model.encoder_config.register_interleave=true \
    --train_module.contrastive_config=null
