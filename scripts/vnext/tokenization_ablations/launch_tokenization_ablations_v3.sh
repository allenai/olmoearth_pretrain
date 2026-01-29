#!/bin/bash
# Launch script for tokenization ablation experiments (_3 with OOM fixes)
#
# This script launches 2 experiments testing different Sentinel-2 tokenization strategies:
# 1. Single-band: Each band as its own token (12 tokens)
# 2. Spectral grouping: Bands grouped by spectral similarity (5 tokens)
#
# Note: all_bands_single_token already has a running version
#
# OOM fixes applied:
# - flash_attn=true (encoder + decoder)
# - global_batch_size=256 (was 512)
# - rank_microbatch_size=8 (was 32)

set -e  # Exit on error

CLUSTERS='[ai2/jupiter,ai2/ceres]'
NUM_GPUS=8
PRIORITY=high
WANDB_PROJECT=2026_01_13_tokenization_ablations

echo "=================================="
echo "Launching Tokenization Ablations (_3 with OOM fixes)"
echo "Clusters: ${CLUSTERS}"
echo "GPUs: ${NUM_GPUS}"
echo "Priority: ${PRIORITY}"
echo "W&B Project: ${WANDB_PROJECT}"
echo ""
echo "OOM fixes:"
echo "  - flash_attn=true"
echo "  - global_batch_size=256"
echo "  - rank_microbatch_size=8"
echo "=================================="
echo ""

# Experiment 1: Single-band tokenization
echo "Launching experiment 1/2: Single-band tokenization (12 tokens)..."
python3 scripts/vnext/tokenization_ablations/base_single_band_s2.py launch base_single_band_s2_3 ai2/jupiter \
  --launch.num_gpus=${NUM_GPUS} \
  --launch.clusters="${CLUSTERS}" \
  --launch.priority=${PRIORITY} \
  --trainer.callbacks.wandb.project=${WANDB_PROJECT} \
  --model.encoder_config.use_flash_attn=true \
  --model.decoder_config.use_flash_attn=true \
  --data_loader.global_batch_size=256 \
  --train_module.rank_microbatch_size=8
echo "✓ Experiment 1 launched"
echo ""

# Experiment 2: Spectral grouping tokenization
echo "Launching experiment 2/2: Spectral grouping tokenization (5 tokens)..."
python3 scripts/vnext/tokenization_ablations/base_spectral_grouping.py launch base_spectral_grouping_3 ai2/jupiter \
  --launch.num_gpus=${NUM_GPUS} \
  --launch.clusters="${CLUSTERS}" \
  --launch.priority=${PRIORITY} \
  --trainer.callbacks.wandb.project=${WANDB_PROJECT} \
  --model.encoder_config.use_flash_attn=true \
  --model.decoder_config.use_flash_attn=true \
  --data_loader.global_batch_size=256 \
  --train_module.rank_microbatch_size=8
echo "✓ Experiment 2 launched"
echo ""

echo "=================================="
echo "Tokenization ablation experiments launched!"
echo ""
echo "Experiments:"
echo "  - base_single_band_s2_3:      12 tokens (each band separate)"
echo "  - base_spectral_grouping_3:   5 tokens (spectral similarity)"
echo ""
echo "Note: base_all_bands_single_token already running separately"
echo "Compare against baseline: base.py (3 tokens, resolution-based)"
echo "=================================="
