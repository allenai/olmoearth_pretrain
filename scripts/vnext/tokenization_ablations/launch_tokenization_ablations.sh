#!/bin/bash
# Launch script for tokenization ablation experiments
#
# This script launches 3 experiments testing different Sentinel-2 tokenization strategies:
# 1. Single-band: Each band as its own token (12 tokens)
# 2. Spectral grouping: Bands grouped by spectral similarity (5 tokens)
# 3. All-bands single token: All 12 bands in one token (1 token)
#
# Baseline comparison: base.py (3 resolution-based tokens)

set -e  # Exit on error

CLUSTERS='[ai2/jupiter,ai2/ceres]'
NUM_GPUS=8
PRIORITY=high
WANDB_PROJECT=2026_01_13_tokenization_ablations

echo "=================================="
echo "Launching Tokenization Ablations"
echo "Clusters: ${CLUSTERS}"
echo "GPUs: ${NUM_GPUS}"
echo "Priority: ${PRIORITY}"
echo "W&B Project: ${WANDB_PROJECT}"
echo "=================================="
echo ""

# Experiment 1: Single-band tokenization
echo "Launching experiment 1/3: Single-band tokenization (12 tokens)..."
python3 scripts/vnext/tokenization_ablations/base_single_band_s2.py launch base_single_band_s2_2 ai2/jupiter \
  --launch.num_gpus=${NUM_GPUS} \
  --launch.clusters="${CLUSTERS}" \
  --launch.priority=${PRIORITY} \
  --trainer.callbacks.wandb.project=${WANDB_PROJECT}
echo "✓ Experiment 1 launched"
echo ""

# Experiment 2: Spectral grouping tokenization
echo "Launching experiment 2/3: Spectral grouping tokenization (5 tokens)..."
python3 scripts/vnext/tokenization_ablations/base_spectral_grouping.py launch base_spectral_grouping_2 ai2/jupiter \
  --launch.num_gpus=${NUM_GPUS} \
  --launch.clusters="${CLUSTERS}" \
  --launch.priority=${PRIORITY} \
  --trainer.callbacks.wandb.project=${WANDB_PROJECT}
echo "✓ Experiment 2 launched"
echo ""

# Experiment 3: All bands in single token
echo "Launching experiment 3/3: All bands in single token (1 token)..."
python3 scripts/vnext/tokenization_ablations/base_all_bands_single_token.py launch base_all_bands_single_token_2 ai2/jupiter \
  --launch.num_gpus=${NUM_GPUS} \
  --launch.clusters="${CLUSTERS}" \
  --launch.priority=${PRIORITY} \
  --trainer.callbacks.wandb.project=${WANDB_PROJECT}
echo "✓ Experiment 3 launched"
echo ""

echo "=================================="
echo "All tokenization ablation experiments launched successfully!"
echo ""
echo "Experiments:"
echo "  - base_single_band_s2_2:      12 tokens (each band separate)"
echo "  - base_spectral_grouping_2:   5 tokens (spectral similarity)"
echo "  - base_all_bands_single_token_2: 1 token (all bands combined)"
echo ""
echo "Compare against baseline: base.py (3 tokens, resolution-based)"
echo "=================================="
