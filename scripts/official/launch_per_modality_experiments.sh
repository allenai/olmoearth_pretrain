#!/bin/bash
# Launch script for per-modality projection experiments
# This script launches three variants:
# 1. Encoder-only per-modality projections
# 2. Decoder-only per-modality output heads
# 3. Both encoder and decoder with per-modality projections

set -e  # Exit on error

CLUSTERS='[ai2/jupiter,ai2/ceres]'
NUM_GPUS=8

echo "=================================="
echo "Launching Per-Modality Projection Experiments"
echo "Clusters: ${CLUSTERS}"
echo "GPUs: ${NUM_GPUS}"
echo "=================================="
echo ""

# 1. Encoder-only per-modality projection
echo "Launching experiment 1/3: Encoder-only per-modality projection..."
python3 scripts/official/base_encoder_per_mod_proj.py launch base_encoder_per_mod_proj ai2/jupiter \
  --launch.num_gpus=${NUM_GPUS} \
  --launch.clusters="${CLUSTERS}" --launch.priority=normal --trainer.callbacks.wandb.project=11_21_per_modality_projection_experiments
echo "✓ Encoder-only experiment launched"
echo ""

# 2. Decoder-only per-modality output heads
echo "Launching experiment 2/3: Decoder-only per-modality output heads..."
python3 scripts/official/base_decoder_per_mod_proj.py launch base_decoder_per_mod_proj ai2/jupiter \
  --launch.num_gpus=${NUM_GPUS} \
  --launch.clusters="${CLUSTERS}" --launch.priority=normal --trainer.callbacks.wandb.project=11_21_per_modality_projection_experiments
echo "✓ Decoder-only experiment launched"
echo ""

# 3. Both encoder and decoder with per-modality projections
echo "Launching experiment 3/3: Both encoder and decoder with per-modality..."
python3 scripts/official/base_both_per_mod_proj.py launch base_both_per_mod_proj ai2/jupiter \
  --launch.num_gpus=${NUM_GPUS} \
  --launch.clusters="${CLUSTERS}" --launch.priority=normal --trainer.callbacks.wandb.project=11_21_per_modality_projection_experiments
echo "✓ Both encoder+decoder experiment launched"
echo ""

echo "=================================="
echo "All 3 experiments launched successfully!"
echo "=================================="
echo ""
echo "Experiment names:"
echo "  1. base_encoder_per_mod_proj"
echo "  2. base_decoder_per_mod_proj"
echo "  3. base_both_per_mod_proj"
echo ""
echo "Check Beaker for job status"
