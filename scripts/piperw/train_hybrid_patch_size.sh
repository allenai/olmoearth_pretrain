#!/bin/bash
# Training script for hybrid patch size experiment
# Processes at patch size 8 for speed, outputs at patch size 1 for fine-grained embeddings
#
# Usage:
#   ./train_hybrid_patch_size.sh [RUN_NAME] [CLUSTER] [NUM_GPUS] [PRIORITY]
#
# Examples:
#   # Basic usage with defaults
#   ./train_hybrid_patch_size.sh
#
#   # Custom run name and cluster
#   ./train_hybrid_patch_size.sh my_experiment ai2/ceres-cirrascale
#
#   # Full customization
#   ./train_hybrid_patch_size.sh my_experiment ai2/jupiter-cirrascale-2 16 urgent

set -e  # Exit on error

# Default values (can be overridden via command line)
RUN_NAME="${1:-nano_hybrid_ps8_to_ps1_$(date +%Y%m%d_%H%M%S)}"
CLUSTER="${2:-ai2/jupiter-cirrascale-2}"
NUM_GPUS="${3:-8}"
PRIORITY="${4:-normal}"
WANDB_PROJECT="${WANDB_PROJECT:-2025_10_02_phase2}"

echo "=================================="
echo "Hybrid Patch Size Training"
echo "=================================="
echo "Run Name: ${RUN_NAME}"
echo "Cluster: ${CLUSTER}"
echo "GPUs: ${NUM_GPUS}"
echo "Priority: ${PRIORITY}"
echo "W&B Project: ${WANDB_PROJECT}"
echo ""
echo "Configuration:"
echo "  - Processing patch size: 8 (fast training, ~64x faster attention)"
echo "  - Output patch size: 1 (fine-grained embeddings)"
echo "  - Target encoder: Always uses ps=1 for consistent targets"
echo ""
echo "Benefits:"
echo "  ✓ Fast training with ps=8 processing"
echo "  ✓ Fine-grained ps=1 embeddings for downstream tasks"
echo "  ✓ Memory efficient (only upsamples at end)"
echo "=================================="
echo ""

# Launch the training
# Note: The patch sizes are already set in script.py, but we can override here if needed
python3 scripts/piperw/nano.py launch "${RUN_NAME}" "${CLUSTER}" \
  --launch.num_gpus="${NUM_GPUS}" \
  --launch.clusters="[${CLUSTER}]" \
  --launch.priority="${PRIORITY}" \
  --trainer.callbacks.wandb.project="${WANDB_PROJECT}" \
  --train_module.processing_patch_size=8 \
  --train_module.output_patch_size=1

echo ""
echo "✓ Training job launched successfully!"
echo "  Run name: ${RUN_NAME}"
echo "  Monitor at: https://wandb.ai/${WANDB_PROJECT}/${RUN_NAME}"

