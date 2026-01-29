#!/bin/bash
# Launch DDP vs FSDP comparison experiments
#
# This script launches two identical training runs:
# 1. One using DDP (DistributedDataParallel)
# 2. One using FSDP (FullyShardedDataParallel)
#
# Both runs use:
# - Tiny model (192 embedding, 12 depth)
# - 4 GPUs
# - 500 training steps
# - Same random seed
# - Same data
# - Logging to wandb project: ddp_vs_fsdp_comparison
#
# Usage:
#   ./scripts/ddp_fsdp_comparison/launch_comparison.sh [CLUSTER]
#
# Default cluster: ai2/ceres-cirrascale

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLUSTER="${1:-ai2/saturn-cirrascale}"
RUN_PREFIX="ddp_fsdp_comparison_$(date +%Y%m%d_%H%M%S)"

echo "========================================"
echo "DDP vs FSDP Comparison Experiment"
echo "========================================"
echo "Cluster: $CLUSTER"
echo "Run prefix: $RUN_PREFIX"
echo ""

# Launch FSDP run
echo "[1/2] Launching FSDP run..."
python "$SCRIPT_DIR/compare_ddp_fsdp.py" launch "${RUN_PREFIX}" "$CLUSTER" \
    --dp_type=fsdp \
    --launch.num_gpus=4 \
    --launch.num_nodes=1

echo ""

# Launch DDP run
echo "[2/2] Launching DDP run..."
python "$SCRIPT_DIR/compare_ddp_fsdp.py" launch "${RUN_PREFIX}" "$CLUSTER" \
    --dp_type=ddp \
    --launch.num_gpus=4 \
    --launch.num_nodes=1

echo ""
echo "========================================"
echo "Both experiments launched!"
echo "========================================"
echo ""
echo "Monitor progress at:"
echo "  https://wandb.ai/eai-ai2/ddp_vs_fsdp_comparison"
echo ""
echo "Compare runs:"
echo "  - ${RUN_PREFIX}_fsdp"
echo "  - ${RUN_PREFIX}_ddp"
echo ""
echo "Key metrics to compare:"
echo "  - train/PatchDisc (loss curve)"
echo "  - optim/total grad norm"
echo "  - m-eurosat downstream accuracy"
echo ""
