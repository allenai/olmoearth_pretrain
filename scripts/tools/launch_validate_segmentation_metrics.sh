#!/bin/bash
# Launch LP and Finetune evals on PASTIS to validate segmentation metrics
#
# Usage:
#   ./scripts/tools/launch_validate_segmentation_metrics.sh
#
# This will launch two Beaker jobs:
#   1. Linear Probe on pastis_sentinel2
#   2. Finetune on pastis_sentinel2

set -e

CLUSTER="${CLUSTER:-ai2/saturn-cirrascale}"
SCRIPT_PATH="scripts/tools/validate_segmentation_metrics.py"

echo "Launching segmentation metrics validation on $CLUSTER..."
echo ""

# Linear Probe
echo "=== Launching Linear Probe eval ==="
python "$SCRIPT_PATH" launch_evaluate validate_seg_metrics_lp "$CLUSTER"

echo ""

# Finetune
echo "=== Launching Finetune eval ==="
FINETUNE=1 python "$SCRIPT_PATH" launch_evaluate validate_seg_metrics_ft "$CLUSTER"

echo ""
echo "Done! Check wandb project: 2025_01_30_validate_seg_metrics"
