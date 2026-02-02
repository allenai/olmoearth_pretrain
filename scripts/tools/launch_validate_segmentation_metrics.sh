#!/bin/bash
# Launch LP and Finetune evals on PASTIS to validate segmentation metrics
#
# Usage:
#   ./scripts/tools/launch_validate_segmentation_metrics.sh
#
# This launches a single Beaker job that runs both:
#   1. Linear Probe on pastis (pastis_lp)
#   2. Finetune on pastis (pastis_ft)

set -e

CLUSTER="${CLUSTER:-ai2/saturn-cirrascale}"
SCRIPT_PATH="scripts/tools/validate_segmentation_metrics.py"

echo "Launching segmentation metrics validation on $CLUSTER..."
echo "Tasks: pastis_lp (Linear Probe), pastis_ft (Finetune)"
echo ""

python "$SCRIPT_PATH" launch_evaluate validate_seg_metrics_with_class "$CLUSTER"

echo ""
echo "Done! Check wandb project: 2025_01_30_validate_seg_metrics"
