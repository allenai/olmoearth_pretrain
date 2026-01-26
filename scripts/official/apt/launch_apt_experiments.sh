#!/bin/bash
# Launch script for APT experiments
#
# This script launches APT (Adaptive Patch Transformers) experiments
# for testing content-aware adaptive patchification on remote sensing data.
#
# Usage:
#   ./scripts/official/apt/launch_apt_experiments.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Default settings
CLUSTER="${CLUSTER:-ai2/allennlp-cirrascale}"
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-apt_s2}"

echo "Launching APT experiments..."
echo "Cluster: $CLUSTER"
echo "Run name prefix: $RUN_NAME_PREFIX"

# Experiment 1: APT baseline with default thresholds (5.5, 4.0)
echo ""
echo "=== Experiment 1: APT S2-only baseline ==="
python "$REPO_ROOT/scripts/official/apt/apt_s2_finetune.py" launch \
    --run-name "${RUN_NAME_PREFIX}_baseline" \
    --cluster "$CLUSTER"

# Experiment 2: APT with more aggressive thresholds (higher reduction)
# echo ""
# echo "=== Experiment 2: APT aggressive thresholds ==="
# python "$REPO_ROOT/scripts/official/apt/apt_s2_finetune.py" launch \
#     --run-name "${RUN_NAME_PREFIX}_aggressive" \
#     --cluster "$CLUSTER" \
#     --overrides "apt.thresholds=[6.0,5.0]"

# Experiment 3: APT with conservative thresholds (lower reduction, preserve accuracy)
# echo ""
# echo "=== Experiment 3: APT conservative thresholds ==="
# python "$REPO_ROOT/scripts/official/apt/apt_s2_finetune.py" launch \
#     --run-name "${RUN_NAME_PREFIX}_conservative" \
#     --cluster "$CLUSTER" \
#     --overrides "apt.thresholds=[4.5,3.5]"

echo ""
echo "All experiments launched!"
echo "Monitor progress with: beaker job list --workspace <workspace>"
