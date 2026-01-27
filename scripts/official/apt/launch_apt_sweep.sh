#!/bin/bash
# =============================================================================
# APT Evaluation Sweep Script
# =============================================================================
# This script launches experiments to compare:
#   1. Baseline (no APT) at patch sizes 4 and 8
#   2. APT with different entropy thresholds
#
# Datasets: EuroSAT, MADOS, SO2Sat, BigEarthNet
#
# Usage:
#   ./scripts/official/apt/launch_apt_sweep.sh [local|beaker] [checkpoint_path]
#
# Examples:
#   # Run locally
#   ./scripts/official/apt/launch_apt_sweep.sh local /weka/dfive-default/helios/checkpoints/joer/tiny_lr0.0002_wd0.02/step360000
#
#   # Launch on Beaker
#   ./scripts/official/apt/launch_apt_sweep.sh beaker /weka/dfive-default/helios/checkpoints/joer/tiny_lr0.0002_wd0.02/step360000
# =============================================================================

set -e

# Configuration
MODE="${1:-local}"  # "local" or "beaker"
CHECKPOINT="${2:-/weka/dfive-default/helios/checkpoints/joer/tiny_lr0.0002_wd0.02/step360000}"
CLUSTER="ai2/saturn-cirrascale"

# APT thresholds to test (for 2-scale: base 4px, max 8px)
APT_THRESHOLDS=("0.25" "0.5" "0.8" "1.2")

# Patch sizes for baseline (no APT)
PATCH_SIZES=("4" "8")

# Datasets and their task names
declare -A DATASETS_NO_APT=(
    ["eurosat"]="m_eurosat_finetune"
    ["mados"]="mados_finetune"
    ["so2sat"]="so2sat_finetune"
    ["bigearthnet"]="bigearthnet_finetune"
)

declare -A DATASETS_APT=(
    ["eurosat"]="m_eurosat_finetune_apt"
    ["mados"]="mados_finetune_apt"
    ["so2sat"]="so2sat_finetune_apt"
    ["bigearthnet"]="bigearthnet_finetune_apt"
)

# Script directory
SCRIPT_DIR="scripts/official/apt"

# Command based on mode
if [ "$MODE" == "beaker" ]; then
    CMD="launch"
    CLUSTER_ARG="$CLUSTER"
else
    CMD="evaluate"
    CLUSTER_ARG="local"
fi

echo "=============================================="
echo "APT Evaluation Sweep"
echo "=============================================="
echo "Mode: $MODE"
echo "Checkpoint: $CHECKPOINT"
echo "Cluster: $CLUSTER_ARG"
echo "=============================================="
echo ""

# -----------------------------------------------------------------------------
# PART 1: Baseline experiments (no APT) at different patch sizes
# -----------------------------------------------------------------------------
echo "=============================================="
echo "PART 1: Baseline (No APT) Experiments"
echo "=============================================="

for dataset in eurosat mados so2sat bigearthnet; do
    task_name="${DATASETS_NO_APT[$dataset]}"

    for patch_size in "${PATCH_SIZES[@]}"; do
        run_name="${dataset}_baseline_p${patch_size}"
        script="${SCRIPT_DIR}/${dataset}_eval_tiny.py"

        echo ""
        echo ">>> Launching: $run_name"
        echo "    Dataset: $dataset, Patch Size: $patch_size"

        python "$script" $CMD \
            "$run_name" "$CLUSTER_ARG" \
            --trainer.load_path="$CHECKPOINT" \
            --trainer.callbacks.downstream_evaluator.tasks.${task_name}.patch_size=$patch_size
    done
done

# -----------------------------------------------------------------------------
# PART 2: APT experiments with different thresholds
# -----------------------------------------------------------------------------
echo ""
echo "=============================================="
echo "PART 2: APT Experiments with Threshold Sweep"
echo "=============================================="

for dataset in eurosat mados so2sat bigearthnet; do
    task_name="${DATASETS_APT[$dataset]}"

    for threshold in "${APT_THRESHOLDS[@]}"; do
        run_name="${dataset}_apt_t${threshold}"
        script="${SCRIPT_DIR}/apt_${dataset}_eval_tiny.py"

        echo ""
        echo ">>> Launching: $run_name"
        echo "    Dataset: $dataset, APT Threshold: $threshold"

        python "$script" $CMD \
            "$run_name" "$CLUSTER_ARG" \
            --trainer.load_path="$CHECKPOINT" \
            --trainer.callbacks.downstream_evaluator.tasks.${task_name}.apt_config.partitioner.thresholds="[$threshold]"
    done
done

echo ""
echo "=============================================="
echo "All experiments launched!"
echo "=============================================="
echo ""
echo "Summary:"
echo "  - Baseline experiments: ${#DATASETS_NO_APT[@]} datasets x ${#PATCH_SIZES[@]} patch sizes = $((${#DATASETS_NO_APT[@]} * ${#PATCH_SIZES[@]})) runs"
echo "  - APT experiments: ${#DATASETS_APT[@]} datasets x ${#APT_THRESHOLDS[@]} thresholds = $((${#DATASETS_APT[@]} * ${#APT_THRESHOLDS[@]})) runs"
echo "  - Total: $((${#DATASETS_NO_APT[@]} * ${#PATCH_SIZES[@]} + ${#DATASETS_APT[@]} * ${#APT_THRESHOLDS[@]})) runs"
