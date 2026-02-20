#!/bin/bash
# =============================================================================
# 4-Scale APT threshold sweep — targeting sub-1 token_ratio_vs_p4
# =============================================================================
# Thresholds: [2px→1px, 4px→2px, 8px→4px]
# Higher = keep coarse (fewer tokens). thresholds[2] is the most impactful.
# thresholds[0] is kept high so 1px is reserved for truly complex regions.
#
# Usage:
#   ./scripts/official/apt/launch_apt_4scale_sweep.sh [local|beaker] [checkpoint_path]
# =============================================================================

set -e

MODE="${1:-local}"
CHECKPOINT="${2:-/weka/dfive-default/helios/checkpoints/joer/tiny_lr0.0002_wd0.02/step360000}"
CLUSTER="ai2/saturn-cirrascale"
CLUSTERS_OVERRIDE='--launch.clusters=["ai2/titan","ai2/saturn-cirrascale","ai2/jupiter"]'

# Threshold combos: [2→1, 4→2, 8→4]
# All keep thresholds[0] high (rare 1px), vary coarser splits
THRESHOLD_NAMES=("mostly8"       "mix84"         "balanced"      "fine_detail")
THRESHOLD_VALUES=("3.0,2.5,1.5"  "3.0,2.0,1.0"  "3.0,2.0,0.6"  "2.0,1.5,0.5")
# Expected ratio:   ~0.3           ~0.5            ~0.7            ~0.9

declare -A DATASETS=(
    ["eurosat"]="m_eurosat_finetune"
    ["mados"]="mados_finetune"
    ["so2sat"]="m_so2sat_finetune"
    ["bigearthnet"]="m_bigearthnet_finetune"
)

SCRIPT_DIR="scripts/official/apt"

if [ "$MODE" == "beaker" ]; then
    CMD="launch"
    CLUSTER_ARG="$CLUSTER"
    EXTRA_ARGS="$CLUSTERS_OVERRIDE"
else
    CMD="evaluate"
    CLUSTER_ARG="local"
    EXTRA_ARGS=""
fi

echo "=============================================="
echo "4-Scale APT Sweep — sub-1 token ratio targets"
echo "Mode: $MODE"
echo "=============================================="

for dataset in eurosat mados so2sat bigearthnet; do
    task_name="${DATASETS[$dataset]}"

    for i in "${!THRESHOLD_NAMES[@]}"; do
        thresh_name="${THRESHOLD_NAMES[$i]}"
        thresh_val="${THRESHOLD_VALUES[$i]}"
        run_name="4scale_${dataset}_${thresh_name}"
        script="${SCRIPT_DIR}/apt_4scale_${dataset}_eval_tiny.py"

        echo ">>> $run_name  thresholds=[$thresh_val]"

        python "$script" $CMD \
            "$run_name" "$CLUSTER_ARG" \
            --trainer.load_path="$CHECKPOINT" \
            --trainer.callbacks.downstream_evaluator.tasks.${task_name}.apt_config.partitioner.thresholds="[$thresh_val]" \
            $EXTRA_ARGS
    done
done

echo ""
echo "Done! ${#DATASETS[@]} datasets x ${#THRESHOLD_NAMES[@]} combos = $(( ${#DATASETS[@]} * ${#THRESHOLD_NAMES[@]} )) runs"
