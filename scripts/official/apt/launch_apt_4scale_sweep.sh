#!/bin/bash
# =============================================================================
# 4-Scale APT Finetuning (patch sizes 1, 2, 4, 8) — Aggressive splitting
# =============================================================================
# Tests whether adaptive patching with more fine-grained scales can beat
# uniform patch-4 while using fewer tokens on average.
#
# Uses avg-init conv downsampling. Aggressive thresholds = more splitting
# into finer patches where the image is complex.
#
# 3 thresholds: [2->1, 4->2, 8->4]
#
# Usage:
#   ./scripts/official/apt/launch_apt_4scale_sweep.sh [local|beaker] [checkpoint_path]
# =============================================================================

set -e

MODE="${1:-local}"
CHECKPOINT="${2:-/weka/dfive-default/helios/checkpoints/joer/tiny_lr0.0002_wd0.02/step360000}"
CLUSTER="ai2/saturn-cirrascale"
CLUSTERS_OVERRIDE='--launch.clusters=["ai2/titan","ai2/saturn-cirrascale","ai2/jupiter"]'

# Aggressive: more splitting at every level
THRESHOLDS="0.3,0.5,0.8"

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
echo "4-Scale APT (1,2,4,8) — Aggressive splitting"
echo "Thresholds: [$THRESHOLDS]"
echo "Mode: $MODE"
echo "=============================================="

for dataset in eurosat mados so2sat bigearthnet; do
    task_name="${DATASETS[$dataset]}"
    run_name="4scale_${dataset}_apt_aggressive"
    script="${SCRIPT_DIR}/apt_4scale_${dataset}_eval_tiny.py"

    echo ">>> $run_name"

    python "$script" $CMD \
        "$run_name" "$CLUSTER_ARG" \
        --trainer.load_path="$CHECKPOINT" \
        --trainer.callbacks.downstream_evaluator.tasks.${task_name}.apt_config.partitioner.thresholds="[$THRESHOLDS]" \
        $EXTRA_ARGS
done

echo ""
echo "Done! 4 datasets launched."
