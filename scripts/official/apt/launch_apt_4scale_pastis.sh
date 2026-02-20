#!/bin/bash
# =============================================================================
# PASTIS: 4-Scale APT + patch-size baselines (2, 4, 8)
# =============================================================================
# Launches 4 experiments:
#   1. Baseline patch_size=2
#   2. Baseline patch_size=4
#   3. Baseline patch_size=8
#   4. 4-scale APT (1,2,4,8) with aggressive thresholds
#
# Usage:
#   ./scripts/official/apt/launch_apt_4scale_pastis.sh [local|beaker] [checkpoint_path]
# =============================================================================

set -e

MODE="${1:-local}"
CHECKPOINT="${2:-/weka/dfive-default/helios/checkpoints/joer/tiny_lr0.0002_wd0.02/step360000}"
CLUSTER="ai2/saturn-cirrascale"
CLUSTERS_OVERRIDE='--launch.clusters=["ai2/titan","ai2/saturn-cirrascale","ai2/jupiter"]'

SCRIPT_DIR="scripts/official/apt"
TASK="pastis_finetune"

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
echo "PASTIS: Baselines + 4-Scale APT"
echo "Mode: $MODE"
echo "=============================================="

# --- Baselines at patch sizes 2, 4, 8 ---
for ps in 2 4 8; do
    run_name="pastis_baseline_p${ps}"
    echo ">>> $run_name"
    python "${SCRIPT_DIR}/pastis_eval_tiny.py" $CMD \
        "$run_name" "$CLUSTER_ARG" \
        --trainer.load_path="$CHECKPOINT" \
        --trainer.callbacks.downstream_evaluator.tasks.${TASK}.patch_size=$ps \
        $EXTRA_ARGS
done

# --- 4-scale APT (aggressive thresholds) ---
THRESHOLDS="0.3,0.5,0.8"
run_name="pastis_apt_4scale_aggressive"
echo ">>> $run_name  thresholds=[$THRESHOLDS]"
python "${SCRIPT_DIR}/apt_4scale_pastis_eval_tiny.py" $CMD \
    "$run_name" "$CLUSTER_ARG" \
    --trainer.load_path="$CHECKPOINT" \
    --trainer.callbacks.downstream_evaluator.tasks.${TASK}.apt_config.partitioner.thresholds="[$THRESHOLDS]" \
    $EXTRA_ARGS

echo ""
echo "Done! 3 baselines + 1 APT = 4 runs launched."
