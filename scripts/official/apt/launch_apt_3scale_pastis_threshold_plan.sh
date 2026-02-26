#!/bin/bash
# =============================================================================
# 3-scale PASTIS APT recovery sweep (prioritized)
# =============================================================================
# Builds experiment runs from threshold sweep outputs and launches them in
# priority order.
#
# Target token reductions requested:
#   - 25% reduction  => token ratio 0.75
#   - 50% reduction  => token ratio 0.50
#   - 75% reduction  => token ratio 0.25
#
# Best matches from current sweep:
#   - closest to 25% reduction: thresholds [0.7617737651, 1.2109026194]
#       realized ratio=0.5531 (44.69% reduction)  <-- no near-0.75 candidate yet
#   - closest to 50% reduction: thresholds [0.9102376699, 1.2109026194]
#       realized ratio=0.5101 (48.99% reduction)
#   - closest to 75% reduction: thresholds [0.7617737651, 2.0439155579]
#       realized ratio=0.2446 (75.54% reduction)
#
# Variants per threshold:
#   A) avg init, no freeze
#   B) conv_init=zero + add_resize_residual=true, no freeze
#   C) conv_init=zero + add_resize_residual=true, freeze 0.2, but keep
#      conv_downsample trainable during freeze
#
# Usage:
#   bash scripts/official/apt/launch_apt_3scale_pastis_threshold_plan.sh [local|beaker] [checkpoint_path]
#
# Examples:
#   bash scripts/official/apt/launch_apt_3scale_pastis_threshold_plan.sh local
#   bash scripts/official/apt/launch_apt_3scale_pastis_threshold_plan.sh beaker /path/to/checkpoint
# =============================================================================

set -euo pipefail

MODE="${1:-local}"
CHECKPOINT="${2:-/weka/dfive-default/helios/checkpoints/joer/tiny_lr0.0002_wd0.02/step360000}"
SCRIPT="scripts/official/apt/apt_3scale_pastis_eval_tiny.py"
TASK="pastis_finetune"

CLUSTER="ai2/saturn-cirrascale"
CLUSTERS_OVERRIDE='--launch.clusters=["ai2/titan","ai2/saturn-cirrascale","ai2/jupiter"]'
WANDB_PROJECT="01_2026_apt_investigation"

if [ "$MODE" == "beaker" ]; then
  CMD="launch"
  CLUSTER_ARG="$CLUSTER"
  EXTRA_ARGS="$CLUSTERS_OVERRIDE"
else
  CMD="evaluate"
  CLUSTER_ARG="local"
  EXTRA_ARGS=""
fi

# -------------------------
# Threshold presets
# -------------------------
THRESH_T25_CLOSE='[0.7617737650871277,1.2109026193618775]'   # realized ratio 0.5531
THRESH_T50='[0.9102376699447632,1.2109026193618775]'         # realized ratio 0.5101
THRESH_T75='[0.7617737650871277,2.0439155578613284]'         # realized ratio 0.2446

echo "=============================================="
echo "3-scale PASTIS APT prioritized launch plan"
echo "Mode: $MODE"
echo "Checkpoint: $CHECKPOINT"
echo "=============================================="

# -------------------------
# Priority order
# -------------------------
# P1: balanced compute regime (near 50% reduction)
# P2: aggressive compression (near 75% reduction)
# P3: low compression request (closest available to 25% reduction target)

launch_variant() {
  local threshold_tag="$1"      # t50 / t75 / t25close
  local thresholds="$2"         # json-like list string
  local variant_tag="$3"        # avg_nf / zero_res_nf / zero_res_frz
  local conv_init="$4"          # average / zero
  local add_resize="$5"         # true / false
  local freeze_frac="$6"        # 0.0 / 0.2
  local keep_conv_train="$7"    # true / false

  local run_name="pastis3_${threshold_tag}_${variant_tag}"
  echo ""
  echo ">>> Launching ${run_name}"
  python "$SCRIPT" "$CMD" \
    "$run_name" "$CLUSTER_ARG" \
    --trainer.load_path="$CHECKPOINT" \
    --trainer.callbacks.wandb.project="$WANDB_PROJECT" \
    --trainer.callbacks.downstream_evaluator.tasks.${TASK}.apt_config.partitioner.thresholds="$thresholds" \
    --trainer.callbacks.downstream_evaluator.tasks.${TASK}.apt_config.embed.conv_init="$conv_init" \
    --trainer.callbacks.downstream_evaluator.tasks.${TASK}.apt_config.embed.add_resize_residual="$add_resize" \
    --trainer.callbacks.downstream_evaluator.tasks.${TASK}.freeze_epoch_fraction="$freeze_frac" \
    --trainer.callbacks.downstream_evaluator.tasks.${TASK}.train_apt_conv_downsample_during_freeze="$keep_conv_train" \
    $EXTRA_ARGS
}

run_threshold_block() {
  local threshold_tag="$1"
  local thresholds="$2"
  echo ""
  echo "------------------------------------------------"
  echo "Threshold preset: ${threshold_tag}  ${thresholds}"
  echo "------------------------------------------------"

  # A) avg init + no freeze
  launch_variant "$threshold_tag" "$thresholds" "avg_nf" "average" "false" "0.0" "true"
  # B) zero init + resize residual + no freeze
  launch_variant "$threshold_tag" "$thresholds" "zero_res_nf" "zero" "true" "0.0" "true"
  # C) zero init + resize residual + freeze/unfreeze, conv trainable during freeze
  launch_variant "$threshold_tag" "$thresholds" "zero_res_frz" "zero" "true" "0.2" "true"
}

# Priority 1
run_threshold_block "t50" "$THRESH_T50"
# Priority 2
run_threshold_block "t75" "$THRESH_T75"
# Priority 3
run_threshold_block "t25close" "$THRESH_T25_CLOSE"

echo ""
echo "=============================================="
echo "All runs launched in priority order."
echo "Note: t25close is not truly 25% reduction; rerun threshold sweep with"
echo "lower thresholds if you need token ratio closer to 0.75."
echo "=============================================="
