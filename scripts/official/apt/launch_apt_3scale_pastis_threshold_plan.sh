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
# Best matches from latest sweep:
#   - 50% reduction target: thresholds [0.8986719251, 1.1866975069]
#       realized ratio=0.509447 (49.06% reduction)
#   - 75% reduction target: thresholds [0.7465922236, 2.0133407116]
#       realized ratio=0.244595 (75.54% reduction)
#   - 25% reduction target: not found in sweep; use a guessed threshold pair
#       intended to increase token count vs the 50% setting.
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
THRESH_T50='[0.8986719250679016,1.1866975069046022]'         # realized ratio 0.509447
THRESH_T75='[0.7465922236442566,2.0133407115936284]'         # realized ratio 0.244595
THRESH_T25_GUESS='[0.6,0.9]'                                 # guessed; expect higher token ratio

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
# P3: guessed low-compression run (aiming toward 25% reduction)

launch_variant() {
  local threshold_tag="$1"      # t50 / t75 / t25guess
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
run_threshold_block "t25guess" "$THRESH_T25_GUESS"

echo ""
echo "=============================================="
echo "All runs launched in priority order."
echo "Note: t25guess is heuristic. If ratio is still too low/high, run a focused"
echo "explicit-threshold sweep around [0.6,0.9] to calibrate toward 0.75."
echo "=============================================="
