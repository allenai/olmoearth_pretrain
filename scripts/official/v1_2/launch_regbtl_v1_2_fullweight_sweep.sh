#!/bin/bash
# v1.2 register-bottleneck HIGHER-SUPERVISION-WEIGHT test.
#
# The regsup runs use a deliberately low supervision base weight (SUPERVISION_WEIGHT
# =0.01 in regbtl_v1_2_regsup_common.py) -- an inductive "nudge", not a competing
# learning signal. This sweep raises that base weight to test whether the nudge was
# undertuned, at two settings (per-task balancing TASK_TYPE_WEIGHTS unchanged):
#   w0p1  -- 10x  (0.01 -> 0.1): effective regression 0.1, classification/BCE 0.01
#   w1    -- 100x (0.01 -> 1.0): effective regression 1.0, classification/BCE 0.1
#
# Trains at the base.py default 300 epochs (NO max_duration override) -- this is the
# weight axis, held at the standard horizon so it compares directly against the in-flight
# 300-epoch regsup runs. The duration axis lives in launch_regbtl_v1_2_longtrain_sweep.sh.
set -e

PROJECT="2026_07_02_perceiver"
CLUSTER="ai2/jupiter"
LAUNCH_ARGS="--launch.num_gpus=8 --launch.priority=urgent --launch.clusters=[ai2/jupiter,ai2/ceres]"

# --- w0p1: base weight 0.1 (10x) ---
D768_REGSUP_W0P1="scripts/official/v1_2/regbtl_v1_2_gdyn_d768_il_pdproj_noic_lsa_regsup_w0p1.py"
D768_REGSUP_LATLON_W0P1="scripts/official/v1_2/regbtl_v1_2_gdyn_d768_il_pdproj_noic_lsa_regsup_latlon_w0p1.py"
D128_REGSUP_W0P1="scripts/official/v1_2/regbtl_v1_2_gdyn_d128_il_pdproj_noic_lsa_wideread_regsup_w0p1.py"
D128_REGSUP_LATLON_W0P1="scripts/official/v1_2/regbtl_v1_2_gdyn_d128_il_pdproj_noic_lsa_wideread_regsup_latlon_w0p1.py"
# --- w1: base weight 1.0 (100x) ---
D768_REGSUP_W1="scripts/official/v1_2/regbtl_v1_2_gdyn_d768_il_pdproj_noic_lsa_regsup_w1.py"
D768_REGSUP_LATLON_W1="scripts/official/v1_2/regbtl_v1_2_gdyn_d768_il_pdproj_noic_lsa_regsup_latlon_w1.py"
D128_REGSUP_W1="scripts/official/v1_2/regbtl_v1_2_gdyn_d128_il_pdproj_noic_lsa_wideread_regsup_w1.py"
D128_REGSUP_LATLON_W1="scripts/official/v1_2/regbtl_v1_2_gdyn_d128_il_pdproj_noic_lsa_wideread_regsup_latlon_w1.py"

python "$D768_REGSUP_W0P1" launch "regbtl_v1_2_gdyn_d768_regsup_w0p1" "$CLUSTER" \
    $LAUNCH_ARGS \
    --trainer.callbacks.wandb.project="$PROJECT"

python "$D768_REGSUP_LATLON_W0P1" launch "regbtl_v1_2_gdyn_d768_regsup_latlon_w0p1" "$CLUSTER" \
    $LAUNCH_ARGS \
    --trainer.callbacks.wandb.project="$PROJECT"

python "$D128_REGSUP_W0P1" launch "regbtl_v1_2_gdyn_d128_wideread_regsup_w0p1" "$CLUSTER" \
    $LAUNCH_ARGS \
    --trainer.callbacks.wandb.project="$PROJECT"

python "$D128_REGSUP_LATLON_W0P1" launch "regbtl_v1_2_gdyn_d128_wideread_regsup_latlon_w0p1" "$CLUSTER" \
    $LAUNCH_ARGS \
    --trainer.callbacks.wandb.project="$PROJECT"

python "$D768_REGSUP_W1" launch "regbtl_v1_2_gdyn_d768_regsup_w1" "$CLUSTER" \
    $LAUNCH_ARGS \
    --trainer.callbacks.wandb.project="$PROJECT"

python "$D768_REGSUP_LATLON_W1" launch "regbtl_v1_2_gdyn_d768_regsup_latlon_w1" "$CLUSTER" \
    $LAUNCH_ARGS \
    --trainer.callbacks.wandb.project="$PROJECT"

python "$D128_REGSUP_W1" launch "regbtl_v1_2_gdyn_d128_wideread_regsup_w1" "$CLUSTER" \
    $LAUNCH_ARGS \
    --trainer.callbacks.wandb.project="$PROJECT"

python "$D128_REGSUP_LATLON_W1" launch "regbtl_v1_2_gdyn_d128_wideread_regsup_latlon_w1" "$CLUSTER" \
    $LAUNCH_ARGS \
    --trainer.callbacks.wandb.project="$PROJECT"
