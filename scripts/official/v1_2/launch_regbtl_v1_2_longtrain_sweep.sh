#!/bin/bash
# v1.2 register-bottleneck LONGER-TRAINING test.
#
# Hypothesis: the fixed 300-epoch schedule (Duration.epochs(300) in base.py) has
# been undertraining these models. This relaunches the four regsup frontier configs
# from scratch at a longer horizon and compares against the in-flight 300-epoch runs
# (which serve as the baseline -- do NOT relaunch those).
#
# Why this is confound-free: trainer.max_steps (derived from max_duration) is passed
# straight to CosWithWarmup as t_max (olmo_core scheduler.py:71), and base.py sets no
# explicit t_max. So bumping max_duration automatically stretches the cosine decay to
# the new end -- same recipe, longer schedule, same peak LR (1e-4) and warmup (8000).
#
# Everything else is identical to launch_regbtl_v1_2_sweep.sh. Peak LR is held fixed;
# a limited peak-LR follow-up (only if 2x wins) belongs in a separate script.
set -e

PROJECT="2026_07_02_perceiver"
CLUSTER="ai2/jupiter"
LAUNCH_ARGS="--launch.num_gpus=8 --launch.priority=urgent --launch.clusters=[ai2/jupiter,ai2/ceres]"

# Longer horizon. Baseline is 300 (in-flight). 600 = 2x (decisive yes/no on
# undertraining); set to 450 for a cheaper 1.5x probe.
MAX_EPOCHS=600
SUFFIX="ep${MAX_EPOCHS}"

D768_REGSUP="scripts/official/v1_2/regbtl_v1_2_gdyn_d768_il_pdproj_noic_lsa_regsup.py"
D768_REGSUP_LATLON="scripts/official/v1_2/regbtl_v1_2_gdyn_d768_il_pdproj_noic_lsa_regsup_latlon.py"
D128_REGSUP="scripts/official/v1_2/regbtl_v1_2_gdyn_d128_il_pdproj_noic_lsa_wideread_regsup.py"
D128_REGSUP_LATLON="scripts/official/v1_2/regbtl_v1_2_gdyn_d128_il_pdproj_noic_lsa_wideread_regsup_latlon.py"

python "$D768_REGSUP" launch "regbtl_v1_2_gdyn_d768_regsup_${SUFFIX}" "$CLUSTER" \
    $LAUNCH_ARGS \
    --trainer.max_duration.value=$MAX_EPOCHS \
    --trainer.callbacks.wandb.project="$PROJECT"

python "$D768_REGSUP_LATLON" launch "regbtl_v1_2_gdyn_d768_regsup_latlon_${SUFFIX}" "$CLUSTER" \
    $LAUNCH_ARGS \
    --trainer.max_duration.value=$MAX_EPOCHS \
    --trainer.callbacks.wandb.project="$PROJECT"

python "$D128_REGSUP" launch "regbtl_v1_2_gdyn_d128_wideread_regsup_${SUFFIX}" "$CLUSTER" \
    $LAUNCH_ARGS \
    --trainer.max_duration.value=$MAX_EPOCHS \
    --trainer.callbacks.wandb.project="$PROJECT"

python "$D128_REGSUP_LATLON" launch "regbtl_v1_2_gdyn_d128_wideread_regsup_latlon_${SUFFIX}" "$CLUSTER" \
    $LAUNCH_ARGS \
    --trainer.max_duration.value=$MAX_EPOCHS \
    --trainer.callbacks.wandb.project="$PROJECT"
