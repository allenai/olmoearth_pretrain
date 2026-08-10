#!/bin/bash
# Combined GPS + simple-temporal arm on the d128 wideread regsup+NDVI tanchor
# psuniform run: both learned metadata encodings stacked (GPS xyz MLP in the
# RoPE-idle spatial slot with dropout 0.5; month table replaced by the 4-number
# temporal MLP with year dropout 0.5 on encoder+decoder). Completes the 2x2 over
# {GPS, simple-temporal} with the base run and the two single-change arms.
set -e

PROJECT="2026_07_02_perceiver"
CLUSTER="ai2/jupiter"
LAUNCH_ARGS="--launch.num_gpus=8 --launch.priority=urgent --launch.clusters=[ai2/jupiter,ai2/ceres]"

# Defragment the CUDA allocator (see launch_regbtl_v1_2_newsampling.sh).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Launch from THIS worktree's package and venv (see launch_regbtl_v1_2_gps.sh).
cd "$(dirname "$0")/../../.."
export PYTHONPATH=.
PYTHON=".venv/bin/python"

NAME="regbtl_v1_2_gdyn_d128_wideread_regsup_ndvi_w0p1_tanchor_newsamp_psuniform_gps_simpletemporal"
SCRIPT="scripts/official/v1_2/regbtl_v1_2_gdyn_d128_wideread_regsup_ndvi_w0p1_tanchor_newsampling_psuniform_gps_simpletemporal.py"
"$PYTHON" "$SCRIPT" launch "$NAME" "$CLUSTER" \
    $LAUNCH_ARGS \
    --trainer.callbacks.wandb.project="$PROJECT"
