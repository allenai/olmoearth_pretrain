#!/bin/bash
# Gram-matching pretext loss arm on the d128 wideread regsup+NDVI tanchor psuniform
# run: patch-discrimination InfoNCE replaced by within-sample cosine-Gram MSE against
# the frozen random-projection targets (relational/RKD matching; no temperature, no
# same-target masking). Single-change arm; A/B partner:
# regbtl_v1_2_gdyn_d128_wideread_regsup_ndvi_w0p1_tanchor_newsamp_psuniform.
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

NAME="regbtl_v1_2_gdyn_d128_wideread_regsup_ndvi_w0p1_tanchor_newsamp_psuniform_gramloss_w30"
SCRIPT="scripts/official/v1_2/regbtl_v1_2_gdyn_d128_wideread_regsup_ndvi_w0p1_tanchor_newsampling_psuniform_gramloss_w30.py"
"$PYTHON" "$SCRIPT" launch "$NAME" "$CLUSTER" \
    $LAUNCH_ARGS \
    --trainer.callbacks.wandb.project="$PROJECT"
