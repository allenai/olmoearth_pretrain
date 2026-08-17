#!/bin/bash
# Simple-temporal-encoding arm on the d128 wideread regsup+NDVI tanchor psuniform
# run: the frozen month table is replaced by a learned 2-layer MLP of the minimal
# 4-number temporal signal [frac_year (years since 2020), sin/cos annual phase,
# year_valid] on both encoder and decoder, with per-sample year dropout 0.5
# (frac_year + year_valid zeroed, phase kept). Single-change arm; A/B partner:
# regbtl_v1_2_gdyn_d128_wideread_regsup_ndvi_w0p1_tanchor_newsamp_psuniform.
set -e

PROJECT="2026_07_02_perceiver"
CLUSTER="ai2/jupiter"
LAUNCH_ARGS="--launch.num_gpus=8 --launch.priority=urgent --launch.clusters=[ai2/jupiter,ai2/ceres]"

# Defragment the CUDA allocator (see launch_regbtl_v1_2_newsampling.sh).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Launch from THIS worktree's package and venv: without PYTHONPATH=. python
# resolves olmoearth_pretrain from the editable install at the main repo
# (see experimentor runbooks/pretrain-launch.md step 4), and the repo-pinned
# beaker-py lives in the worktree venv.
cd "$(dirname "$0")/../../.."
export PYTHONPATH=.
PYTHON=".venv/bin/python"

NAME="regbtl_v1_2_gdyn_d128_wideread_regsup_ndvi_w0p1_tanchor_newsamp_psuniform_simpletemporal"
SCRIPT="scripts/official/v1_2/regbtl_v1_2_gdyn_d128_wideread_regsup_ndvi_w0p1_tanchor_newsampling_psuniform_simpletemporal.py"
"$PYTHON" "$SCRIPT" launch "$NAME" "$CLUSTER" \
    $LAUNCH_ARGS \
    --trainer.callbacks.wandb.project="$PROJECT"
