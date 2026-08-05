#!/bin/bash
# GPS token encoding on the d128 wideread regsup+NDVI tanchor psuniform run: the
# sample's (lat, lon) as unit-sphere (x, y, z) through a 2-layer MLP into the
# RoPE-idle spatial encoding slot, with per-sample GPS dropout at 0.5 so the
# no-GPS path (all evals) is trained. Single-change arm; A/B partner:
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

# _v2: v1 (9cc8ce19) was stopped ~2h in and relaunched at 115154fb so the
# in-loop eval jobs (which inherit the training job's GIT_REF) include the
# eval-side latlon attachment; training behavior is identical across the two.
NAME="regbtl_v1_2_gdyn_d128_wideread_regsup_ndvi_w0p1_tanchor_newsamp_psuniform_gps_v2"
SCRIPT="scripts/official/v1_2/regbtl_v1_2_gdyn_d128_wideread_regsup_ndvi_w0p1_tanchor_newsampling_psuniform_gps.py"
"$PYTHON" "$SCRIPT" launch "$NAME" "$CLUSTER" \
    $LAUNCH_ARGS \
    --trainer.callbacks.wandb.project="$PROJECT"
