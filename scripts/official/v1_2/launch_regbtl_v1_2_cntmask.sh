#!/bin/bash
# cand_ndvi (regbtl_v1_2_gdyn_d128_wideread_regsup_ndvi_w0p1_tanchor_newsampling_psuniform,
# W&B sb2yr2pe) with the masking strategy swapped to random_count_time_with_decode:
# Uniform{1..N} encode-bandset count, encode-side masked tokens as DECODER targets,
# contiguous-timestep time drop with one retained (randomly masked) image modality.
# Single-change A/B against the cand_ndvi baseline; everything else (model, sampler,
# schedule, evals) is inherited from the base script's builders.
set -e

PROJECT="2026_07_02_perceiver"
CLUSTER="ai2/jupiter"
LAUNCH_ARGS="--launch.num_gpus=8 --launch.priority=urgent --launch.clusters=[ai2/jupiter,ai2/ceres]"

# Defragment the CUDA allocator to avoid fragmentation OOMs at the larger token budget
# (matches the newsampling launches). Propagated to the Beaker job by internal/common.py.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CNTMASK="scripts/official/v1_2/regbtl_v1_2_gdyn_d128_wideread_regsup_ndvi_w0p1_tanchor_newsampling_psuniform_cntmask.py"

python "$CNTMASK" launch "regbtl_v1_2_gdyn_d128_wideread_regsup_ndvi_w0p1_tanchor_newsamp_psuniform_cntmask" "$CLUSTER" \
    $LAUNCH_ARGS \
    --trainer.callbacks.wandb.project="$PROJECT"
