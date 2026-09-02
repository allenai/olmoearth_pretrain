#!/bin/bash
# New-maps (GLO30 DSM + Meta Canopy Height) register-bottleneck runs: the official v1.2
# wideread + regsup + w0p1 + newsampling + psuniform recipe (Perceiver read, faster 1fwd
# train module, decorrelated shape sampler at uniform patch sizes, register-grid
# supervision) at two register widths -- d128 (compressed storage) and d768 (full width,
# no compression). No temporal anchor, no NDVI. Only the H5 file + map modalities differ
# from the official runs; everything is baked into the scripts (not CLI overrides) so the
# Beaker eval jobs reconstruct the matching model.
set -e

PROJECT="2026_07_24_new_maps"
CLUSTER="ai2/jupiter"
LAUNCH_ARGS="--launch.num_gpus=8 --launch.priority=urgent --launch.clusters=[ai2/jupiter,ai2/ceres]"

# Defragment the CUDA allocator to avoid fragmentation OOMs at the larger (3072) token
# budget; propagated to the Beaker job by internal/common.py.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

D128="scripts/vnext/2026_07_24_new_maps/regbtl_v1_2_gdyn_d128_wideread_regsup_w0p1_newsampling_psuniform.py"
D768="scripts/vnext/2026_07_24_new_maps/regbtl_v1_2_gdyn_d768_wideread_regsup_w0p1_newsampling_psuniform.py"

python "$D128" launch "regbtl_v1_2_gdyn_d128_wideread_regsup_w0p1_newsampling_psuniform_newmaps" "$CLUSTER" \
    $LAUNCH_ARGS \
    --trainer.callbacks.wandb.project="$PROJECT"

python "$D768" launch "regbtl_v1_2_gdyn_d768_wideread_regsup_w0p1_newsampling_psuniform_newmaps" "$CLUSTER" \
    $LAUNCH_ARGS \
    --trainer.callbacks.wandb.project="$PROJECT"
