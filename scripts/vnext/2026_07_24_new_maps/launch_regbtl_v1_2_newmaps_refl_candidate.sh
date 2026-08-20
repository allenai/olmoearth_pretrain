#!/bin/bash
# The candidate recipe on the new map set (GLO30 DSM + Meta Canopy Height) and Landsat
# TOA reflectance, in three arms:
#
#   A  cand_ndvi as-is:      d128 + regsup w0p1 + NDVI arm + year_start anchor
#   B  A + the full DSM:     GLO30 elevation+slope, plus aspect as sin/cos
#   C  the shipping student: d768 teacher + detached linear [128, 64] student,
#                            sup768 w1, full DSM (no NDVI, no anchor)
#
# A's A/B partner is the in-flight regbtl_v1_2_gdyn_d128_wideread_regsup_w0p1_
# newsampling_psuniform_newmaps_landsat_refl (same everything, minus NDVI + anchor);
# B's is A. Everything is baked into the scripts rather than passed as CLI overrides
# so the Beaker eval jobs reconstruct the matching model.
#
# Run names are shortened relative to the file names: the in-loop eval callback has
# only 94 characters of train-run name before Beaker's 128-char experiment-name limit
# truncates it, and the full-convention names overrun that.
set -e

PROJECT="2026_07_24_new_maps"
CLUSTER="ai2/jupiter"
LAUNCH_ARGS="--launch.num_gpus=8 --launch.priority=urgent --launch.clusters=[ai2/jupiter,ai2/ceres]"

# Defragment the CUDA allocator to avoid fragmentation OOMs at the larger (3072) token
# budget; propagated to the Beaker job by internal/common.py.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DIR="scripts/vnext/2026_07_24_new_maps"
A="$DIR/regbtl_v1_2_gdyn_d128_wideread_regsup_ndvi_w0p1_tanchor_newsampling_psuniform_landsat_refl.py"
B="$DIR/regbtl_v1_2_gdyn_d128_wideread_regsup_ndvi_w0p1_tanchor_newsampling_psuniform_landsat_refl_dsm3.py"
C="$DIR/regbtl_v1_2_gdyn_d768_proj128lin_sup768_w1_newsampling_psuniform_landsat_refl_dsm3.py"

python "$A" launch "regbtl_v1_2_gdyn_d128_wideread_regsup_ndvi_w0p1_tanchor_psuniform_newmaps_refl" "$CLUSTER" \
    $LAUNCH_ARGS \
    --trainer.callbacks.wandb.project="$PROJECT"

python "$B" launch "regbtl_v1_2_gdyn_d128_wideread_regsup_ndvi_w0p1_tanchor_psuniform_newmaps_refl_dsm3" "$CLUSTER" \
    $LAUNCH_ARGS \
    --trainer.callbacks.wandb.project="$PROJECT"

python "$C" launch "regbtl_v1_2_gdyn_d768_proj128lin_sup768_w1_psuniform_newmaps_refl_dsm3" "$CLUSTER" \
    $LAUNCH_ARGS \
    --trainer.callbacks.wandb.project="$PROJECT"
