#!/bin/bash
# cand_ndvi + cloud-aware patch discrimination, swept over the skip threshold.
#
#   arm            drops a decode token when it is ...    scripts
#   cloudmask0p25  >25% cloud/shadow  (aggressive)        ..._psuniform_cloudmask0p25.py
#   cloudmask0p5   >50% cloud/shadow  (middle)            ..._psuniform_cloudmask0p5.py
#   cloudmask0p75  >75% cloud/shadow  (conservative)      ..._psuniform_cloudmask0p75.py
#
# A/B partner for all three is the existing
# regbtl_v1_2_gdyn_d128_wideread_regsup_ndvi_w0p1_tanchor_newsamp_psuniform (cand_ndvi),
# which these match knob for knob apart from the cloud skip -- same model, sampler,
# anchor, NDVI arm, token budget, LR schedule, epochs(300) and loss set, so the comparison
# is single-variable. (cand_ndvi uses LatentMIMTrainModuleConfig, which has no instance-
# contrastive term at all, so there was never an InfoNCE knob to set here.)
# Same wandb project as cand_ndvi so the curves overlay directly.
#
# WHY: the pretext task currently asks the model to predict the latent content of cloudy
# pixels -- the target there is weather, not ground, and is unpredictable by construction.
# The NDVI supervision head sharpens this: a vegetation index read through cloud is noise,
# so forcing per-cell temporal trajectories to fit it should actively corrupt the register
# grid. The skip covers S2, Landsat AND the S2-derived ndvi decode target.
#
# PREREQUISITE, and it fails SILENTLY: the OmniCloudMask sidecars must be complete at
#   /weka/dfive-default/helios/dataset/osm_sampling/cloud_masks_omnicloudmask/
#     cdl_gse_landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcereal_\
#     worldcover_worldpop_wri_canopy_height_map/1138828/
# A batch is cloud-masked only if EVERY sample in it is cached (see
# data/collate.py:collate_olmoearth_pretrain), so a partial cache disables the skip for
# most batches while the run looks entirely normal. Expect 1,138,828 .npz files across
# 114 shard dirs. Do not read a null result from these arms without checking that count.
set -e

PROJECT="2026_07_02_perceiver"
CLUSTER="ai2/jupiter"
LAUNCH_ARGS="--launch.num_gpus=8 --launch.priority=urgent --launch.clusters=[ai2/jupiter,ai2/ceres]"

# Defragment the CUDA allocator to avoid fragmentation OOMs at the larger token budget
# of the newsampling runs. Propagated to the Beaker job by internal/common.py.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

D=scripts/official/v1_2
STEM=regbtl_v1_2_gdyn_d128_wideread_regsup_ndvi_w0p1_tanchor

# Middle arm first: it is the one that decides whether the effect exists at all.
for ARM in 0p5 0p25 0p75; do
    python "$D/${STEM}_newsampling_psuniform_cloudmask${ARM}.py" launch \
        "${STEM}_newsamp_psuniform_cloudmask${ARM}" \
        "$CLUSTER" \
        $LAUNCH_ARGS \
        --trainer.callbacks.wandb.project="$PROJECT"
done
