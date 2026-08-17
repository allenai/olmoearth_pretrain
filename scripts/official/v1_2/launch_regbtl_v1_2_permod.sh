#!/bin/bash
# cand_ndvi + per-modality encoder trunk layers (QKV/output/MLP routed by modality).
#
#   arm      encoder trunk parameters                    script
#   permod   one QKV/proj/MLP set per supported modality ..._psuniform_permod.py
#
# A/B partner is the existing
# regbtl_v1_2_gdyn_d128_wideread_regsup_ndvi_w0p1_tanchor_newsamp_psuniform (cand_ndvi),
# which this matches knob for knob apart from the per-modality trunk routing -- same
# register bottleneck, sampler, anchor, NDVI arm, token budget, LR schedule, epochs(300),
# loss set AND the same in-loop eval catalog (add_loop_eval_beaker_job), so its curves
# overlay cand_ndvi's directly in the shared wandb project.
#
# WHY: re-test of the 2026-06 per-modality-capacity experiment
# (favyen/20260608-per-modality-layers), which was null on the pre-perceiver stack; the
# trunk's job has since changed (it now feeds a shared spatial register grid), so
# per-modality trunk capacity may land differently.
#
# NOTE: ~850M extra params (mostly per-modality MLPs; 9 modality routes at d768 x 12
# blocks). Expect slower steps and more optimizer memory than cand_ndvi. If the run
# OOMs, add --train_module.rank_microbatch_size=32 (cand_ndvi runs at 64).
set -e

PROJECT="2026_07_02_perceiver"
CLUSTER="ai2/jupiter"
LAUNCH_ARGS="--launch.num_gpus=8 --launch.priority=urgent --launch.clusters=[ai2/jupiter,ai2/ceres]"

# Defragment the CUDA allocator to avoid fragmentation OOMs at the larger token budget
# of the newsampling runs. Propagated to the Beaker job by internal/common.py.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

D=scripts/official/v1_2
STEM=regbtl_v1_2_gdyn_d128_wideread_regsup_ndvi_w0p1_tanchor

python "$D/${STEM}_newsampling_psuniform_permod.py" launch \
    "${STEM}_newsamp_psuniform_permod" \
    "$CLUSTER" \
    $LAUNCH_ARGS \
    --trainer.callbacks.wandb.project="$PROJECT"
