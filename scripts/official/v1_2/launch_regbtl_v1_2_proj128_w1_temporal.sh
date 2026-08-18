#!/bin/bash
# The 3 temporal arms on the pcv_sup768_w1 detached-student run: time-conditioned
# NDVI supervision, the anchored register read (tanchor), and both combined. Both
# arms measured ~null at d128/w0p1; these re-test them on the d768/w1 teacher (and,
# for tanchor, on the pcv student, which mirrors the primary's anchor). A/B partner
# for all three: regbtl_v1_2_gdyn_d768_proj128pcv_sup768_w1_newsamp_psuniform.
set -e

PROJECT="2026_07_02_perceiver"
CLUSTER="ai2/jupiter"
LAUNCH_ARGS="--launch.num_gpus=8 --launch.priority=urgent --launch.clusters=[ai2/jupiter,ai2/ceres]"

# Defragment the CUDA allocator (see launch_regbtl_v1_2_newsampling.sh).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for ARM in "sup768_ndvi_w1" "sup768_w1_tanchor" "sup768_ndvi_w1_tanchor"; do
    NAME="regbtl_v1_2_gdyn_d768_proj128pcv_${ARM}_newsamp_psuniform"
    SCRIPT="scripts/official/v1_2/regbtl_v1_2_gdyn_d768_proj128pcv_${ARM}_newsampling_psuniform.py"
    python "$SCRIPT" launch "$NAME" "$CLUSTER" \
        $LAUNCH_ARGS \
        --trainer.callbacks.wandb.project="$PROJECT"
done
