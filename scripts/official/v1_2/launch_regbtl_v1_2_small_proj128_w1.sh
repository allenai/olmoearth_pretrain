#!/bin/bash
# The 2 w1 pcv detached-student runs on the SMALL backbone (384-d, depth 12, 6 heads)
# with a d384 teacher register grid. Motivation: the v1.2 size sweep's small model is
# very performant relative to base for its cost, and the proj128 program's product is
# the 128d student -- if the student's ceiling comes from distillation quality rather
# than raw teacher capacity, small should retain most of the d768 student's quality far
# more cheaply. Only {sup768, supboth} x pcv is run: pcv is the deployed student
# architecture and w1 is the leading teacher arm under the new sampler. LR is the small
# preset's 2e-4; everything else (evals every 40k with PASTIS first, uniform patch
# sizes, fused AdamW, [128, 64] student) matches the d768 w1 runs.
set -e

PROJECT="2026_07_02_perceiver"
CLUSTER="ai2/jupiter"
LAUNCH_ARGS="--launch.num_gpus=8 --launch.priority=urgent --launch.clusters=[ai2/jupiter,ai2/ceres]"

# Defragment the CUDA allocator (see launch_regbtl_v1_2_newsampling.sh).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for SUP in sup768 supboth; do
    NAME="regbtl_v1_2_small_gdyn_d384_proj128pcv_${SUP}_w1_newsamp_psuniform"
    SCRIPT="scripts/official/v1_2/regbtl_v1_2_small_gdyn_d384_proj128pcv_${SUP}_w1_newsampling_psuniform.py"
    python "$SCRIPT" launch "$NAME" "$CLUSTER" \
        $LAUNCH_ARGS \
        --trainer.callbacks.wandb.project="$PROJECT"
done
