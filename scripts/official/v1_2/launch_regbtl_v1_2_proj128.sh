#!/bin/bash
# The 6 detached-student ("proj128") runs: a d768 wideread regbtl teacher trained as
# usual, plus a DETACHED 128d student distilled from it online (cosine + Gram), with
# supervision heads on {d768 only, both, 128d only} x student {linear, perceiver}.
# See regbtl_v1_2_proj_common.py for the motivation and variant matrix. The
# architecture is baked into each script (not CLI overrides) so the in-loop Beaker
# eval jobs reconstruct the matching model; every run evaluates BOTH heads in-loop
# (the _proj128 eval tasks probe the student).
set -e

PROJECT="2026_07_02_perceiver"
CLUSTER="ai2/jupiter"
LAUNCH_ARGS="--launch.num_gpus=8 --launch.priority=urgent --launch.clusters=[ai2/jupiter,ai2/ceres]"

# Defragment the CUDA allocator (see launch_regbtl_v1_2_newsampling.sh).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for ARCH in lin pcv; do
    for SUP in sup768 supboth sup128; do
        NAME="regbtl_v1_2_gdyn_d768_proj128${ARCH}_${SUP}_w1_newsamp_psuniform"
        SCRIPT="scripts/official/v1_2/regbtl_v1_2_gdyn_d768_proj128${ARCH}_${SUP}_w1_newsampling_psuniform.py"
        python "$SCRIPT" launch "$NAME" "$CLUSTER" \
            $LAUNCH_ARGS \
            --trainer.callbacks.wandb.project="$PROJECT"
    done
done
