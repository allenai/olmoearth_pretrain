#!/bin/bash
# The 4 w1 detached-student runs: {lin, pcv} x {sup768, supboth} at supervision base
# weight 1.0. Motivation: under the new sampler the w1 teacher leads w0p1 by
# ~1.4-1.9 mIoU on the ps=1 PASTIS exports (the ceiling the student inherits), and
# the classification penalty seen for w1 lives on the ps=4 geobench probes, not the
# ps=1 embedding suite. supboth is re-tested because the w0p1 sup768-vs-supboth
# comparison rested on a single eval round. sup128 is dropped (the teacher collapses
# without encoder supervision -- settled). Embedding evals every 40k with PASTIS
# tasks first (see add_proj_loop_eval_beaker_job).
set -e

PROJECT="2026_07_02_perceiver"
CLUSTER="ai2/jupiter"
LAUNCH_ARGS="--launch.num_gpus=8 --launch.priority=urgent --launch.clusters=[ai2/jupiter,ai2/ceres]"

# Defragment the CUDA allocator (see launch_regbtl_v1_2_newsampling.sh).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for ARCH in lin pcv; do
    for SUP in sup768 supboth; do
        NAME="regbtl_v1_2_gdyn_d768_proj128${ARCH}_${SUP}_w1_newsamp_psuniform"
        SCRIPT="scripts/official/v1_2/regbtl_v1_2_gdyn_d768_proj128${ARCH}_${SUP}_w1_newsampling_psuniform.py"
        python "$SCRIPT" launch "$NAME" "$CLUSTER" \
            $LAUNCH_ARGS \
            --trainer.callbacks.wandb.project="$PROJECT"
    done
done
