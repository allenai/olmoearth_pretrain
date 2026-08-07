#!/bin/bash
# The frozen-teacher student continuations of lin_sup768_w1: re-warmed vs floor LR.
#
# Both resume the parent's step667200 weights with everything except the detached
# student frozen (optim lr 0 + a student param group). See
# regbtl_v1_2_frozen_student_common.py for why, and for the two signals that
# verify the freeze took (flat d768 evals; distill_cosine_d128 turning DOWN).
#
# The three load flags matter and are not defaults:
#   load_path            resume the parent's weights
#   load_optim_state=false   REQUIRED -- adding the student param group changes the
#                            optimizer's group structure, so parent optimizer state
#                            would mis-map. Costs a short fresh-Adam transient.
#   load_trainer_state=false restart the step counter at 0, so each arm's
#                            max_duration (50 epochs) is the EXTENSION budget
#                            rather than a new total. This is also what avoids the
#                            cosine-recompute LR discontinuity, and what makes the
#                            re-warm arm's 2k warmup start where we want it.
#
# Same wandb project as the parent so the in-loop proj128 curves sit alongside it.
set -e

PROJECT="2026_07_02_perceiver"
CLUSTER="ai2/jupiter"
LAUNCH_ARGS="--launch.num_gpus=8 --launch.priority=high --launch.clusters=[ai2/jupiter,ai2/ceres]"
CKPT="/weka/dfive-default/olmoearth_pretrain/checkpoints/gabrielt/regbtl_v1_2_gdyn_d768_proj128lin_sup768_w1_newsamp_psuniform/step667200"

# Defragment the CUDA allocator (see launch_regbtl_v1_2_newsampling.sh).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for ARM in "frozen_rewarm" "frozen_floor"; do
    NAME="regbtl_v1_2_gdyn_d768_proj128lin_sup768_w1_${ARM}"
    SCRIPT="scripts/official/v1_2/${NAME}.py"
    python "$SCRIPT" launch "$NAME" "$CLUSTER" \
        $LAUNCH_ARGS \
        --trainer.load_path="$CKPT" \
        --trainer.load_optim_state=false \
        --trainer.load_trainer_state=false \
        --trainer.callbacks.wandb.project="$PROJECT"
done
