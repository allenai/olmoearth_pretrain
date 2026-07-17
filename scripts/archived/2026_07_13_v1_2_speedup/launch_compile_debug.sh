#!/bin/bash
# Compile-debug arms: attribute WHY compile_model=true degrades training
# (ab_compile_only diverged from production run q3x9lvww at the ~3000-step
# phase transition; fused AdamW alone tracked it to ~1e-4).
#
# Arm 1 (emulate_casts): Inductor emulates eager's precision casts.
# Arm 2 (eager_rope): the learnable-RoPE trig chain runs eagerly, rest of the
#                     block stays compiled.
#
# Both: compile hardcoded ON, fused OFF, 8000 steps (all LR warmup, so
# directly comparable to q3x9lvww's first 8k steps and to ab_compile_only).
# Compare in the same W&B project: 2026_07_09_compile_fused_ab.
set -e

PROJECT="2026_07_09_compile_fused_ab"
CLUSTER="ai2/jupiter"
LAUNCH_ARGS="--launch.num_gpus=8 --launch.priority=urgent --launch.clusters=[$CLUSTER]"
COMMON_OVERRIDES="--trainer.callbacks.wandb.project=$PROJECT"

python scripts/archived/2026_07_13_v1_2_speedup/compile_fix_emulate_casts.py launch ab_compile_emulate_casts $CLUSTER \
    $LAUNCH_ARGS $COMMON_OVERRIDES

python scripts/archived/2026_07_13_v1_2_speedup/compile_fix_eager_rope.py launch ab_compile_eager_rope $CLUSTER \
    $LAUNCH_ARGS $COMMON_OVERRIDES
