#!/bin/bash
# 20k-step A/B validating compile_model + fused AdamW (the 1.37x winner from
# the speed-benchmark ladder) before adopting it in production runs.
#
# Both arms share the init seed and dataloader seed: identical weights and
# data order, so the loss curves should overlay within bf16 noise. Compare
# train/ModalityPatchDiscMaskedVec and train/InfoNCE in the W&B project; also
# grep the compile arm's Beaker log for torch._dynamo cache-limit warnings.
# Permanent checkpoints land every 5000 steps if you want to point eval
# Beaker jobs at both arms afterwards.
set -e

PROJECT="2026_07_09_compile_fused_ab"
CLUSTER="ai2/jupiter"
LAUNCH_ARGS="--launch.num_gpus=8 --launch.priority=high --launch.clusters=[$CLUSTER]"
COMMON_OVERRIDES="--trainer.callbacks.wandb.project=$PROJECT"

python scripts/archived/2026_07_13_v1_2_speedup/compile_fused_ab.py launch ab_baseline $CLUSTER \
    $LAUNCH_ARGS $COMMON_OVERRIDES

python scripts/archived/2026_07_13_v1_2_speedup/compile_fused_ab.py launch ab_compile_fused $CLUSTER \
    $LAUNCH_ARGS $COMMON_OVERRIDES \
    --train_module.compile_model=true \
    --train_module.optim_config.fused=true
