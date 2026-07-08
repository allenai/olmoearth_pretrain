#!/bin/bash
# End-to-end fused-AdamW comparison on the v1.2 base model: two FULL production
# runs (300 epochs, ~667k steps), identical except the optimizer kernel.
#
# base.py now defaults to fused=True, so the fused arm runs as-is and the
# baseline arm overrides it back to false. Both route in-loop evals to
# separate Beaker jobs (base_fused_e2e.py), so training wall clock is a clean
# end-to-end measurement: expect ~4d5h (fused) vs ~5d6h (baseline), with
# matching loss curves and downstream evals.
#
# Same init/data seeds in both arms; compare in W&B project
# 2026_07_08_fused_adamw_e2e (loss + throughput) and the eval-job metrics.
set -e

PROJECT="2026_07_08_fused_adamw_e2e"
CLUSTER="ai2/jupiter"
LAUNCH_ARGS="--launch.num_gpus=8 --launch.priority=urgent --launch.clusters=[$CLUSTER]"
COMMON_OVERRIDES="--trainer.callbacks.wandb.project=$PROJECT"

python scripts/official/v1_2/base_fused_e2e.py launch v1_2_base_fused_adamw $CLUSTER \
    $LAUNCH_ARGS $COMMON_OVERRIDES

python scripts/official/v1_2/base_fused_e2e.py launch v1_2_base_unfused_baseline $CLUSTER \
    $LAUNCH_ARGS $COMMON_OVERRIDES \
    --train_module.optim_config.fused=false
