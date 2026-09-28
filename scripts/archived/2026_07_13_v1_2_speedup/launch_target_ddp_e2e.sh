#!/bin/bash
# Speedup A/B arms on the v1.2 base model: projection-only target encoder and
# FSDP -> replicated DP (manual gradient all-reduce), motivated by the
# 2026_07_08_speed_benchmark profiler traces (54% GPU idle in the fused arm;
# NCCL all-gathers + CPU launch overhead dominate, gemm is ~11ms of a 407ms step).
#
# All arms inherit fused AdamW from base.py (fused=True default), so:
#   1. proj_target     : fused + frozen projection-only target (FSDP unchanged)
#   2. ddp             : fused + replicated DP + bf16 autocast (full target copy)
#   3. proj_target_ddp : fused + BOTH -- all speedups stacked (production candidate)
#
# Baseline: the fused arm of launch_fused_e2e.sh (v1_2_base_fused_adamw),
# already running with the same init/data seeds. All arms log to the SAME
# W&B project (2026_07_08_fused_adamw_e2e) so loss curves overlay directly.
# Compare loss + downstream eval-job metrics for regressions, and
# throughput/device/BPS + power draw for the speedup.
set -e

PROJECT="2026_07_08_fused_adamw_e2e"
CLUSTER="ai2/jupiter"
LAUNCH_ARGS="--launch.num_gpus=8 --launch.priority=urgent --launch.clusters=[$CLUSTER]"
COMMON_OVERRIDES="--trainer.callbacks.wandb.project=$PROJECT"

# python scripts/archived/2026_07_13_v1_2_speedup/base_proj_target_e2e.py launch v1_2_base_proj_target $CLUSTER \
#     $LAUNCH_ARGS $COMMON_OVERRIDES

# python scripts/archived/2026_07_13_v1_2_speedup/base_ddp_e2e.py launch v1_2_base_ddp_v2 $CLUSTER \
#     $LAUNCH_ARGS $COMMON_OVERRIDES

python scripts/archived/2026_07_13_v1_2_speedup/base_proj_target_ddp_e2e.py launch v1_2_base_proj_target_ddp_v2 $CLUSTER \
    $LAUNCH_ARGS $COMMON_OVERRIDES
