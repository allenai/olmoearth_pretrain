#!/bin/bash
# Speed-benchmark ladder for the v1.2 / regbtl training config.
#
# Each arm is a ~500-step run of speed_benchmark.py (production config minus
# evals/checkpointing, plus a torch.profiler trace on rank 0 and a per-step
# padding-fraction metric). One flag changes per arm; compare
# throughput/device/BPS across arms in the W&B project below.
#
# Requires flash-attn in the image for the flash arms (confirmed present).
# Model variant: auto (regbtl if regbtl_v1_2_common is on the branch, else
# v1.2 base) -- override with SPEED_BENCH_MODEL=base|regbtl in the env.
set -e

PROJECT="2026_07_08_speed_benchmark"
CLUSTER="ai2/jupiter"
LAUNCH_ARGS="--launch.num_gpus=8 --launch.priority=high --launch.clusters=[$CLUSTER]"
COMMON_OVERRIDES="--trainer.callbacks.wandb.project=$PROJECT"

FLASH="--model.encoder_config.use_flash_attn=true --model.decoder_config.use_flash_attn=true"
COMPILE="--train_module.compile_model=true"
FUSED="--train_module.optim_config.fused=true"

# 1. Baseline: exactly the production step, for the reference BPS, the
#    profiler trace, and the padding-fraction measurement.
python scripts/archived/2026_07_13_v1_2_speedup/speed_benchmark.py launch bench_baseline $CLUSTER \
    $LAUNCH_ARGS $COMMON_OVERRIDES

# 2. FlashAttention + varlen packing (removes padding FLOPs and the per-layer
#    dense attention-mask materialization).
python scripts/archived/2026_07_13_v1_2_speedup/speed_benchmark.py launch bench_flash $CLUSTER \
    $LAUNCH_ARGS $COMMON_OVERRIDES $FLASH

# 3. torch.compile only. Watch step time over the whole run: variable patch
#    sizes / sequence lengths may cause recompilation; if step time never
#    stabilizes, rerun with SPEED_BENCH_STEPS=1500.
python scripts/archived/2026_07_13_v1_2_speedup/speed_benchmark.py launch bench_compile $CLUSTER \
    $LAUNCH_ARGS $COMMON_OVERRIDES $COMPILE

# 4. Flash + compile (the expected production candidate).
python scripts/archived/2026_07_13_v1_2_speedup/speed_benchmark.py launch bench_flash_compile $CLUSTER \
    $LAUNCH_ARGS $COMMON_OVERRIDES $FLASH $COMPILE

# 5. Fused AdamW (independent, expected small).
python scripts/archived/2026_07_13_v1_2_speedup/speed_benchmark.py launch bench_fused_adamw $CLUSTER \
    $LAUNCH_ARGS $COMMON_OVERRIDES $FUSED
