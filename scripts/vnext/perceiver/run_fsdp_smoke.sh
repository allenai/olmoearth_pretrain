#!/bin/bash
# 1-GPU FSDP bf16 crash-test for a vnext script. Usage:
#   bash scripts/vnext/perceiver/run_fsdp_smoke.sh <script.py> <name> [overrides...]
# Fast-start setup: shared dataloader work_dir (reuses the global-index
# cache across smokes) + small global batch (less first-fetch I/O). Both
# irrelevant to what a smoke validates (code paths / dtypes / shapes).
set -eu
SCRIPT=$1; NAME=$2; shift 2
SAVE=/weka/dfive-default/helios/checkpoints/joer/$NAME
CACHE=/weka/dfive-default/helios/checkpoints/joer/smoke-dataloader-cache
rm -rf "$SAVE"
CUDA_VISIBLE_DEVICES=${SMOKE_GPU:-0} PYTHONPATH=. \
  /root/dev/.venv/bin/torchrun --nproc_per_node=1 "$SCRIPT" train "$NAME" local \
  --trainer.save_folder="$SAVE" --trainer.work_dir="$SAVE" \
  --trainer.checkpointer.work_dir="$SAVE" \
  --data_loader.work_dir="$CACHE" \
  --data_loader.global_batch_size=128 \
  --data_loader.num_workers=4 \
  --trainer.callbacks.wandb.enabled=false \
  "$@"
