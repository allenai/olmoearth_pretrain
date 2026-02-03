#!/bin/bash

PROJECT="mike-quant-experiment-int8-sweep"

python -m olmoearth_pretrain.internal.full_eval_sweep \
  --cluster=ai2/saturn \
  --checkpoint_path=/weka/dfive-default/helios/checkpoints/joer/phase2.0_base_lr0.0001_wd0.02/step667200 \
  --module_path=scripts/official/base.py \
  --trainer.max_duration.value=700000 \
  --trainer.max_duration.unit=steps \
  --trainer.callbacks.wandb.project="$PROJECT" \
  --task-skip-names=pastis128_sentinel2,pastis128_sentinel1,pastis128_sentinel1_sentinel2 \
  --select_best_val

python -m olmoearth_pretrain.internal.full_eval_sweep \
  --cluster=ai2/saturn \
  --checkpoint_path=/weka/dfive-default/helios/checkpoints/joer/phase2.0_base_lr0.0001_wd0.02/step667200 \
  --module_path=scripts/official/base.py \
  --trainer.max_duration.value=700000 \
  --trainer.max_duration.unit=steps \
  --trainer.callbacks.wandb.project="$PROJECT" \
  --select_best_val \
  --task-skip-names=pastis128_sentinel2,pastis128_sentinel1,pastis128_sentinel1_sentinel2 \
  --quantize_embeddings
