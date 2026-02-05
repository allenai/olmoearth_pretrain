#!/bin/bash

PROJECT="mike-quant-experiment"

# baseline, olmoearth base, no quant, no dim reduction
python -m olmoearth_pretrain.internal.full_eval_sweep \
  --cluster=ai2/saturn \
  --checkpoint_path=/weka/dfive-default/helios/checkpoints/joer/phase2.0_base_lr0.0001_wd0.02/step667200 \
  --module_path=scripts/official/base.py \
  --trainer.max_duration.value=700000 \
  --trainer.max_duration.unit=steps \
  --trainer.callbacks.wandb.project="$PROJECT" \
  --defaults_only \
  --task-skip-names=pastis128_sentinel2,pastis128_sentinel1,pastis128_sentinel1_sentinel2

# olmoearth base, int8 quant, no dim reduction
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

# olmoearth base, int8 quant, 512 dim reduction
python -m olmoearth_pretrain.internal.full_eval_sweep \
  --cluster=ai2/saturn \
  --checkpoint_path=/weka/dfive-default/helios/checkpoints/joer/phase2.0_base_lr0.0001_wd0.02/step667200 \
  --module_path=scripts/official/base.py \
  --trainer.max_duration.value=700000 \
  --trainer.max_duration.unit=steps \
  --trainer.callbacks.wandb.project="$PROJECT" \
  --select_best_val \
  --task-skip-names=pastis128_sentinel2,pastis128_sentinel1,pastis128_sentinel1_sentinel2 \
  --quantize_embeddings \
  --embedding_dim=512

# olmoearth base, int8 quant, 256 dim reduction
python -m olmoearth_pretrain.internal.full_eval_sweep \
  --cluster=ai2/saturn \
  --checkpoint_path=/weka/dfive-default/helios/checkpoints/joer/phase2.0_base_lr0.0001_wd0.02/step667200 \
  --module_path=scripts/official/base.py \
  --trainer.max_duration.value=700000 \
  --trainer.max_duration.unit=steps \
  --trainer.callbacks.wandb.project="$PROJECT" \
  --select_best_val \
  --task-skip-names=pastis128_sentinel2,pastis128_sentinel1,pastis128_sentinel1_sentinel2 \
  --quantize_embeddings \
  --embedding_dim=256
