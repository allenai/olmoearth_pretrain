#!/bin/bash
# Comparison experiments for rank_max_lr feature
# Run MADOS eval every 500 steps for 2000 steps total
# Requires 8 GPUs for rank_max_lr to be active

# Baseline (without rank_max_lr)
python scripts/official/base.py launch mados-baseline ai2/jupiter-cirrascale-2 \
  --trainer.max_duration='Duration.steps(2000)' \
  --trainer.callbacks.downstream_evaluator.tasks_to_run='["mados"]' \
  --trainer.callbacks.downstream_evaluator.tasks.mados.eval_interval='Duration.steps(500)' \
  --trainer.callbacks.downstream_evaluator.tasks.mados.rank_max_lr=False \
  --launch.num_gpus=8

# With rank_max_lr (each GPU uses different LR, take max)
python scripts/official/base.py launch mados-rank-max-lr ai2/jupiter-cirrascale-2 \
  --trainer.max_duration='Duration.steps(2000)' \
  --trainer.callbacks.downstream_evaluator.tasks_to_run='["mados"]' \
  --trainer.callbacks.downstream_evaluator.tasks.mados.eval_interval='Duration.steps(500)' \
  --trainer.callbacks.downstream_evaluator.tasks.mados.rank_max_lr=True \
  --launch.num_gpus=8
