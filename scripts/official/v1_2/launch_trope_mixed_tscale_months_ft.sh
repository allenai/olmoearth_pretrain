#!/bin/bash
set -euo pipefail

# Full finetuning eval sweep for trope_mixed_tscale_months.
#
# Sweeps FT_LRS = [1e-4, 5e-4, 1e-3] per seed, over 3 seeds (0, 42, 1234):
#   3 seeds x 3 lrs = 9 beaker jobs, each finetuning all 10 FT eval tasks on 1 GPU.
#
# Test metrics are produced automatically: the downstream_evaluator callback sets
# run_on_test=True whenever FINETUNE=1 (exported by full_eval_sweep_finetune), so no
# extra flag is needed at launch. Pull them out of W&B afterwards with the
# get_max_eval_metrics command at the bottom of this file.
#
# The four rope_* overrides are REQUIRED: this checkpoint was trained before
# --load_arch_from_checkpoint existed (added 2026-06-06), so its arch is not stored in
# the checkpoint and must be passed on the CLI. Values match the KNN/LP test eval for
# this same checkpoint (rope_eval_beaker/rope_eval_test_runs1-19.yaml).

PROJECT="trope_mixed_tscale_months_finetune"
CLUSTER="ai2/jupiter"
CHECKPOINT="/weka/dfive-default/helios/checkpoints/yawenzzzz/v1_2_nano/step667200"
MODULE="scripts/official/v1_2/nano.py"
SCRIPT="python -m olmoearth_pretrain.internal.full_eval_sweep_finetune"

COMMON_ARGS="--cluster=${CLUSTER} --launch.priority=urgent --project_name=${PROJECT} \
  --module_path=${MODULE} \
  --checkpoint_path=${CHECKPOINT}"

SEEDS=(0 42 1234)

for SEED in "${SEEDS[@]}"; do
    ${SCRIPT} ${COMMON_ARGS} --finetune_seed="${SEED}"
done

echo "All eval jobs submitted to wandb project: ${PROJECT}"

# To pull val + test metrics once the runs finish:
#   python scripts/tools/get_max_eval_metrics_from_wandb.py \
#     --project_name=${PROJECT} \
#     --run_prefix=trope_mixed_tscale_months_step667200 \
#     --finetune --get_test_metrics
