#!/bin/bash
set -euo pipefail

# --model_name overrides the checkpoint-derived base run name, so the
# trainer save_folder and wandb run id diverge from the prior runs (which
# were launched with the same checkpoint + seeds and would otherwise
# collide/resume). --finetune_seed=<seed> sweeps seeds and appends
# _seed<seed> to the run_name.

PROJECT="v1_1_finetuning_v2"
CLUSTER="ai2/jupiter"
SCRIPT="python -m olmoearth_pretrain.internal.full_eval_sweep_finetune"
COMMON_ARGS="--cluster=${CLUSTER} --launch.priority=urgent --project_name=${PROJECT}"

SEEDS=(0 42 1234)

for SEED in "${SEEDS[@]}"; do
    # single_bandset base (no s1 drop, random time)
    ${SCRIPT} ${COMMON_ARGS} \
        --module_path=scripts/archived/2026_04_22_add_hidden_layer_to_initial_projection/hidden1.py \
        --checkpoint_path=/weka/dfive-default/helios/checkpoints/favyen/hidden1/step667200 \
        --model_name=hidden1_step667200_nobanddropout \
        --finetune_seed=${SEED}
done

echo "All eval jobs submitted to wandb project: ${PROJECT}"
