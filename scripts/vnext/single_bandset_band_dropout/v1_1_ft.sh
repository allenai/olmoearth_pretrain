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
COMMON_ARGS="--cluster=${CLUSTER} --launch.priority=urgent --trainer.callbacks.wandb.project=${PROJECT}"

SEEDS=(0 42 1234)

for SEED in "${SEEDS[@]}"; do
    # single_bandset base (no s1 drop, random time)
    ${SCRIPT} ${COMMON_ARGS} \
        --module_path=scripts/vnext/single_bandset_band_dropout/base_band_dropout_no_s1_drop_random_time.py \
        --checkpoint_path=/weka/dfive-default/helios/checkpoints/yawenzzzz/single_bandset_no_s1_drop_random_time_dropout_0.2/step667200 \
        --model_name=single_bandset_no_s1_drop_random_time_dropout_0.2_step667200_nobanddropout \
        --finetune_seed=${SEED}

    # nano 1.1
    ${SCRIPT} ${COMMON_ARGS} \
        --module_path=scripts/vnext/single_bandset_band_dropout/nano.py \
        --checkpoint_path=/weka/dfive-default/helios/checkpoints/gabrielt/nano_1.1_lr0.0002_wd0.02/step667200 \
        --model_name=nano_1.1_lr0.0002_wd0.02_step667200_nobanddropout \
        --finetune_seed=${SEED}
done

echo "All eval jobs submitted to wandb project: ${PROJECT}"
