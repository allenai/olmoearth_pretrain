#!/bin/bash
set -euo pipefail

# Re-run KNN/LP eval sweeps with --select_best_val so the test metric is
# computed against the best-val probe (not just the end-of-training probe).
#
# Run name is overridden via --model_name=<orig>_inctest so the trainer
# save_folder and wandb_runid.txt diverge from the prior runs (avoids
# resuming the old wandb run and avoids checkpoint/save_folder collisions).

PROJECT="v1_1_knn_lp_inctest"
CLUSTER="ai2/jupiter"
SCRIPT="python -m olmoearth_pretrain.internal.full_eval_sweep"
COMMON_ARGS="--cluster=${CLUSTER} --select_best_val --launch.priority=high --trainer.callbacks.wandb.project=${PROJECT}"

# single_bandset base (no s1 drop, random time)
${SCRIPT} ${COMMON_ARGS} \
    --module_path=scripts/vnext/single_bandset_band_dropout/base_band_dropout_no_s1_drop_random_time.py \
    --checkpoint_path=/weka/dfive-default/helios/checkpoints/yawenzzzz/single_bandset_no_s1_drop_random_time_dropout_0.2/step667200 \
    --model_name=single_bandset_no_s1_drop_random_time_dropout_0.2_step667200_inctest

# nano 1.1
${SCRIPT} ${COMMON_ARGS} \
    --module_path=scripts/vnext/single_bandset_band_dropout/nano.py \
    --checkpoint_path=/weka/dfive-default/helios/checkpoints/gabrielt/nano_1.1_lr0.0002_wd0.02/step667200 \
    --model_name=nano_1.1_lr0.0002_wd0.02_step667200_inctest

echo "All eval jobs submitted to wandb project: ${PROJECT}"
