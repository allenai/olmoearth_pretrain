#!/bin/bash
# Launch minimum-4 continue-anneal experiment to test if data splits
# matter when annealing from a pretrained checkpoint.
#
# Resumes from v2 full-baseline at step 200k (first plateau) and anneals
# for 50k steps with cosine decay on two v1 infrastructure splits plus
# a no-filter control and a noise-floor seed.
#
# Usage:
#   bash scripts/tools/launch_continue_anneal.sh

set -euo pipefail

SCRIPT="scripts/official/ablations/continue_anneal.py"
FILTER_DIR="/weka/dfive-default/henryh/helios/olmoearth_pretrain/ablation_filters"
LOAD_PATH="/weka/dfive-default/helios/checkpoints/henryh/data_ablation_v2_full_baseline_1/step200000"
RESUME_STEP=200000
T_MAX=$((RESUME_STEP + 50000))
WANDB_PROJECT="2026_04_continue_anneal"
CLUSTERS="[ai2/jupiter,ai2/titan,ai2/ceres]"
NUM_GPUS=8

launch_one() {
    local run_name="$1"
    local filter_arg="$2"
    local seed="$3"

    echo "=== Launching ${run_name} (seed=${seed}) ==="
    python "${SCRIPT}" launch "${run_name}" ai2/jupiter \
        --launch.num_gpus="${NUM_GPUS}" \
        --launch.clusters="${CLUSTERS}" \
        --trainer.load_path="${LOAD_PATH}" \
        ${filter_arg} \
        --data_loader.num_dataset_repeats_per_epoch=10 \
        --data_loader.seed="${seed}" \
        --trainer.max_duration.value="${T_MAX}" \
        --trainer.max_duration.unit=steps \
        --train_module.scheduler.warmup="${RESUME_STEP}" \
        --train_module.scheduler.t_max="${T_MAX}" \
        --train_module.scheduler.alpha_f=0.0 \
        --trainer.callbacks.wandb.project="${WANDB_PROJECT}"
    echo ""
}

# 1. Control (no filter, seed A)
launch_one "anneal_v2full_cos50k" "" 3622

# 2. Noise floor (no filter, seed B — delta vs #1 is the noise estimate)
launch_one "anneal_v2full_cos50k_s2" "" 8711

# 3. Treatment A: infrastructure-rich
launch_one "anneal_infra_rich_cos50k" \
    "--dataset.filter_idx_file=${FILTER_DIR}/infrastructure_content_infrastructure_rich.npy" 3622

# 4. Treatment B: rural
launch_one "anneal_rural_cos50k" \
    "--dataset.filter_idx_file=${FILTER_DIR}/infrastructure_content_rural.npy" 3622

echo "Launched 4 experiments."
