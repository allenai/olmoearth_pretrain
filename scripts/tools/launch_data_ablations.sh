#!/bin/bash
# Launch all data ablation experiments using base_speedup on 8 GPUs.
# Each experiment trains from scratch on 100k samples filtered by a single axis.
#
# Usage:
#   bash scripts/tools/launch_data_ablations.sh
#
# To dry-run first (replace 'launch' with 'dry_run' and cluster with 'local'):
#   sed 's/launch/dry_run/g; s/ai2\/jupiter/local/g' scripts/tools/launch_data_ablations.sh | bash

set -euo pipefail

SCRIPT="scripts/vnext/speedups/base_speedup.py"
FILTER_DIR="/weka/dfive-default/henryh/helios/olmoearth_pretrain/ablation_filters"
WANDB_PROJECT="2025_03_30_data_ablations"
CLUSTERS="[ai2/jupiter,ai2/titan,ai2/ceres]"
NUM_GPUS=8

EXPERIMENTS=(
    "random_baseline"
    "quality_gate"
    "spatial_complexity_complex"
    "spatial_complexity_smooth"
    "temporal_dynamics_static"
    "temporal_dynamics_dynamic"
    "land_cover_diversity_homogeneous"
    "land_cover_diversity_diverse"
    "spectral_richness_flat_spectrum"
    "spectral_richness_rich_spectrum"
    "infrastructure_content_rural"
    "infrastructure_content_infrastructure_rich"
    "geographic_regime_tropical_subtropical"
    "geographic_regime_temperate_boreal"
)

for exp in "${EXPERIMENTS[@]}"; do
    run_name="data_ablation_${exp}"
    filter_path="${FILTER_DIR}/${exp}.npy"

    echo "=== Launching ${run_name} ==="
    python "${SCRIPT}" launch "${run_name}" ai2/jupiter \
        --launch.num_gpus="${NUM_GPUS}" \
        --launch.clusters="${CLUSTERS}" \
        --dataset.filter_idx_file="${filter_path}" \
        --trainer.callbacks.wandb.project="${WANDB_PROJECT}"

    echo ""
done

echo "All ${#EXPERIMENTS[@]} experiments launched."
