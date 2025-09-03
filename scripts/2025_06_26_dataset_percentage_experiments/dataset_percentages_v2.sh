#!/bin/bash

# Script to launch latent_mim_all_data.py with varying dataset percentages

PERCENTAGES=(
    0.00004
    0.0001
    # 0.0004
    # 0.004
    # 0.0078125
    # 0.015625
    # 0.03125
    # 0.0625
    # 0.125
)
# for i in $(seq $(( (${#PERCENTAGES[@]} + 1) / 2 )) $((${#PERCENTAGES[@]} - 1))); do
#     pct="${PERCENTAGES[$i]}"
#     pct_str=$(printf "%.8f" "$pct" | sed 's/^0*//;s/\.$//;s/0*$//')
#     name="latent_mim_cross_only_dec_wc_osm_srtm_dataset_percentage_sweep_${pct_str}"
#     echo "Launching for percentage $pct as $name on ai2/saturn-cirrascale"
#     python3 scripts/2025_06_26_dataset_percentage_experiments/latent_mim_all_data.py launch "$name" ai2/saturn-cirrascale --launch.priority=high --launch.num_gpus=8 --common.dataset_percentage="$pct"
# done

for pct in "${PERCENTAGES[@]}"; do
    # Remove leading zero for name, keep full precision
    pct_str=$(printf "%.8f" "$pct" | sed 's/^0*//;s/\.$//;s/0*$//')
    name="latent_mim_cross_only_dec_wc_osm_srtm_dataset_percentage_sweep_${pct_str}"
    echo "Launching for percentage $pct as $name"
    python3 scripts/2025_06_26_dataset_percentage_experiments/latent_mim_all_data.py launch "$name" ai2/saturn-cirrascale --launch.priority=high --launch.num_gpus=8 --common.dataset_percentage="$pct"
done
