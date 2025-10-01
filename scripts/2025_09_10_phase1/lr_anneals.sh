#!/usr/bin/env bash
set -euo pipefail

SCRIPT="scripts/2025_09_10_phase1/script_chm_cdl_worldcereal_anneals.py"
CHECKPOINT_BASE="/weka/dfive-default/helios/checkpoints/henryh/base_v6.1_add_chm_cdl_worldcereal"

TOKEN_BUDGET=1750
MICROBATCH=32

# Edit these lists to sweep different anneals
RESTART_STEPS=(450000 250000)
TMAX_NUM_STEPS=(50000 100000 200000)           # step n (resume here)

for n in "${RESTART_STEPS[@]}"; do
	n_minus_1=$((n - 1))
	load_path="${CHECKPOINT_BASE}/step${n}"
	TMAX_STEPS=()
	for m in "${TMAX_NUM_STEPS[@]}"; do
		TMAX_STEPS+=("$((n + m))")
	done
	for m in "${TMAX_NUM_STEPS[@]}"; do
		set -x
        launch_name="base_v6.1_add_chm_cdl_worldcereal_${n}_${TMAX_STEPS}"
		python3 "$SCRIPT" $launch_name \ ai2/jupiter
			--data_loader.token_budget="${TOKEN_BUDGET}" \
			--train_module.rank_microbatch_size="${MICROBATCH}" \
			--trainer.load_path="${load_path}" \
			--train_module.scheduler.schedulers.1.t_max="${TMAX_STEPS}" \
			--train_module.scheduler.schedulers_max="[${n_minus_1}]"
		set +x
	done
done
