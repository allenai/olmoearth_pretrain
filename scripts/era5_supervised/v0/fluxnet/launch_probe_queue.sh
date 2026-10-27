#!/usr/bin/env bash
#
# Launch one queue of eval-only checkpoint sweeps (probe_queue.py) as a single
# 1-GPU Beaker job, so dozens of re-probes share one allocation. The job runs
# the same environment as base.py launches (image, uv lock, post-setup, W&B and
# Beaker secrets); each queue entry is its own torchrun process and W&B run
# (project era5_fluxnet_eval), and writes its evals to
# /weka/dfive-default/rslearn-eai/datasets/fluxnet/probe_results/<name>.jsonl.
# Finished entries leave a .done marker, so a preempted / retried job resumes.
#
# Usage (repo root; the commit must be pushed):
#   QUEUE=curves TASKS=all bash scripts/era5_supervised/v0/fluxnet/launch_probe_queue.sh
#   QUEUE=arms TASKS=all bash ...           # entries may override TASKS
#   DRY_RUN=1 QUEUE=... bash ...          # print the gantry command only
#
# TASKS: flux (the four FLUXNET evals), flux_lfmc (+ LFMC as an anchor to the
# earlier sweeps), all (+ the six existing evals). LR_GRID overrides the probe
# LR grid (default: the 8-value grid of the earlier probe-LR sweeps).

set -euo pipefail

QUEUE="${QUEUE:?set QUEUE to a file name in scripts/era5_supervised/v0/fluxnet/queues (no .json)}"
TASKS="${TASKS:-all}"
CLUSTER="${CLUSTER:-ai2/saturn}"
LR_GRID="${LR_GRID:-[1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1,3e-1]}"
NAME="${NAME:-fluxnet_probe_${QUEUE}}"
ACCOUNT=$(beaker account whoami --format text | awk 'NR==2{print $2}')
REF=$(git rev-parse HEAD)

FLUX="fluxnet_gpp_eval,fluxnet_le_eval,fluxnet_nee_eval,fluxnet_gpp_anom_eval"
OLD="dfmc1000h_grav_eval,lfmc_woody_eval,cybench_wheat_eval,cybench_maize_eval,burnrisk_canada_nbac_eval,landslide_era5_eval"
case "${TASKS}" in
  flux) TASK_LIST="[${FLUX}]" ;;
  flux_lfmc) TASK_LIST="[${FLUX},lfmc_woody_eval]" ;;
  all) TASK_LIST="[${FLUX},${OLD}]" ;;
  *) echo "unknown TASKS=${TASKS}" >&2; exit 1 ;;
esac

cmd=(
  .venv/bin/gantry run
  --name "${NAME}"
  --description "FLUXNET eval probe queue ${QUEUE} (tasks ${TASKS}, LR grid ${LR_GRID}); 1 GPU ${CLUSTER}; commit ${REF:0:9}"
  --beaker-image tylerr/olmo-core-tch291cu128-2025-11-25
  --ref "${REF}" --branch "$(git branch --show-current)"
  --workspace ai2/earth-systems --budget ai2/atec-olmoearth
  --gh-token-secret "${ACCOUNT}_GITHUB_TOKEN"
  --env-secret "WANDB_API_KEY=${ACCOUNT}_WANDB_API_KEY"
  --env-secret "BEAKER_TOKEN=${ACCOUNT}_BEAKER_TOKEN"
  --priority urgent --gpus 1 --cluster "${CLUSTER}"
  --shared-memory 256GiB
  --min-runtime 90m --retries 3
  --weka dfive-default:/weka/dfive-default
  --env UV_CACHE_DIR=/weka/dfive-default/olmoearth_pretrain/uv-cache
  --env OLMO_SHARED_FS=1
  --env OMP_NUM_THREADS=8
  --env LOG_FILTER_TYPE=local_rank0_only
  --install "uv sync --locked --all-extras"
  --post-setup "uv pip install breizhcrops==0.0.4.1"
  --yes
  -- python scripts/era5_supervised/v0/fluxnet/probe_queue.py
  --queue "scripts/era5_supervised/v0/fluxnet/queues/${QUEUE}.json"
  --cluster "${CLUSTER}" --tasks "${TASK_LIST}" --lr-grid "${LR_GRID}"
)
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf '%q ' "${cmd[@]}"; echo
  exit 0
fi
GANTRY_CHECK_FOR_UPGRADES=0 "${cmd[@]}"
