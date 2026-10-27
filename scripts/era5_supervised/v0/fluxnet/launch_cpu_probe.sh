#!/usr/bin/env bash
#
# Launch cpu_probe.py shards as CPU-only Beaker jobs, one per Jupiter node
# (the probe-queue path needs a GPU: olmo-core's process-group setup calls
# torch.cuda.set_device). Same queue files and results directory as the GPU
# queue; finished entries are skipped by either path.
#
# Usage (repo root; commit pushed):
#   QUEUE=arms TASKS="[...]" HOSTS="host1 host2" bash scripts/era5_supervised/v0/fluxnet/launch_cpu_probe.sh
#   DRY_RUN=1 ...    # print the gantry commands only
#
# No --cpus / --memory (either would land the job on a GPU slot); at most 60
# torch threads per job.

set -euo pipefail

QUEUE="${QUEUE:?queue file name in scripts/era5_supervised/v0/fluxnet/queues (no .json)}"
TASKS="${TASKS:?task list, e.g. [fluxnet_gpp_eval,lfmc_woody_eval]}"
HOSTS="${HOSTS:?space-separated jupiter hostnames, one shard each}"
LR_GRID="${LR_GRID:-[1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1,3e-1]}"
THREADS="${THREADS:-60}"
ACCOUNT=$(beaker account whoami --format text | awk 'NR==2{print $2}')
REF=$(git rev-parse HEAD)
read -r -a HOST_LIST <<<"${HOSTS}"
N=${#HOST_LIST[@]}

for i in "${!HOST_LIST[@]}"; do
  cmd=(
    .venv/bin/gantry run
    --name "fluxnet_cpu_${QUEUE}_${i}of${N}"
    --description "FLUXNET CPU probes, queue ${QUEUE} shard ${i}/${N}, tasks ${TASKS}; 0 GPU; commit ${REF:0:9}"
    --beaker-image tylerr/olmo-core-tch291cu128-2025-11-25
    --ref "${REF}" --branch "$(git branch --show-current)"
    --workspace ai2/earth-systems --budget ai2/atec-olmoearth
    --gh-token-secret "${ACCOUNT}_GITHUB_TOKEN"
    --priority urgent --gpus 0 --hostname "${HOST_LIST[$i]}"
    --min-runtime 45m --retries 2
    --weka dfive-default:/weka/dfive-default
    --env UV_CACHE_DIR=/weka/dfive-default/olmoearth_pretrain/uv-cache
    --env OMP_NUM_THREADS="${THREADS}"
    --install "uv sync --locked --all-extras"
    --post-setup "uv pip install breizhcrops==0.0.4.1"
    --yes
    -- python scripts/era5_supervised/v0/fluxnet/cpu_probe.py
    --queue "scripts/era5_supervised/v0/fluxnet/queues/${QUEUE}.json"
    --tasks "${TASKS}" --lr-grid "${LR_GRID}" --shard "${i}/${N}" --threads "${THREADS}"
  )
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '%q ' "${cmd[@]}"; echo
  else
    GANTRY_CHECK_FOR_UPGRADES=0 "${cmd[@]}" 2>&1 | grep -E 'beaker.org/ex|rror' || true
  fi
done
