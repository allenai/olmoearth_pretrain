#!/usr/bin/env bash
#
# Run one command of the USDM eval tooling as a CPU-only Beaker job on one
# Jupiter node (embedding dumps, baselines, target lab, CPU probes, tower
# labels). Same environment as the dataset build.
#
# Usage (repo root; the commit must be pushed):
#   NAME=usdm_dump_0of2 HOST=<jupiter host> CMD="python scripts/era5_supervised/v0/usdm/dump_embeddings.py --queue scripts/era5_supervised/v0/usdm/queues/embeddings.json --shard 0/2" \
#     bash scripts/era5_supervised/v0/usdm/launch_cpu.sh
#   DRY_RUN=1 ...    # print the gantry command only
#
# No --cpus / --memory (either would land the job on a GPU slot); one job per
# node, at most 64 threads / workers per job (the commands' defaults).

set -euo pipefail

NAME="${NAME:?set NAME}"
HOST="${HOST:?set HOST to a jupiter hostname (pick_nodes.sh ai2/jupiter 1)}"
CMD="${CMD:?set CMD to the command to run from the repo root}"
THREADS="${THREADS:-56}"
ACCOUNT=$(beaker account whoami --format text | awk 'NR==2{print $2}')
REF=$(git rev-parse HEAD)

cmd=(
  "${GANTRY:-.venv/bin/gantry}" run
  --name "${NAME}"
  --description "USDM eval CPU job: ${CMD}; 0 GPU; commit ${REF:0:9}"
  --beaker-image tylerr/olmo-core-tch291cu128-2025-11-25
  --ref "${REF}" --branch "$(git branch --show-current)"
  --workspace ai2/earth-systems --budget ai2/atec-olmoearth
  --gh-token-secret "${ACCOUNT}_GITHUB_TOKEN"
  --priority urgent --gpus 0 --hostname "${HOST}"
  --min-runtime 45m --retries 2
  --weka dfive-default:/weka/dfive-default
  --env UV_CACHE_DIR=/weka/dfive-default/olmoearth_pretrain/uv-cache
  --env OMP_NUM_THREADS="${THREADS}"
  --install "uv sync --locked --all-extras"
  --post-setup "uv pip install zarr breizhcrops==0.0.4.1"
  --yes
  -- bash -c "${CMD}"
)
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf '%q ' "${cmd[@]}"; echo
  exit 0
fi
GANTRY_CHECK_FOR_UPGRADES=0 "${cmd[@]}" 2>&1 | grep -E 'beaker.org/ex|rror' || true
