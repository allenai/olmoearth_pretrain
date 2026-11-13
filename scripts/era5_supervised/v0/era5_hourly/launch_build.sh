#!/usr/bin/env bash
#
# Build the ERA5-Land hourly pretraining corpus
# (olmoearth_pretrain/dataset_creation/era5_hourly) in one CPU-only Beaker job:
# the plan (once, before any shard), the static fields, one shard and its
# validation. Every stage resumes or skips finished work.
#
# Usage (from the repo root; the commit must be pushed):
#   HOST=jupiter-cs-aus-167.reviz.ai2.in bash scripts/era5_supervised/v0/era5_hourly/launch_build.sh  # pilot: shard 0
#   STEPS="shard validate" SHARD=3 HOST=... bash ...
#   DRY_RUN=1 HOST=... bash ...                               # print the gantry command only
#
# CPU-only data job on a jupiter, neptune or saturn node (pick_nodes.sh), one
# job per node, at most 64 workers, at most 50 jobs at once. No --cpus /
# --memory (either would land the job on a GPU slot). The download is
# network-bound: FETCH_WORKERS concurrent chunk reads (default 32, max 64).
# The EarthDataHub token is the workspace secret RSLEARN_EARTHDATAHUB_TOKEN.

set -euo pipefail

HOST="${HOST:?set HOST to a jupiter/neptune/saturn hostname (pick_nodes.sh <cluster> 1)}"
STEPS="${STEPS:-plan static shard validate}"
SHARD="${SHARD:-0}"
FETCH_WORKERS="${FETCH_WORKERS:-32}"
NAME="${NAME:-era5h_build_s${SHARD}}"
ACCOUNT=$(beaker account whoami --format text | awk 'NR==2{print $2}')
REF=$(git rev-parse HEAD)
MOD=olmoearth_pretrain.dataset_creation.era5_hourly

case "${HOST}" in
  jupiter-*|neptune-*|saturn-*) ;;
  *) echo "HOST must be a jupiter, neptune or saturn node, got ${HOST}" >&2; exit 1 ;;
esac
if (( FETCH_WORKERS > 64 )); then echo "FETCH_WORKERS must be <= 64" >&2; exit 1; fi

STAGES=""
for step in ${STEPS}; do
  case "${step}" in
    plan) STAGES+="python -m ${MOD}.plan --workers 64 && " ;;
    static) STAGES+="python -m ${MOD}.static && " ;;
    shard) STAGES+="python -m ${MOD}.shards --shard ${SHARD} --fetch-workers ${FETCH_WORKERS} && " ;;
    validate) STAGES+="python -m ${MOD}.validate --shard ${SHARD} && " ;;
    *) echo "unknown step ${step}" >&2; exit 1 ;;
  esac
done
STAGES+="echo DONE"

cmd=(
  "${GANTRY:-.venv/bin/gantry}" run
  --name "${NAME}"
  --description "ERA5-Land hourly corpus build (${STEPS}; shard ${SHARD}); 0 GPU, ${FETCH_WORKERS} fetch workers; commit ${REF:0:9}"
  --beaker-image tylerr/olmo-core-tch291cu128-2025-11-25
  --ref "${REF}" --branch "$(git branch --show-current)"
  --workspace ai2/earth-systems --budget ai2/atec-olmoearth
  --gh-token-secret "${ACCOUNT}_GITHUB_TOKEN"
  --env-secret "EARTHDATAHUB_TOKEN=RSLEARN_EARTHDATAHUB_TOKEN"
  --priority urgent --gpus 0 --hostname "${HOST}"
  --min-runtime 45m --retries 3
  --weka dfive-default:/weka/dfive-default
  --env UV_CACHE_DIR=/weka/dfive-default/olmoearth_pretrain/uv-cache
  --install "uv sync --locked --all-extras"
  # zarr v3 and the numcodecs bit-round codec the store uses are not in the lock.
  --post-setup "uv pip install zarr==3.4.2 numcodecs==0.17.1"
  --yes
  -- bash -c "${STAGES}"
)
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf '%q ' "${cmd[@]}"; echo
  exit 0
fi
GANTRY_CHECK_FOR_UPGRADES=0 "${cmd[@]}"
