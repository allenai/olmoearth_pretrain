#!/usr/bin/env bash
#
# Build the snow eval dataset in one CPU-only Beaker job (stages of
# olmoearth_pretrain/dataset_creation/snow): MODIS tile screening and per-cell
# snow counts from the Planetary Computer, cell phenology, cell sampling + ERA5-Land
# series from the weka mirror, candidates (targets + splits), then the rslearn
# windows. Every stage is resumable or idempotent.
#
# Usage (from the repo root; the commit must be pushed):
#   HOST=jupiter-cs-aus-192.reviz.ai2.in bash scripts/era5_supervised/v0/snow/launch_build.sh
#   STEPS="phenology cells" HOST=... bash ...   # any of screen counts phenology cells candidates windows
#   STEPS=counts SHARD=1/2 HOST=... bash ...    # half of the MODIS tiles (one job per node)
#   DRY_RUN=1 HOST=... bash ...                 # print the gantry command only
#
# CPU-only: no --cpus / --memory (either would land the job on a GPU slot);
# one job per node, at most 64 workers (WORKERS, default 48; the MODIS reads are
# network-bound, MODIS_WORKERS default 64).

set -euo pipefail

HOST="${HOST:?set HOST to a jupiter hostname (pick_nodes.sh ai2/jupiter 1)}"
STEPS="${STEPS:-screen counts}"
WORKERS="${WORKERS:-48}"
MODIS_WORKERS="${MODIS_WORKERS:-64}"
NAME="${NAME:-snow_build}"
WINDOW_ARGS="${WINDOW_ARGS:-}"
SHARD="${SHARD:-0/1}"
ACCOUNT=$(beaker account whoami --format text | awk 'NR==2{print $2}')
REF=$(git rev-parse HEAD)
MOD=olmoearth_pretrain.dataset_creation.snow

STAGES=""
for step in ${STEPS}; do
  case "${step}" in
    screen) STAGES+="python -m ${MOD}.modis screen --workers ${MODIS_WORKERS} && " ;;
    counts) STAGES+="python -m ${MOD}.modis counts --workers ${MODIS_WORKERS} --shard ${SHARD} && " ;;
    phenology) STAGES+="python -m ${MOD}.phenology --workers ${WORKERS} && " ;;
    cells) STAGES+="python -m ${MOD}.cells --workers ${WORKERS} && " ;;
    candidates) STAGES+="python -m ${MOD}.write_windows candidates --workers ${WORKERS} && " ;;
    windows) STAGES+="python -m ${MOD}.write_windows windows --workers ${WORKERS} ${WINDOW_ARGS} && " ;;
    *) echo "unknown step ${step}" >&2; exit 1 ;;
  esac
done
STAGES+="echo DONE"

cmd=(
  "${GANTRY:-.venv/bin/gantry}" run
  --name "${NAME}"
  --description "Snow eval dataset build (${STEPS}); 0 GPU, ${WORKERS} workers; commit ${REF:0:9}"
  --beaker-image tylerr/olmo-core-tch291cu128-2025-11-25
  --ref "${REF}" --branch "$(git branch --show-current)"
  --workspace ai2/earth-systems --budget ai2/atec-olmoearth
  --gh-token-secret "${ACCOUNT}_GITHUB_TOKEN"
  --priority urgent --gpus 0 --hostname "${HOST}"
  --min-runtime 45m --retries 2
  --weka dfive-default:/weka/dfive-default
  --env UV_CACHE_DIR=/weka/dfive-default/olmoearth_pretrain/uv-cache
  --install "uv sync --locked --all-extras"
  # zarr is not in the lock (the ERA5 mirror reader needs it).
  --post-setup "uv pip install zarr"
  --yes
  -- bash -c "${STAGES}"
)
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf '%q ' "${cmd[@]}"; echo
  exit 0
fi
GANTRY_CHECK_FOR_UPGRADES=0 "${cmd[@]}"
