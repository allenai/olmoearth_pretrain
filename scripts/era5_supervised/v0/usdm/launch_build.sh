#!/usr/bin/env bash
#
# Build the USDM eval dataset in one CPU-only Beaker job (stages of
# olmoearth_pretrain/dataset_creation/usdm): weekly map download (resumes),
# rasterization onto the CONUS ERA5-Land grid, cell sampling + ERA5-Land series
# from the weka mirror, candidates (targets + splits), then the rslearn windows.
# Every stage is resumable or idempotent.
#
# Usage (from the repo root; the commit must be pushed):
#   HOST=jupiter-cs-aus-192.reviz.ai2.in bash scripts/era5_supervised/v0/usdm/launch_build.sh
#   STEPS="windows" HOST=... bash ...   # only some stages: any of download rasterize cells candidates windows
#   DRY_RUN=1 HOST=... bash ...         # print the gantry command only
#
# CPU-only: no --cpus / --memory (either would land the job on a GPU slot);
# one job per node, at most 64 workers (WORKERS, default 48).

set -euo pipefail

HOST="${HOST:?set HOST to a jupiter hostname (pick_nodes.sh ai2/jupiter 1)}"
STEPS="${STEPS:-download rasterize cells candidates}"
WORKERS="${WORKERS:-48}"
NAME="${NAME:-usdm_build}"
WINDOW_ARGS="${WINDOW_ARGS:-}"
ACCOUNT=$(beaker account whoami --format text | awk 'NR==2{print $2}')
REF=$(git rev-parse HEAD)
MOD=olmoearth_pretrain.dataset_creation.usdm

STAGES=""
for step in ${STEPS}; do
  case "${step}" in
    download) STAGES+="python -m ${MOD}.download --workers 8 && " ;;
    rasterize) STAGES+="python -m ${MOD}.rasterize --workers ${WORKERS} && " ;;
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
  --description "USDM eval dataset build (${STEPS}); 0 GPU, ${WORKERS} workers; commit ${REF:0:9}"
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
