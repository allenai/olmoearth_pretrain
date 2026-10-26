#!/usr/bin/env bash
#
# Build the FLUXNET eval dataset in one CPU-only Beaker job (stages 1-4 of
# olmoearth_pretrain/dataset_creation/fluxnet): catalogue + download (resumes,
# re-validates what is on weka), daily targets, ERA5-Land series from the weka
# mirror, then the rslearn windows. Every stage is resumable or idempotent, so a
# retried job just redoes the cheap parts.
#
# Usage (from the repo root; the commit must be pushed):
#   HOST=jupiter-cs-aus-192.reviz.ai2.in bash scripts/era5_supervised/v0/fluxnet/launch_build.sh
#   STEPS="windows" HOST=... bash ...   # only some stages: any of catalog targets era5 windows
#   DRY_RUN=1 HOST=... bash ...         # print the gantry command only
#
# CPU-only: no --cpus / --memory (either would land the job on a GPU slot);
# one job per node, at most 64 workers (WORKERS, default 48).

set -euo pipefail

HOST="${HOST:?set HOST to a jupiter hostname (pick_nodes.sh ai2/jupiter 1)}"
STEPS="${STEPS:-catalog targets era5 windows}"
WORKERS="${WORKERS:-48}"
NAME="${NAME:-fluxnet_build}"
ACCOUNT=$(beaker account whoami --format text | awk 'NR==2{print $2}')
REF=$(git rev-parse HEAD)
MOD=olmoearth_pretrain.dataset_creation.fluxnet

STAGES=""
for step in ${STEPS}; do
  case "${step}" in
    catalog) STAGES+="python -m ${MOD}.download all --workers 8 && " ;;
    targets) STAGES+="python -m ${MOD}.targets --workers ${WORKERS} && " ;;
    era5) STAGES+="python -m ${MOD}.era5 extract --workers ${WORKERS} && " ;;
    windows) STAGES+="python -m ${MOD}.write_windows all --workers ${WORKERS} && " ;;
    *) echo "unknown step ${step}" >&2; exit 1 ;;
  esac
done
STAGES+="echo DONE"

cmd=(
  .venv/bin/gantry run
  --name "${NAME}"
  --description "FLUXNET eval dataset build (${STEPS}); 0 GPU, ${WORKERS} workers; commit ${REF:0:9}"
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
