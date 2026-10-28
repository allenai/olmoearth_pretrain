#!/usr/bin/env bash
#
# Run one platonic_space command as a Beaker job: 1 GPU on ai2/saturn by default
# (hadriens's standing allowance for this project: up to 5 saturn GPU jobs, urgent),
# or CPU-only with GPUS=0 (then pick a node: pick_nodes.sh ai2/jupiter 1, HOST=...).
#
# Usage (repo root; the commit must be pushed):
#   NAME=plato_paired_l128 CMD="python scripts/platonic_space/train.py --run paired_l128" \
#     bash scripts/platonic_space/launch.sh
#   DRY_RUN=1 ...    # print the gantry command only
#
# The scripts import each other, so CMD runs from scripts/platonic_space.

set -euo pipefail

NAME="${NAME:?set NAME}"
CMD="${CMD:?set CMD to the command to run from scripts/platonic_space}"
GPUS="${GPUS:-1}"
CLUSTER="${CLUSTER:-ai2/saturn}"
ACCOUNT=$(beaker account whoami --format text | awk 'NR==2{print $2}')
REF=$(git rev-parse HEAD)

where=(--cluster "${CLUSTER}" --gpus "${GPUS}")
if [[ "${GPUS}" == "0" ]]; then
  where=(--gpus 0 --hostname "${HOST:?GPUS=0 needs HOST (pick_nodes.sh ai2/jupiter 1)}")
else
  where+=(--shared-memory 96GiB)
fi

cmd=(
  "${GANTRY:-.venv/bin/gantry}" run
  --name "${NAME}"
  --description "platonic_space (vec2vec over AEF/OE v1.3/TESSERA v2): ${CMD}; ${GPUS} GPU; commit ${REF:0:9}"
  --beaker-image tylerr/olmo-core-tch291cu128-2025-11-25
  --ref "${REF}" --branch "$(git branch --show-current)"
  --workspace ai2/earth-systems --budget ai2/atec-olmoearth
  --gh-token-secret "${ACCOUNT}_GITHUB_TOKEN"
  --env-secret "WANDB_API_KEY=${ACCOUNT}_WANDB_API_KEY"
  --priority urgent "${where[@]}"
  --min-runtime "${MIN_RUNTIME:-2h}" --retries 3
  --weka dfive-default:/weka/dfive-default
  --env UV_CACHE_DIR=/weka/dfive-default/olmoearth_pretrain/uv-cache
  --install "uv sync --locked --all-extras"
  --yes
  -- bash -c "cd scripts/platonic_space && ${CMD}"
)
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf '%q ' "${cmd[@]}"; echo
  exit 0
fi
GANTRY_CHECK_FOR_UPGRADES=0 "${cmd[@]}" 2>&1 | grep -E 'beaker.org/ex|rror' || true
