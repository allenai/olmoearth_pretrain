#!/usr/bin/env bash
# One seagrass run inside a gantry job: fine-tune/probe, then polygon eval of best.ckpt.
# Usage: seagrass_job.sh <run name>   (configs/<run>.yaml and configs/<run>_polyeval.yaml)
set -euo pipefail

RUN="$1"
HERE="/weka/dfive-default/agent/piperbots/2026-10-29-piperw-bathymetry-direct-supervision/downstream/seagrass"
ROOT="/weka/dfive-default/piperw"
CKPT="${ROOT}/rslearn_projects/projects/seagrass_bathymetry/${RUN}/best.ckpt"

# Reproduce rslearn_projects/.venv, where the bundles were verified: its frozen packages
# (constraints.txt, incl. rslearn from PyPI), then the two olmoearth trees it has as editable
# installs, without re-resolving their declared deps (the
# olmoearth_pretrain tree now asks for torch 2.9, the venv runs 2.7.1; newer jsonargparse
# breaks rslearn's CLI).
# --no-deps: the venv itself is not pip-consistent (e.g. gcsfs 2026.3.0 with fsspec 2025.12.0),
# so install exactly its frozen versions without resolving.
pip install --no-deps -r "${HERE}/constraints.txt" --extra-index-url https://download.pytorch.org/whl/cu128
for r in olmoearth_pretrain olmoearth_pretrain_minimal; do
  pip install --no-deps -e "${ROOT}/${r}"
done
python -c "import importlib.metadata as m; print('[versions]', *(f'{p}={m.version(p)}' for p in ('rslearn', 'jsonargparse', 'lightning', 'torch')))"

for r in olmoearth_pretrain olmoearth_pretrain_minimal rslearn_projects; do
  echo "[versions] ${r} $(git -C "${ROOT}/${r}" rev-parse HEAD)"
done

cd "${ROOT}/rslearn_projects"
rslearn model fit --config "${HERE}/configs/${RUN}.yaml"
rslearn model test --config "${HERE}/configs/${RUN}_polyeval.yaml" --ckpt_path "${CKPT}"
