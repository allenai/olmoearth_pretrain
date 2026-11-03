#!/usr/bin/env bash
#
# Batch 2 (2026-11-03): a faithful run of the paper's unsupervised vec2vec (paper.py,
# train_paper.py), one pair of products at a time as in the paper, plus all three at once.
# Four 1-GPU jobs (saturn / ceres / jupiter), each training seeds 0-2 side by side, since the
# reference README warns that GAN convergence needs several seeds.
#   bash scripts/platonic_space/launch_paper.sh

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
export GANTRY="${GANTRY:-/Users/hadriens/Code/olmoearth_pretrain/.venv/bin/gantry}"
L=scripts/platonic_space/launch.sh

for models in oe,aef oe,tess aef,tess oe,aef,tess; do
  tag="${models//,/_}"
  NAME="plato_paper_${tag}" MIN_RUNTIME=8h \
    CMD="bash run_seeds.sh paper_${tag} '0 1 2' --models ${models}" bash "${L}"
done
