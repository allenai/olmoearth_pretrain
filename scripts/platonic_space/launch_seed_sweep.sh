#!/usr/bin/env bash
#
# Batch 3 (2026-11-18): seeds 3-29 of each batch-2 pair (paper.py / train_paper.py, unchanged),
# capped at 200k steps, so every pair has 30 seeds; the paper's seed study used 15 and found
# only 3 of 15 converging between unrelated text models. One 1-GPU job per argument, each
# training its seeds side by side (9 fit on one 80 GB GPU). hadriens's allowance is 5 GPU jobs
# at a time, so launch the nine chunks in two rounds.
#   bash scripts/platonic_space/launch_seed_sweep.sh oe,aef:3-11 oe,tess:3-11 ...

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
export GANTRY="${GANTRY:-/Users/hadriens/Code/olmoearth_pretrain/.venv/bin/gantry}"

for chunk in "$@"; do
  models="${chunk%%:*}"
  range="${chunk#*:}"
  tag="${models//,/_}"
  seeds=$(seq -s ' ' "${range%-*}" "${range#*-}")
  NAME="plato_seeds_${tag}_${range/-/_}" MIN_RUNTIME=8h \
    CMD="bash run_seeds.sh paper_${tag} '${seeds}' --models ${models} --steps 200000" \
    bash scripts/platonic_space/launch.sh
done
