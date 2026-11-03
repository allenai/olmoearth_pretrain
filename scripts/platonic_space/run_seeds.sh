#!/usr/bin/env bash
#
# Train several seeds of one train_paper.py config side by side on one GPU: the paper's
# models are small and batch-256 steps leave the GPU mostly idle, and its README warns that
# GAN convergence needs several seeds. Fails if any seed fails.
#   bash run_seeds.sh <run prefix> "<seeds>" [train_paper.py flags...]
#   bash run_seeds.sh paper_oe_aef "0 1 2" --models oe,aef     # runs paper_oe_aef_s{0,1,2}

set -uo pipefail

prefix="$1"
seeds="$2"
shift 2
pids=()
for s in ${seeds}; do
  python -u train_paper.py --run "${prefix}_s${s}" --seed "${s}" "$@" \
    > >(sed -u "s/^/[s${s}] /") 2>&1 &
  pids+=($!)
done
rc=0
for p in "${pids[@]}"; do wait "${p}" || rc=1; done
exit "${rc}"
