#!/usr/bin/env bash
#
# Batch 1 (2026-10-28): the pair set, the baselines, and five translators.
#   HOST=<jupiter host for the CPU job> bash scripts/platonic_space/launch_batch1.sh
#
# - plato_pairs2024 (CPU): build_pairs.py, then the concat / concat+PCA baselines.
# - five 1-GPU saturn runs, each waiting for the pair set, then train + export:
#     paired_l128          main line: rec + translation + InfoNCE alignment, 128-d latent
#     paired_l128_noalign  no latent alignment (does the shared latent need it?)
#     paired_l256          256-d latent
#     self_l128            control: same network, reconstruction only, no other product
#     unpaired_l128        the paper's recipe: LSGAN on outputs + latent, cycle 10, VSP 1,
#                          disjoint pixels per product (fidelity side experiment)

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
export GANTRY="${GANTRY:-/Users/hadriens/Code/olmoearth_pretrain/.venv/bin/gantry}"
L=scripts/platonic_space/launch.sh
PAIRS=/weka/dfive-default/hadriens/platonic_space/pairs2024
WAIT="for i in \$(seq 1 180); do [ -f ${PAIRS}/test.npz ] && break; sleep 60; done"

GPUS=0 HOST="${HOST:?set HOST (pick_nodes.sh ai2/jupiter 1)}" NAME=plato_pairs2024 \
  CMD="python build_pairs.py ${PAIRS} && python export.py --baselines" bash "${L}"

run() {  # name, train flags
  NAME="plato_$1" CMD="${WAIT}; python train.py --run $1 ${2:-} && python export.py --run $1" bash "${L}"
}
run paired_l128
run paired_l128_noalign "--w_align 0"
run paired_l256 "--latent 256"
run self_l128 "--mode self"
run unpaired_l128 "--mode unpaired --w_cyc 10 --w_vsp 1 --batch 1024 --lr 1e-4 --steps 50000 --eval_every 5000"
