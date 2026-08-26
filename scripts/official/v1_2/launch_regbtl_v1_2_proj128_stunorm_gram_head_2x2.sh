#!/bin/bash
# Fill the 2x2 {Gram weight} x {back-projection head} matrix on the stunorm base.
#
# The (gram=1, linear head) cell is the ALREADY-LAUNCHED run
#   regbtl_v1_2_gdyn_d768_proj128lin_sup768_w1_newsamp_psuniform_stunorm
# and is NOT re-run. These three fill the rest:
#
#              | linear head        | 2-layer MLP head (H=256)
#   gram = 1.0 | (the base run)     | mlpgram1
#   gram = 0.0 | gram0              | mlpgram0
#
# Verified by dry_run diff against the base: apart from these two fields the four
# configs are identical, so this is a clean factorial.
#
# WHY. Tessera v2's distillation loss is per-prefix cosine ALONE -- Gram is our
# addition -- so gram0 is the arm that matches the recipe this family follows.
# And the back-projection is a bare Linear(d, 768), which is exactly the head
# SimReg ablated and beat by 3.7 pts 1-NN / 10.2 pts linear probe with a hidden
# layer. The heads are discarded at inference; the shipped d128 is unchanged.
# The axes are expected to interact -- see any of the three module docstrings.
#
# NOTE the Beaker job clones the repo and checks out $GIT_REF, so the code these
# runs train is whatever is COMMITTED AND PUSHED -- not the working tree. The MLP
# head is new code (register_back_projection_hidden), so an unpushed tree would
# silently train three copies of the base.
#
# Cost: 8 GPUs x 667,200 steps each, ~6-7 days, matching the four runs already in
# flight in this W&B project. In-loop evals are the base's set_proj_aeftrial_loop_evals
# (18 tasks at 80k steps, own Beaker job), unchanged.

set -euo pipefail

cd "$(dirname "$0")/../../.."

PROJECT=2026_08_26_student_norm
COMMON=(ai2/jupiter
        --launch.num_gpus=8
        --launch.priority=urgent
        --launch.clusters=[ai2/jupiter,ai2/ceres]
        --trainer.callbacks.wandb.project="$PROJECT")

V=scripts/official/v1_2
B=regbtl_v1_2_gdyn_d768_proj128lin_sup768_w1_newsampling_psuniform_stunorm
N=regbtl_v1_2_gdyn_d768_proj128lin_sup768_w1_newsamp_psuniform_stunorm

# --- gram=0, linear head (the exact Tessera-v2 recipe) -----------------------------
python $V/${B}_gram0.py launch ${N}_gram0 "${COMMON[@]}"

# --- gram=1, 2-layer MLP head ------------------------------------------------------
python $V/${B}_mlpgram1.py launch ${N}_mlpgram1 "${COMMON[@]}"

# --- gram=0, 2-layer MLP head ------------------------------------------------------
python $V/${B}_mlpgram0.py launch ${N}_mlpgram0 "${COMMON[@]}"
