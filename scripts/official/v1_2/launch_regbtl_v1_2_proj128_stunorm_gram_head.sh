#!/bin/bash
# Launch mlpgram1: the stunorm base with a 2-layer MLP back-projection head (H=256).
#
# One cell of the {Gram variant} x {back-projection head} matrix on the stunorm base
# (the ALREADY-RUN regbtl_v1_2_gdyn_d768_proj128lin_sup768_w1_newsamp_psuniform_stunorm
# is the (flat gram=1, linear head) cell). Apart from register_back_projection_hidden
# the two configs are identical (verified by dry_run diff). See the module's docstring
# for the motivation.
#
# NOTE the Beaker job clones the repo and checks out $GIT_REF, so the code this run
# trains is whatever is COMMITTED AND PUSHED -- not the working tree. The MLP head
# (register_back_projection_hidden) is what distinguishes this run from its base, so
# an unpushed tree would silently train a copy of the base.
#
# Cost: 8 GPUs x 667,200 steps, ~6-7 days. In-loop evals are the base's
# set_proj_aeftrial_loop_evals (18 tasks at 80k steps, own Beaker job), unchanged.

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

python $V/${B}_mlpgram1.py launch ${N}_mlpgram1 "${COMMON[@]}"
