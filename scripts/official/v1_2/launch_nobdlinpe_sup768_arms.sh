#!/bin/bash
# The nobdlinpe pair on the DISTILLED RELEASE CANDIDATE line (2026-08-28).
#
# launch_nobdlinpe_arms.sh put the two-flag input-stem ablation on the supstu0p1
# arms only:
#   01M14P0Q5SX8NCSQKY5MEZE08V  nobdlinpe supstu0p1 w1
#   01M14P1TK7TVJA99CC7Y38E2Y2  nobdlinpe supstu0p1 w1 stunorm
# These two are the same ablation with supstu0p1 swapped for sup768 -- i.e. on
# lin_sup768_w1, the release candidate -- so the stem question is answered on the
# line we actually ship rather than only on the supervision-only variant.
#
# ROPE BASE deliberately NOT overridden: these are the controls for arms 3 and 4 of
# launch_trope10_arms.sh, which run these same two modules with rope_mixed_base=10.
# Leaving the default 10000.0 here is what makes that pair a one-variable diff.
# See launch_trope10_arms.sh for why the constant is wrong in the first place.
#
# Everything else matches the two linked jobs: 8 GPUs, urgent, jupiter/ceres, W&B
# 2026_08_26_student_norm, set_proj_aeftrial_loop_evals (18 tasks, d128 + d64).

set -euo pipefail

cd "$(dirname "$0")/../../.."

COMMON=(ai2/jupiter
        --launch.num_gpus=8
        --launch.priority=urgent
        --launch.clusters=[ai2/jupiter,ai2/ceres]
        --trainer.callbacks.wandb.project=2026_08_26_student_norm)

V=scripts/official/v1_2
N=regbtl_v1_2_nobdlinpe_gdyn_d768_proj128lin_sup768_w1_newsampling_psuniform

# --- 1. no band dropout + linear patch stem -----------------------------------------
python $V/${N}.py \
    launch regbtl_v1_2_nobdlinpe_gdyn_d768_proj128lin_sup768_w1_newsamp_psuniform \
    "${COMMON[@]}"

# --- 2. + stunorm -------------------------------------------------------------------
python $V/${N}_stunorm.py \
    launch regbtl_v1_2_nobdlinpe_gdyn_d768_proj128lin_sup768_w1_newsamp_psuniform_stunorm \
    "${COMMON[@]}"
