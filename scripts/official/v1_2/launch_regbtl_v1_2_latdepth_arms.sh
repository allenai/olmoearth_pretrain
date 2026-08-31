#!/bin/bash
# Launch the four register_latent_depth arms: {ld2, ld3} x {plain, stunorm}.
#
# Depth prices the perceiver head's block count on the RC recipe: the head costs
# +79.8G MACs (1.51x v1.2 Base) at 4 blocks, scaling linearly (~19.9G/block pair;
# ld3 = 1.38x, ld2 = 1.25x). Prior 3-vs-4-read evidence (-1pt) is from the mdr
# family and may not transfer. 4-block anchors: the RC run (plain) and
# regbtl_v1_2_gdyn_d768_proj128lin_sup768_w1_newsamp_psuniform_stunorm (stunorm,
# same project). See each module's docstring.
#
# NOTE the Beaker job clones the repo and checks out $GIT_REF, so the code these
# runs train is whatever is COMMITTED AND PUSHED -- not the working tree. Commit and
# push before launching, or the runs silently train the 4-block architecture.
#
# In-loop evals are the AEF-trial set at 80k (see launch_regbtl_v1_2_proj128_stunorm.sh
# for the cost arithmetic and why not 40k).

set -euo pipefail

cd "$(dirname "$0")/../../.."

PROJECT=2026_08_26_student_norm
COMMON=(ai2/jupiter
        --launch.num_gpus=8
        --launch.priority=urgent
        --launch.clusters=[ai2/jupiter,ai2/ceres]
        --trainer.callbacks.wandb.project="$PROJECT")

V=scripts/official/v1_2

for LD in 2 3; do
    # --- plain RC line -------------------------------------------------------------
    python $V/regbtl_v1_2_gdyn_d768_ld${LD}_proj128lin_sup768_w1_newsampling_psuniform.py \
        launch regbtl_v1_2_gdyn_d768_ld${LD}_proj128lin_sup768_w1_newsamp_psuniform \
        "${COMMON[@]}"

    # --- stunorm line --------------------------------------------------------------
    python $V/regbtl_v1_2_gdyn_d768_ld${LD}_proj128lin_sup768_w1_newsampling_psuniform_stunorm.py \
        launch regbtl_v1_2_gdyn_d768_ld${LD}_proj128lin_sup768_w1_newsamp_psuniform_stunorm \
        "${COMMON[@]}"
done
