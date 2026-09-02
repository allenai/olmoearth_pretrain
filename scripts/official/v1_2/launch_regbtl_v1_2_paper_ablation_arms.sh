#!/bin/bash
# Launch the two v1.3-report ablation arms (one-flag-off the mlpgram1 RC where
# the flag exists in both):
#
#   1. qtc  -- query-token compaction: native d128 wideread registers, no
#              student/distillation. The Compaction section's A/B partner.
#   2. nosup -- the RC with supervision_head_config=None. The Aggregation
#              section's direct-supervision ablation.
#
# The ICL ablation is deliberately NOT run: mlpgram1+ICL needs the contrastive
# train module, which refuses register_projection_dims; the report argues ICL's
# role by induction from OlmoEarth v1, with the July ic/noic 2x2 as older-recipe
# corroboration.
#
# NOTE the Beaker job clones the repo and checks out $GIT_REF, so the code these
# runs train is whatever is COMMITTED AND PUSHED -- not the working tree.

set -euo pipefail

cd "$(dirname "$0")/../../.."

PROJECT=2026_08_26_student_norm
COMMON=(ai2/jupiter
        --launch.num_gpus=8
        --launch.priority=urgent
        --launch.clusters=[ai2/jupiter,ai2/ceres]
        --trainer.callbacks.wandb.project="$PROJECT")

V=scripts/official/v1_2

python $V/regbtl_v1_2_qtc_gdyn_d128_wideread_regsup_w1_newsampling_psuniform.py \
    launch regbtl_v1_2_qtc_gdyn_d128_wideread_regsup_w1_newsamp_psuniform \
    "${COMMON[@]}"

python $V/regbtl_v1_2_nosup_gdyn_d768_proj128lin_newsampling_psuniform_stunorm_mlpgram1.py \
    launch regbtl_v1_2_nosup_gdyn_d768_proj128lin_newsamp_psuniform_stunorm_mlpgram1 \
    "${COMMON[@]}"
