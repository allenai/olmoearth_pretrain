#!/bin/bash
# Launch the v1.3 bug-hunt arms (full 300-epoch / 667,200-step schedule).
#
#   arm_aug       -- control_pixtgt + flip/rotate augmentation on the dataloader
#   arm_center    -- control_pixtgt + per-sample centered latent-MIM targets
#   arm_combined  -- arm_aug + arm_center + the WorldCover class fix
#   arm_jepa      -- control_pix512 with an EMA encoder (final-layer) target
#
# The controls are the existing runs v1_3_rc_ld1_pixtgt_pix512 and v1_3_rc_ld1_pix512
# (control_pixtgt.py / control_pix512.py reproduce them on this branch; see
# diff_against_trained.py). Run names are v1_3_cf_<arm without the arm_ prefix>.
#
# Launch requires a clean tree pushed to the remote; run from the repo root.
set -e

LAUNCH_ARGS="--launch.num_gpus=8 --launch.priority=urgent --launch.clusters=[ai2/jupiter,ai2/ceres] --launch.min_runtime=8h"

if [ "$#" -eq 0 ]; then
    echo "usage: $0 {arm_aug|arm_center|arm_combined|arm_jepa} [...]"
    exit 1
fi

for arm in "$@"; do
    python scripts/official/v1_3/claude_findings/$arm.py launch "v1_3_cf_${arm#arm_}" ai2/jupiter $LAUNCH_ARGS
done
