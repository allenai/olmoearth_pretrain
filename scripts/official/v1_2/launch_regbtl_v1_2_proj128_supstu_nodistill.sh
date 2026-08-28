#!/bin/bash
# Launch the supervision-only student: supstu0p1 with the distillation loss deleted.
#
# One arm, one change. It mirrors the in-flight
#   regbtl_v1_2_gdyn_d768_proj128lin_supstu0p1_w1_newsamp_psuniform
# with projection_distill_cosine_weight = projection_distill_gram_weight = 0, so the
# detached d128 student is trained by its own supervision heads alone and never sees
# the teacher's registers as a target. Verified by a normalized dry_run diff against
# the parent: those two floats are the ONLY fields that differ.
#
# WHY. Every arm of this family has assumed the student needs a teacher to imitate;
# the 2x3 gram x head matrix is testing the SHAPE of that objective. This tests
# whether it is load-bearing at all. The teacher is untouched (registers are detached
# before the student reads them), so d768 is identical in both runs and the contrast
# is purely the map into the shipped 128 dims. _gram0 removes one of the two terms;
# no run has ever scored a student with no teacher signal whatsoever.
#
# The 0.1x student supervision weight is KEPT rather than raised back to 1.0 now that
# it is the only loss: decoupled AdamW makes the per-parameter update essentially
# invariant to a constant loss scale, and holding it keeps this one-thing-changed.
#
# NOTE the Beaker job clones the repo and checks out $GIT_REF, so it trains whatever
# is COMMITTED AND PUSHED, not the working tree. The flags live in the new module, so
# an unpushed tree would silently train a second copy of the parent.
#
# Extra args are forwarded, which is how this was actually launched on 2026-08-28:
#   ./launch_regbtl_v1_2_proj128_supstu_nodistill.sh --launch.allow_dirty=true
# An UNRELATED work-in-progress edit (the FT sweep's --task-names flag) was sitting
# in the tree and tripped olmo-core's dirty-tree guard. That guard protects against
# launching code you have not pushed; the module this run trains WAS committed and
# pushed, and the Beaker job checks out $GIT_REF regardless, so the dirty file never
# reaches the job. Do not add the flag to COMMON -- with a genuinely relevant dirty
# tree it would silently train the pushed version instead.
#
# In-loop evals: the parent's proj-earlyread 12-task set on both heads at 40k, so the
# curves overlay the parent's directly. projection/distill_* metrics are absent here
# by construction -- the terms are not computed.

set -euo pipefail

cd "$(dirname "$0")/../../.."

PROJECT=2026_07_02_perceiver
COMMON=(ai2/jupiter
        --launch.num_gpus=8
        --launch.priority=urgent
        --launch.clusters=[ai2/jupiter,ai2/ceres]
        --trainer.callbacks.wandb.project="$PROJECT")

python scripts/official/v1_2/regbtl_v1_2_gdyn_d768_proj128lin_supstu0p1_w1_newsampling_psuniform_nodistill.py \
    launch regbtl_v1_2_gdyn_d768_proj128lin_supstu0p1_w1_newsamp_psuniform_nodistill \
    "${COMMON[@]}" "$@"
