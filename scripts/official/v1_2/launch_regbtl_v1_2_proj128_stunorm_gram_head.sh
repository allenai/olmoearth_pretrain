#!/bin/bash
# Fill the 2x3 {Gram variant} x {back-projection head} matrix on the stunorm base.
#
# The (flat gram=1, linear head) cell is the ALREADY-RUNNING
#   regbtl_v1_2_gdyn_d768_proj128lin_sup768_w1_newsamp_psuniform_stunorm
# and is NOT re-run. These five fill the rest:
#
#                              | linear head   | 2-layer MLP head (H=256)
#   flat gram   1.0 / within 0 | (base run)    | mlpgram1
#   no gram     0.0 / within 0 | gram0         | mlpgram0
#   gramonly    0.0 / within 1 | gramonly      | mlpgramonly
#
# Verified by dry_run diff against the base: apart from the gram weights and
# register_back_projection_hidden, all six configs are identical. Clean factorial.
#
# WHY THESE TWO AXES.
#
# Gram is OURS, not Tessera v2's -- their distillation loss is per-prefix cosine
# alone -- so the middle row is the arm that matches the recipe this family
# follows, and it has never been run: the previous 16-arm sweep varied gram SCOPE
# and never set the weight to zero. Gramonly (block-diagonal, 100% within-scene
# pairs, against the flat term's ~1/B) is included because it was that sweep's
# least-bad variant and because dense probes discriminate within a scene.
#
# The back-projection is a bare Linear(d, 768) -- exactly the head SimReg ablated
# and beat by 3.7 pts 1-NN / 10.2 pts linear probe with one hidden layer, ~94% of
# the gain from the first layer alone. The heads are discarded at inference, so
# the shipped d128 architecture and serving cost are unchanged.
#
# The axes should INTERACT, in the direction that makes the both-changes corner
# the risky cell rather than the best one: cosine constrains the student only
# THROUGH the head, so a stronger head loosens its grip on the raw prefix and
# leaves Gram as the only term holding the served embedding's geometry. Prediction
# under test: gram matters more with the MLP than it ever did with the Linear.
#
# READOUT: _proj128 / _proj64 in-loop probes are the outcome,
# projection/distill_cosine_d{128,64} the diagnostic. A head that overfits shows
# as the cosine loss falling while the probes do not move.
#
# NOTE the Beaker job clones the repo and checks out $GIT_REF, so the code these
# runs train is whatever is COMMITTED AND PUSHED -- not the working tree. Both the
# MLP head (register_back_projection_hidden) and the within-scene Gram term
# (projection_distill_gram_within_weight, restored from 25d502fe8) are new code
# here, so an unpushed tree would silently train five copies of the base.
#
# Cost: 8 GPUs x 667,200 steps each, ~6-7 days, on top of the four runs already in
# flight in this W&B project. In-loop evals are the base's
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

for cell in gram0 gramonly mlpgram1 mlpgram0 mlpgramonly; do
    echo "=== launching ${N}_${cell}"
    python $V/${B}_${cell}.py launch ${N}_${cell} "${COMMON[@]}"
done
