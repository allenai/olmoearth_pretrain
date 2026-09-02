#!/bin/bash
# Launch the three input-stem ablation arms (2026-08-28).
#
# Every arm cuts the SAME two flags from the pixel -> token stem and changes nothing
# else, so each has an existing run as its control:
#
#   * band dropout OFF   -- base.py's band_dropout_rate=0.2 / random_band_dropout=True
#                           on S2 + Landsat, a train-only augmentation that is off at
#                           eval.
#   * linear patch stem  -- patch_embed_hidden_sizes=None instead of [64], i.e. the
#                           original single nn.Linear projection instead of the
#                           per-pixel Linear->ReLU MLP before patchification.
#
#   1. nobdlinpe supstu0p1 w1              control: ..._supstu0p1_w1_newsamp_psuniform
#   2. nobdlinpe supstu0p1 w1 stunorm      control: ..._supstu0p1_w1_newsamp_psuniform_stunorm
#   3. v1_2_base_linear                    control: trope_mixed_tscale_months
#
# EVALS. Arms 1 and 2 run the embedding evals -- AEF balanced trials + year-aligned
# PASTIS on unmasked S1+S2+Landsat at both student widths (d128, d64), 18 tasks in
# their own Beaker job at an 80k-step interval -- matching the 2026_08_26_student_norm
# arms. That is the eval set arm 2's control already ran; it is NOT what arm 1's
# control ran (that one used the early-read set), so arm 1's comparison is against the
# student_norm project, not against its own lineage's curves. Arm 3 keeps the v1.1
# base's in-process eval set untouched, so trope_mixed_tscale_months is a like-for-like
# control.
#
# Arm 3 restates the rope overrides its control passed on the command line
# (rope_mixed_base=10, rope_temporal_coordinate_scale=0.0333) even though the module
# already sets both, so the two runs differ ONLY in the stem flags -- the control used
# 0.0333, not the module's 1/30.
#
# NOTE the Beaker job clones the repo and checks out $GIT_REF, so the code these runs
# train is whatever is COMMITTED AND PUSHED -- not the working tree. Commit and push
# before launching, or the runs silently train the unflagged architecture.

set -euo pipefail

cd "$(dirname "$0")/../../.."

COMMON=(ai2/jupiter
        --launch.num_gpus=8
        --launch.priority=urgent
        --launch.clusters=[ai2/jupiter,ai2/ceres])

V=scripts/official/v1_2
T=scripts/vnext/temporal_rope

# --- 1. supstu0p1 w1, no band dropout + linear patch stem --------------------------
python $V/regbtl_v1_2_nobdlinpe_gdyn_d768_proj128lin_supstu0p1_w1_newsampling_psuniform.py \
    launch regbtl_v1_2_nobdlinpe_gdyn_d768_proj128lin_supstu0p1_w1_newsamp_psuniform \
    "${COMMON[@]}" \
    --trainer.callbacks.wandb.project=2026_08_26_student_norm

# --- 2. supstu0p1 w1 stunorm, no band dropout + linear patch stem ------------------
python $V/regbtl_v1_2_nobdlinpe_gdyn_d768_proj128lin_supstu0p1_w1_newsampling_psuniform_stunorm.py \
    launch regbtl_v1_2_nobdlinpe_gdyn_d768_proj128lin_supstu0p1_w1_newsamp_psuniform_stunorm \
    "${COMMON[@]}" \
    --trainer.callbacks.wandb.project=2026_08_26_student_norm

# --- 3. v1.2 backbone baseline, no band dropout + linear patch stem ----------------
# Keeps the v1.1 base's own W&B project (set in scripts/official/v1_1/base.py), which
# is where its control trope_mixed_tscale_months logged.
python $T/v1_2_base_linear.py \
    launch v1_2_base_linear \
    "${COMMON[@]}" \
    --model.encoder_config.rope_mixed_base=10 \
    --model.decoder_config.rope_mixed_base=10 \
    --model.encoder_config.rope_temporal_coordinate_scale=0.0333 \
    --model.decoder_config.rope_temporal_coordinate_scale=0.0333
