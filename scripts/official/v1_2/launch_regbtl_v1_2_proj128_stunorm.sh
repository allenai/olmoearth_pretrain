#!/bin/bash
# Launch the five student-output-LayerNorm arms.
#
# Each is a one-flag mirror of an existing run -- register_projection_output_norm=True
# on the linear student, everything else byte-identical to its base -- with the
# in-loop eval set replaced by the AEF balanced trials + PASTIS on unmasked
# S1+S2+Landsat at BOTH student widths (d128 and d64). See each module's docstring.
#
#   1. sup768 w1            official lineage, [128, 64]
#   2. supstu0p1 w1         official lineage, [128, 64]
#   3. supstu0p1 w1         new maps + landsat reflectance + dsm3, [128, 64, 32, 16]
#   4. sup768 w1            new maps + landsat reflectance + dsm3, [128, 64]
#   5. sup768 w1            new maps + landsat reflectance + dsm3, [128, 64, 32, 16]
#
# NOTE the Beaker job clones the repo and checks out $GIT_REF, so the code these
# runs train is whatever is COMMITTED AND PUSHED -- not the working tree. Commit and
# push before launching, or the runs silently train the unflagged architecture.
#
# Cost: the in-loop eval job is 18 tasks (9 per width) over ~159k windows per width
# at three modality passes each, roughly 15-17 GPU-hours per round in its own Beaker
# job. Hence the 80k-step interval rather than the proj chain's 40k: at 40k
# consecutive jobs would overlap on one resumed W&B run and the overlapping writer's
# rows are dropped silently. ~8 eval rounds per 667k-step run.

set -euo pipefail

cd "$(dirname "$0")/../../.."

PROJECT=2026_08_26_student_norm
COMMON=(ai2/jupiter
        --launch.num_gpus=8
        --launch.priority=urgent
        --launch.clusters=[ai2/jupiter,ai2/ceres]
        --trainer.callbacks.wandb.project="$PROJECT")

V=scripts/official/v1_2
N=scripts/vnext/2026_07_24_new_maps

# --- 1. sup768 w1, official lineage ------------------------------------------------
python $V/regbtl_v1_2_gdyn_d768_proj128lin_sup768_w1_newsampling_psuniform_stunorm.py \
    launch regbtl_v1_2_gdyn_d768_proj128lin_sup768_w1_newsamp_psuniform_stunorm \
    "${COMMON[@]}"

# --- 2. supstu0p1 w1, official lineage ---------------------------------------------
python $V/regbtl_v1_2_gdyn_d768_proj128lin_supstu0p1_w1_newsampling_psuniform_stunorm.py \
    launch regbtl_v1_2_gdyn_d768_proj128lin_supstu0p1_w1_newsamp_psuniform_stunorm \
    "${COMMON[@]}"

# --- 3. supstu0p1 w1, new maps + refl + dsm3, mat16 --------------------------------
python $N/regbtl_v1_2_gdyn_d768_proj128lin_supstu0p1_w1_newsampling_psuniform_landsat_refl_dsm3_mat16_stunorm.py \
    launch regbtl_v1_2_gdyn_d768_proj128lin_supstu0p1_w1_psuniform_newmaps_refl_dsm3_mat16_stunorm \
    "${COMMON[@]}"

# --- 4. sup768 w1, new maps + refl + dsm3 ------------------------------------------
python $N/regbtl_v1_2_gdyn_d768_proj128lin_sup768_w1_newsampling_psuniform_landsat_refl_dsm3_stunorm.py \
    launch regbtl_v1_2_gdyn_d768_proj128lin_sup768_w1_psuniform_newmaps_refl_dsm3_stunorm \
    "${COMMON[@]}"

# --- 5. sup768 w1, new maps + refl + dsm3, mat16 -----------------------------------
python $N/regbtl_v1_2_gdyn_d768_proj128lin_sup768_w1_newsampling_psuniform_landsat_refl_dsm3_mat16_stunorm.py \
    launch regbtl_v1_2_gdyn_d768_proj128lin_sup768_w1_psuniform_newmaps_refl_dsm3_mat16_stunorm \
    "${COMMON[@]}"
