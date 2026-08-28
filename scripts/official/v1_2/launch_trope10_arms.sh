#!/bin/bash
# The {stunorm} x {nobdlinpe} 2x2 on the distilled release candidate lin_sup768_w1,
# at the CORRECTED rope_mixed_base (2026-08-28).
#
# WHY. scripts/official/v1_2/base.py sets ROPE_MIXED_BASE = 10000.0, but every
# released v1.2 run trained at 10.0 -- trope_mixed_tscale_months (the backbone the
# v1.2 recipe was selected as) and the whole v1_2 size sweep both have
# rope_mixed_base: 10.0 in their saved config.json. Commit 84478b8ae flipped the
# constant 10.0 -> 10000.0 on 2026-06-25 under the message "Fix scripts to reflect
# our actual runs"; the runs it was meant to reflect used 10.0. Every regbtl_v1_2_*
# arm since has trained off-recipe. These four restate 10 on the command line rather
# than editing base.py, so the constant change is not smuggled into unrelated runs.
#
# Reference for "identical": the nobdlinpe stunorm supstu arm, Beaker
# 01M14P1TK7TVJA99CC7Y38E2Y2 -- 8 GPUs, urgent, jupiter/ceres, W&B project
# 2026_08_26_student_norm, AEF-trial in-loop evals.
#
# EVALS. The stunorm arm already calls set_proj_aeftrial_loop_evals. The release
# candidate itself does NOT -- it runs add_proj_loop_eval_beaker_job (fifty_cities +
# PASTIS at 40k on both heads) -- so arm 1 uses the _aeftrial wrapper, which changes
# the eval set and nothing else. All four then land on one readout.
#
# RUN NAMES all carry the _trope10 suffix. This is load-bearing: save_folder is
# derived from the run name and load_strategy is if_available, so reusing an existing
# name would resume that run's checkpoint instead of starting fresh.
#
# NOTE the Beaker job clones the repo and checks out $GIT_REF, so what these train is
# whatever is COMMITTED AND PUSHED -- not the working tree.

set -euo pipefail

cd "$(dirname "$0")/../../.."

COMMON=(ai2/jupiter
        --launch.num_gpus=8
        --launch.priority=urgent
        --launch.clusters=[ai2/jupiter,ai2/ceres]
        --model.encoder_config.rope_mixed_base=10
        --model.decoder_config.rope_mixed_base=10
        --trainer.callbacks.wandb.project=2026_08_26_student_norm)

V=scripts/official/v1_2
R=regbtl_v1_2_gdyn_d768_proj128lin_sup768_w1_newsampling_psuniform
N=regbtl_v1_2_nobdlinpe_gdyn_d768_proj128lin_sup768_w1_newsampling_psuniform

# --- 1. release candidate ----------------------------------------------------------
python $V/${R}_aeftrial.py \
    launch regbtl_v1_2_gdyn_d768_proj128lin_sup768_w1_newsamp_psuniform_trope10 \
    "${COMMON[@]}"

# --- 2. + stunorm ------------------------------------------------------------------
python $V/${R}_stunorm.py \
    launch regbtl_v1_2_gdyn_d768_proj128lin_sup768_w1_newsamp_psuniform_stunorm_trope10 \
    "${COMMON[@]}"

# --- 3. + no band dropout, linear patch stem ---------------------------------------
python $V/${N}.py \
    launch regbtl_v1_2_nobdlinpe_gdyn_d768_proj128lin_sup768_w1_newsamp_psuniform_trope10 \
    "${COMMON[@]}"

# --- 4. + stunorm + no band dropout, linear patch stem -----------------------------
python $V/${N}_stunorm.py \
    launch regbtl_v1_2_nobdlinpe_gdyn_d768_proj128lin_sup768_w1_newsamp_psuniform_stunorm_trope10 \
    "${COMMON[@]}"
