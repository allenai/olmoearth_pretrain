#!/bin/bash
# The ndvi x student-uniformity 2x2 on the lin_sup768_w1 distillation base.
#
#   arm         teacher change            student change              script suffix
#   ndvi        + time-cond. NDVI head    --                          ..._sup768_ndvi_w1_newsampling_psuniform.py
#   stuunif0p1  --                        + uniformity (w 0.1)        ..._sup768_w1_newsampling_psuniform_stuunif0p1.py
#   both        + time-cond. NDVI head    + uniformity (w 0.1)        ..._sup768_ndvi_w1_newsampling_psuniform_stuunif0p1.py
#
# The (off, off) cell is the completed
# regbtl_v1_2_gdyn_d768_proj128lin_sup768_w1_newsamp_psuniform run and is NOT
# re-trained: its checkpoints are swept under the same year-aligned task set for the
# baseline row. Every arm matches it knob for knob otherwise -- same teacher recipe
# (w1, newsampling psuniform, no tanchor), same lin [128, 64] student with the
# default cosine + Gram distillation, same budget/schedule/epochs(300).
#
# In-loop evals (all three arms): set_proj_earlyread_loop_evals -- the 6 early-read
# year-aligned S1+S2+Landsat probes on the d768 teacher AND their _proj128
# duplicates on the shipped student, 12 tasks at a 40k-step interval, as separate
# Beaker eval jobs. The two ethiopia/descals _knn tasks carry the AEF balanced-trial
# protocol (aeftrial_* metrics) automatically.
#
# Requires the landsat_moNN layers on weka for the year-aligned windows (the Landsat
# input degrades silently to S1+S2 where not materialized -- check coverage before
# reading a Landsat delta off the curves).
set -e

PROJECT="2026_07_02_perceiver"
CLUSTER="ai2/jupiter"
LAUNCH_ARGS="--launch.num_gpus=8 --launch.priority=urgent --launch.clusters=[ai2/jupiter,ai2/ceres]"

# Defragment the CUDA allocator (see launch_regbtl_v1_2_newsampling.sh).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

D=scripts/official/v1_2
STEM=regbtl_v1_2_gdyn_d768_proj128lin_sup768

# Combined cell first: it is the candidate if the axes compose; the single-knob
# arms exist to attribute whatever it shows.
declare -a ARMS=(
    "${STEM}_ndvi_w1_newsampling_psuniform_stuunif0p1|${STEM}_ndvi_w1_newsamp_psuniform_stuunif0p1"
    "${STEM}_ndvi_w1_newsampling_psuniform|${STEM}_ndvi_w1_newsamp_psuniform"
    "${STEM}_w1_newsampling_psuniform_stuunif0p1|${STEM}_w1_newsamp_psuniform_stuunif0p1"
)

for ARM in "${ARMS[@]}"; do
    SCRIPT="${D}/${ARM%%|*}.py"
    NAME="${ARM##*|}"
    python "$SCRIPT" launch "$NAME" "$CLUSTER" \
        $LAUNCH_ARGS \
        --trainer.callbacks.wandb.project="$PROJECT"
done
