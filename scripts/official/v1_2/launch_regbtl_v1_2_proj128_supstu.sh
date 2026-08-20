#!/bin/bash
# The lin supstu0p1 ladder (2026-08-20): student supervision at 0.1x on the
# lin_sup768_w1 distillation base, stacked one axis at a time.
#
#   arm                              adds
#   supstu0p1                        student heads at 0.1x (the dose-response cell;
#                                    supboth = 1.0x flipped Descals kNN vs AEF but
#                                    overpaid ~0.6 pts mean elsewhere)
#   supstu0p1_ndvi                   + time-conditioned NDVI head (both widths)
#   supstu0p1_ndvi_cloudmask0p5      + cloud skip at 0.5 (requested despite the
#                                    threshold sweep's mid-training null)
#   supstu0p1_ndvi_cm0p5_stuunif0p1  + student-only uniformity (w 0.1)
#
# Teacher recipe identical to lin_sup768_w1 everywhere (w1, newsampling psuniform,
# no tanchor). In-loop evals: proj-earlyread 12-task set, student tasks first.
set -e

PROJECT="2026_07_02_perceiver"
CLUSTER="ai2/jupiter"
LAUNCH_ARGS="--launch.num_gpus=8 --launch.priority=urgent --launch.clusters=[ai2/jupiter,ai2/ceres]"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

D=scripts/official/v1_2
STEM=regbtl_v1_2_gdyn_d768_proj128lin_supstu0p1

# Fullest arm first, then peel axes off.
for SUFFIX in \
    "_ndvi_w1_newsampling_psuniform_cloudmask0p5_stuunif0p1|_ndvi_w1_newsamp_psuniform_cloudmask0p5_stuunif0p1" \
    "_ndvi_w1_newsampling_psuniform_cloudmask0p5|_ndvi_w1_newsamp_psuniform_cloudmask0p5" \
    "_ndvi_w1_newsampling_psuniform|_ndvi_w1_newsamp_psuniform" \
    "_w1_newsampling_psuniform|_w1_newsamp_psuniform"; do
    SCRIPT="${D}/${STEM}${SUFFIX%%|*}.py"
    NAME="${STEM}${SUFFIX##*|}"
    python "$SCRIPT" launch "$NAME" "$CLUSTER" \
        $LAUNCH_ARGS \
        --trainer.callbacks.wandb.project="$PROJECT"
done
