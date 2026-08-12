#!/bin/bash
# The psuniform tanchor+NDVI candidate re-run with its register grid on a unit sphere
# and AlphaEarth's batch-uniformity term spreading the embeddings across it. One run;
# its A/B partner is the existing
# regbtl_v1_2_gdyn_d128_wideread_regsup_ndvi_w0p1_tanchor_newsamp_psuniform, which this
# script's model config matches knob for knob apart from the two new flags.
#
# WHY: the served d128 embedding's geometry is only per-vector standardized (the
# bottleneck's LayerNorm), which leaves population-level structure unconstrained --
# measured on that candidate, 71% of a typical embedding's magnitude is a direction every
# embedding shares, and the centered effective rank is 84.5 of 128. Post-hoc arms cannot
# test whether that costs anything: they only re-express information already encoded,
# whereas the uniformity term changes what the encoder builds. Three eval-side arms
# (l2, center_l2 against a no-op baseline) already ruled out the cheap explanations --
# int8 clipping is 5% of coordinates at 0.994 round-trip cosine, repairing it fully moved
# KNN by +/-0.005 with no consistent sign, and removing the shared component post-hoc did
# not help either.
#
# Uniformity is cross-scene only (rolling the batch axis), NOT within-scene: cells of one
# scene genuinely are similar and the dense probes read that smoothness.
#
# The sphere-only ablation script (..._psuniform_sphere.py) is deliberately NOT launched
# here -- it is only worth running if this one moves something, to attribute the effect
# between the constraint and the spread.
set -e

PROJECT="2026_07_02_perceiver"
CLUSTER="ai2/jupiter"
LAUNCH_ARGS="--launch.num_gpus=8 --launch.priority=urgent --launch.clusters=[ai2/jupiter,ai2/ceres]"

# Defragment the CUDA allocator to avoid fragmentation OOMs at the larger token budget
# of the newsampling runs. Propagated to the Beaker job by internal/common.py.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

D=scripts/official/v1_2
SPHERE_UNIF="$D/regbtl_v1_2_gdyn_d128_wideread_regsup_ndvi_w0p1_tanchor_newsampling_psuniform_sphere_unif0p1.py"

python "$SPHERE_UNIF" launch \
    "regbtl_v1_2_gdyn_d128_wideread_regsup_ndvi_w0p1_tanchor_newsamp_psuniform_sphere_unif0p1" \
    "$CLUSTER" \
    $LAUNCH_ARGS \
    --trainer.callbacks.wandb.project="$PROJECT"
