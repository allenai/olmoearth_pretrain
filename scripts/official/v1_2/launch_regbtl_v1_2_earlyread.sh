#!/bin/bash
# The EARLY-READ sweep: invert the encoder/bottleneck depth split, plus the drop_path
# ablation. All arms build on
# regbtl_v1_2_gdyn_d128_wideread_regsup_ndvi_w0p1_tanchor_newsampling_psuniform (cand_ndvi),
# which is also the A/B partner -- they inherit its d128 wideread bottleneck, regsup+NDVI at
# w0p1, the year_start anchor, and the uniform-ps decorrelated sampler by importing its own
# builders, so only the named axis moves in each run.
#
# WHY: in 2026_05_18_v2_knn_lp_evals, at matched block count, four reads all taken from
# encoder layer 12 and four spread over layers 3/6/9/12 differ by +0.11 pts across 54 tasks
# -- a null. The K/V source depth was never the active ingredient; the block count was.
# Layer 3 is already a viable read source, so the deep trunk is not supplying something the
# read depends on. Every mechanism that lets training re-weight toward late depths (lrw,
# learned fsum) is neutral-to-harmful on the frozen evals in that same file, so the late-mass
# preference is a pretext-side artifact rather than an eval-side gain.
#
# THE GRID -- 6 early-read arms, two axes each controlled on the other:
#
#   depth split (vs cand_ndvi @3072):   MACs    params
#     e3_l12    3 trunk + 12 blocks     0.37x    0.61x   cheap
#     e3_l24    3 trunk + 24 blocks     0.47x    0.81x   2x the cheap arm
#     e6_l8     6 trunk +  8 blocks     0.57x    0.72x   cheap
#     e6_l16    6 trunk + 16 blocks     0.64x    0.85x   2x the cheap arm
#
#   token budget (trunk-3 only, added per the 2x2):
#     e3_l12 @6144                      0.99x            doubling the data is FREE here
#     e3_l24 @6144                      1.24x
#
# MACs are measured (scripts/tools/20251111_flops.py's counter, so attention matmuls are
# included) on one full S1+S2+Landsat window; the budget rows are at each budget's largest
# full-year shape, which is an upper bound on the average step because
# MIN_TOKENS_PER_INSTANCE (228) does not scale with the budget.
#
# PLUS one unrelated ablation launched alongside: _dp0 turns off the encoder trunk's
# stochastic depth (drop_path 0.1 -> 0.0), which has been in every run since v1.2 base
# without ever being measured. It keeps the frontier's OWN eval set
# (add_loop_eval_beaker_job), so the existing cand_ndvi curves are its control and nothing
# needs re-running. A null is the useful outcome: it retires a hyperparameter.
#
# EVAL SETS DIFFER BY DESIGN. The six early-read arms REPLACE the catalog with six
# embedding-product tasks (set_earlyread_loop_evals): pastis/ethiopia/descals year-aligned
# at S1+S2+Landsat, plus pastis_ws16_ps1_sentinel1_sentinel2_pretrain_export as the single
# task shared with cand_ndvi's existing in-loop curves. That bridge task is a tripwire, not
# a verdict -- one task sits inside the eval noise floor (LP 0.88 pts mean). For the real
# comparison, sweep cand_ndvi's saved checkpoints under the same six tasks; that also gives
# matched steps, which the in-loop path does not guarantee.
#
# PREREQUISITE: the year-aligned tasks need the landsat_moNN layers on weka
# (setup_extra_layers.py, layer set `landsat`). The Landsat input is optional in every
# model.yaml, so windows the materialize has not reached run WITHOUT it rather than failing
# -- an incomplete dataset silently degrades these into S1+S2 evals.
#
# The architecture, the budget and the eval set are baked into the scripts (not CLI
# overrides) so the Beaker eval jobs reconstruct the matching model.
set -e

PROJECT="2026_07_02_perceiver"
CLUSTER="ai2/jupiter"
LAUNCH_ARGS="--launch.num_gpus=8 --launch.priority=urgent --launch.clusters=[ai2/jupiter,ai2/ceres]"

# Defragment the CUDA allocator to avoid fragmentation OOMs, as in the other newsampling
# launchers. Propagated to the Beaker job by internal/common.py.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

D=scripts/official/v1_2
P=regbtl_v1_2_earlyread
S=d128_ndvi_tanchor_newsampling_psuniform

# --- depth split, token_budget 3072 (rank_microbatch_size 64, inherited) ---------------
python "$D/${P}_e3_l12_${S}.py"  launch "${P}_e3_l12_${S}"  "$CLUSTER" \
    $LAUNCH_ARGS --trainer.callbacks.wandb.project="$PROJECT"

python "$D/${P}_e3_l24_${S}.py"  launch "${P}_e3_l24_${S}"  "$CLUSTER" \
    $LAUNCH_ARGS --trainer.callbacks.wandb.project="$PROJECT"

python "$D/${P}_e6_l8_${S}.py"   launch "${P}_e6_l8_${S}"   "$CLUSTER" \
    $LAUNCH_ARGS --trainer.callbacks.wandb.project="$PROJECT"

python "$D/${P}_e6_l16_${S}.py"  launch "${P}_e6_l16_${S}"  "$CLUSTER" \
    $LAUNCH_ARGS --trainer.callbacks.wandb.project="$PROJECT"

# --- token_budget 6144 (rank_microbatch_size dropped to 32) ----------------------------
# If either OOMs, drop to 16: microbatch affects ONLY memory, not tokens/step, the loss, or
# the LR schedule. e3_l24 is the tighter of the two -- 35 read blocks each materialize a
# k and a v projection over the full token array, so it carries roughly the base's
# activation memory rather than e3_l12's ~half.
python "$D/${P}_e3_l12_${S}_tb6144.py" launch "${P}_e3_l12_${S}_tb6144" "$CLUSTER" \
    $LAUNCH_ARGS --trainer.callbacks.wandb.project="$PROJECT"

python "$D/${P}_e3_l24_${S}_tb6144.py" launch "${P}_e3_l24_${S}_tb6144" "$CLUSTER" \
    $LAUNCH_ARGS --trainer.callbacks.wandb.project="$PROJECT"

# --- drop_path ablation (separate question, frontier eval set) -------------------------
DP0="$D/regbtl_v1_2_gdyn_d128_wideread_regsup_ndvi_w0p1_tanchor_newsampling_psuniform_dp0.py"
python "$DP0" launch "regbtl_v1_2_gdyn_d128_wideread_regsup_ndvi_w0p1_tanchor_newsamp_psuniform_dp0" "$CLUSTER" \
    $LAUNCH_ARGS --trainer.callbacks.wandb.project="$PROJECT"
