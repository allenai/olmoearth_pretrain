#!/bin/bash
# d128 w0p1 register-bottleneck runs with the decorrelated (grid, timestep) shape
# sampler. Two runs -- regsup and regsup+latlon -- matching the committed w0p1
# baselines except for the dataloader shape sampling (independent timestep axis with
# temporal bias, a token floor, ps=1 oversampling, larger grids, maps excluded from
# the budget) and rank_microbatch_size 64->32 to fit the ~6144 token budget. All
# knobs live in regbtl_v1_2_newsampling_common.py; the architecture is baked into the
# scripts (not CLI overrides) so the Beaker eval jobs reconstruct the matching model.
set -e

PROJECT="2026_07_02_perceiver"
CLUSTER="ai2/jupiter"
LAUNCH_ARGS="--launch.num_gpus=8 --launch.priority=urgent --launch.clusters=[ai2/jupiter,ai2/ceres]"

# Defragment the CUDA allocator: the ~6144 token budget pushes per-instance encoder
# self-attention (flash off -> O(seq^2) scores) close to the 80GB ceiling, and the
# earlier OOM was largely fragmentation (6+ GiB reserved-but-unallocated). Propagated
# to the Beaker job by internal/common.py.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REGSUP="scripts/official/v1_2/regbtl_v1_2_gdyn_d128_wideread_regsup_w0p1_newsampling.py"
REGSUP_LATLON="scripts/official/v1_2/regbtl_v1_2_gdyn_d128_wideread_regsup_latlon_w0p1_newsampling.py"

python "$REGSUP" launch "regbtl_v1_2_gdyn_d128_wideread_regsup_w0p1_newsamp_v5_b32" "$CLUSTER" \
    $LAUNCH_ARGS \
    --trainer.callbacks.wandb.project="$PROJECT"

python "$REGSUP_LATLON" launch "regbtl_v1_2_gdyn_d128_wideread_regsup_latlon_w0p1_newsamp_v5_b32" "$CLUSTER" \
    $LAUNCH_ARGS \
    --trainer.callbacks.wandb.project="$PROJECT"
