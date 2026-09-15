#!/bin/bash
# Launch the two open-set supervised pretraining runs:
#   1. open_set_osm  -- osm_sampling + open-set supervised dataset.
#   2. open_set_only -- open-set supervised dataset only.
#
# Both inherit the v1.2-faster stack (projection-only target, DDP + bf16, in-loop
# evals as separate Beaker jobs) and add the supervised open-set probe loss.
#
# Set open_set_base.OPEN_SET_H5_DIR to the built open-set H5 directory first.
set -e

PROJECT="2026_07_14_open_set_supervised"
LAUNCH_ARGS="--launch.num_gpus=8 --launch.priority=high --launch.clusters=[ai2/jupiter]"

python scripts/official/v1_2/open_set_osm.py launch open_set_osm ai2/jupiter \
    $LAUNCH_ARGS \
    --trainer.callbacks.wandb.project="$PROJECT"

python scripts/official/v1_2/open_set_only.py launch open_set_only ai2/jupiter \
    $LAUNCH_ARGS \
    --trainer.callbacks.wandb.project="$PROJECT"
