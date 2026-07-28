#!/bin/bash
# Launch the two open-set mid-training runs: {open_set_mid_d768, open_set_mid_d128}.
#
# Each initializes from a finished regbtl wideread+regsup(w0p1) pretraining
# checkpoint (see open_set_mid_base.LOAD_PATHS), trains only the open-set probe
# for the first 40k steps (backbone frozen), then unfreezes everything at a low
# backbone LR. Full osm_sampling + open-set concat dataset throughout.
#
# OE_LOAD_SKIP_MISMATCHED_KEYS=1 is REQUIRED: the checkpoints lack the probe
# params, so the partial-load escape hatch keeps them freshly initialized (and
# skips the stale optimizer state). internal/common.py propagates it into the
# Beaker jobs.
set -e

export OE_LOAD_SKIP_MISMATCHED_KEYS=1
LAUNCH_ARGS="--launch.num_gpus=8 --launch.priority=urgent --launch.clusters=[ai2/jupiter,ai2/ceres]"

for script in open_set_mid_d768 open_set_mid_d128; do
    python scripts/official/v1_2/$script.py launch $script ai2/jupiter $LAUNCH_ARGS
done
