#!/bin/bash
# Launch the four open-set spatial-latent supervised pretraining runs:
#   {open_set_only, open_set_osm} x {register_dim 768, register_dim 128}
#
# All inherit the register-bottleneck (Perceiver) wideread recipe with the
# decorrelated shape sampler (1fwd, fused AdamW, projection-only target,
# DDP + bf16, in-loop evals as separate Beaker jobs), the register-grid map
# supervision, and the supervised open-set probe over the spatial latent.
#
# Set open_set_base.OPEN_SET_H5_DIR to the built open-set H5 directory first.
set -e

PROJECT="2026_07_23_open_set_latent"
LAUNCH_ARGS="--launch.num_gpus=8 --launch.priority=high --launch.clusters=[ai2/jupiter]"

for script in open_set_only_d768 open_set_only_d128 open_set_osm_d768 open_set_osm_d128; do
    python scripts/official/v1_2/$script.py launch $script ai2/jupiter \
        $LAUNCH_ARGS \
        --trainer.callbacks.wandb.project="$PROJECT"
done
