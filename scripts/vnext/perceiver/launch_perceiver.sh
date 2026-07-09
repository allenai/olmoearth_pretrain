#!/bin/bash
# Set-Latent Perceiver (SLP) pretraining launch: nano + base (ViT-B) presets.
#
# The SLP is a self-contained SSL encoder (docs/perceiver_encoder_spec.md): it
# masks internally and trains with soft-InfoNCE against frozen random targets.
# Corpus note: ERA5 is absent from the v1.2 corpus, so these runs use
# Sentinel-2 / Sentinel-1 / Landsat only.
set -e

PROJECT="perceiver_slp"
LAUNCH_ARGS="--launch.num_gpus=8 --launch.priority=normal --launch.clusters=[ai2/jupiter]"

python scripts/vnext/perceiver/nano.py launch slp_nano ai2/jupiter \
    $LAUNCH_ARGS \
    --trainer.callbacks.wandb.project="$PROJECT"

python scripts/vnext/perceiver/base.py launch slp_base ai2/jupiter \
    $LAUNCH_ARGS \
    --trainer.callbacks.wandb.project="$PROJECT"
