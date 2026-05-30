#!/bin/bash
# v1.1 tuned axial 2D-RoPE sweep (8 GPUs each, ai2/jupiter).
#
# Recalibrates base/scale to the model's ACTUAL positional range. The largest
# tile we feed is ~1.28 km = 128 px @ 10 m, so the RoPE coordinate
#   pos = (meters / BASE_GSD) * scale = (meters / 10) * scale
# tops out at ~128 * scale units. base sets the longest wavelength (2*pi*base);
# the old 10k/100k defaults are NLP-scale (thousands of tokens) and leave most
# frequency bands dead over a 128-unit range. So we sweep base DOWN.
#
#   - scale fixes the fine end: a 10 m gap -> `scale` rad at the top band.
#     scale=0.25 is the known-solid setting; scale=0.5 tests a sharper fine end
#     ("10 m must stay clear") and is still well under pi (no neighbor aliasing).
#   - base fixes the coarse end / extrapolation headroom. base=1000 already
#     gives ~49x headroom over 1.28 km; 100/300 push more of the spectrum into
#     the useful in-range band.
#
# Already running (do not relaunch): base{1k,10k,100k} x scale{1,0.25}.
# These 4 fill the low-base curve at scale=0.25 plus a sharper-fine-end probe.
set -e

SCRIPT="scripts/official/v1_1/rope.py"
PROJECT="2026_04_22_add_hidden_layer_to_initial_projection"
LAUNCH_ARGS="--launch.num_gpus=8 --launch.priority=urgent --launch.clusters=[ai2/jupiter]"

launch_axial() {
    local base="$1"
    local scale="$2"
    local name="rope_base${base}_scale${scale}"
    python "$SCRIPT" launch "$name" ai2/jupiter \
        $LAUNCH_ARGS \
        --trainer.callbacks.wandb.project="$PROJECT" \
        --model.encoder_config.rope_base="$base" \
        --model.decoder_config.rope_base="$base" \
        --model.encoder_config.rope_coordinate_scale="$scale" \
        --model.decoder_config.rope_coordinate_scale="$scale"
}

# Low-base curve at the solid scale=0.25.
launch_axial 100 0.25
launch_axial 300 0.25

# Sharper fine end ("10 m crisp") at the good bases.
launch_axial 300 0.5
launch_axial 1000 0.5
