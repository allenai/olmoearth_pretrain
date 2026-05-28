#!/bin/bash
# Launch the v1_1 + 64-dim output-bottleneck sweep.
#
# All runs share the v1_1 recipe and bottleneck the encoder to a 64-dim output
# embedding. They differ only in the "packer" that maps 768 -> 64:
#   base_out64           single Linear (control / hard bottleneck)
#   out64_mlp_768        768 -> 64
#   out64_mlp_1536_768   1536 -> 768 -> 64
#   out64_mlp_2048x2_1024  2048 -> 2048 -> 1024 -> 64
#
# Usage: bash scripts/official/v1_1/output_bottleneck_64/launch_out64.sh
set -e

DIR="scripts/official/v1_1/output_bottleneck_64"
CLUSTER="ai2/jupiter"
WANDB="--trainer.callbacks.wandb.project=2026_05_28_output_bottleneck_64"
COMMON="--launch.num_gpus=8 --launch.clusters=[ai2/jupiter,ai2/ceres] $WANDB"

# control: single-linear packer
python $DIR/base_out64.py launch out64_linear $CLUSTER $COMMON

# increasingly bigger non-linear packers
python $DIR/out64_mlp_768.py launch out64_mlp_768 $CLUSTER $COMMON
python $DIR/out64_mlp_1536_768.py launch out64_mlp_1536_768 $CLUSTER $COMMON
python $DIR/out64_mlp_2048x2_1024.py launch out64_mlp_2048x2_1024 $CLUSTER $COMMON
