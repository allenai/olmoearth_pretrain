#!/bin/bash
# Launch script for large model undertraining experiments
# 2025_01_28_large_undertraining

set -e

WANDB_PROJECT="2025_01_28_large_undertraining"

# Exp 1a: Large + Constant LR + with maps (jupiter or titan)
echo "Launching Exp 1a: large_const_lr_with_maps"
python3 scripts/official/large_const_lr.py launch large_const_lr_with_maps ai2/jupiter \
  --launch.num_gpus=8 \
  --launch.clusters="[ai2/jupiter,ai2/titan]" \
  --trainer.callbacks.wandb.project=$WANDB_PROJECT

# Exp 1b: Large + Constant LR + no maps (jupiter or titan)
echo "Launching Exp 1b: large_const_lr_no_maps"
python3 scripts/official/large_const_lr_no_maps.py launch large_const_lr_no_maps ai2/jupiter \
  --launch.num_gpus=8 \
  --launch.clusters="[ai2/jupiter,ai2/titan]" \
  --trainer.callbacks.wandb.project=$WANDB_PROJECT

# Exp 2a-1: Large + Deep decoder (8 layers) - titan only
echo "Launching Exp 2a-1: large_deep_decoder_8"
python3 scripts/official/large_deep_decoder.py launch large_deep_decoder_8 ai2/titan \
  --launch.num_gpus=8 \
  --launch.clusters="[ai2/titan]" \
  --model.decoder_config.depth=8 \
  --trainer.callbacks.wandb.project=$WANDB_PROJECT

# Exp 2a-2: Large + Deep decoder (16 layers) - titan only
echo "Launching Exp 2a-2: large_deep_decoder_16"
python3 scripts/official/large_deep_decoder.py launch large_deep_decoder_16 ai2/titan \
  --launch.num_gpus=8 \
  --launch.clusters="[ai2/titan]" \
  --model.decoder_config.depth=16 \
  --trainer.callbacks.wandb.project=$WANDB_PROJECT

echo "All experiments launched!"
