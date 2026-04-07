#!/bin/bash
# Launch text-target ablation experiments.
#
# Ablation 2: text_targets          — text embedding targets + masked negatives loss
# Ablation 3: text_targets_no_mask  — text embedding targets + original vec loss (no masking)
#
# Control (base_speedup) should already be running or have results.

set -euo pipefail

WANDB_PROJECT="2025_text_target_ablations"
CLUSTERS="ai2/jupiter"
NUM_GPUS=8

# Ablation 2: text targets + masked negatives
python scripts/vnext/speedups/text_targets.py launch text_targets "$CLUSTERS" \
    --launch.num_gpus=$NUM_GPUS \
    --trainer.callbacks.wandb.project=$WANDB_PROJECT

# Ablation 3: text targets + original vec loss (no masking)
python scripts/vnext/speedups/text_targets_no_mask.py launch text_targets_no_mask "$CLUSTERS" \
    --launch.num_gpus=$NUM_GPUS \
    --trainer.callbacks.wandb.project=$WANDB_PROJECT
