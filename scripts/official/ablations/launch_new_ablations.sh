#!/bin/bash
# New ablations: L2 loss and encode/decode maps

# L2 loss ablation (urgent priority)
python scripts/official/ablations/base_l2_loss.py launch base_l2_loss ai2/jupiter \
    --launch.clusters='[ai2/jupiter,ai2/titan-cirrascale]' \
    --launch.priority=urgent \
    --trainer.callbacks.wandb.project=2026_01_26_cvpr_ablations

# Encode/decode maps ablation (high priority, flash attn enabled in script)
python scripts/official/ablations/base_encode_decode_maps.py launch base_encode_decode_maps ai2/jupiter \
    --launch.clusters='[ai2/jupiter,ai2/titan-cirrascale]' \
    --launch.priority=high \
    --trainer.callbacks.wandb.project=2026_01_26_cvpr_ablations
