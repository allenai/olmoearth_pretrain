#!/bin/bash
# New ablations: L2 loss and encode/decode maps

# # L2 loss ablation (urgent priority)
# python scripts/official/ablations/base_l2_loss.py launch base_l2_loss ai2/jupiter \
#     --launch.clusters='[ai2/jupiter,ai2/titan-cirrascale]' \
#     --launch.priority=urgent \
#     --trainer.callbacks.wandb.project=2026_01_26_cvpr_ablations

# Encode/decode maps with random masking (high priority)
# Uses random masking to ensure all modalities always encoded on every rank
python scripts/official/ablations/base_encode_decode_maps_random_masking.py launch base_encode_decode_maps_random ai2/jupiter \
    --launch.clusters='[ai2/jupiter,ai2/titan-cirrascale]' \
    --launch.priority=high \
    --trainer.callbacks.wandb.project=2026_01_26_cvpr_ablations

# Encode/decode 3 maps only (srtm, worldcover, osm) with modality_cross_random (high priority)
# Fewer modalities reduces chance of FSDP sync issues
python scripts/official/ablations/base_encode_decode_3maps.py launch base_encode_decode_3maps ai2/jupiter \
    --launch.clusters='[ai2/jupiter,ai2/titan-cirrascale]' \
    --launch.priority=high \
    --trainer.callbacks.wandb.project=2026_01_26_cvpr_ablations
