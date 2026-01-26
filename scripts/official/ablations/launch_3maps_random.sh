#!/bin/bash
# Ablation: 3 maps (srtm, worldcover, osm) + satellite modalities with random masking

python scripts/official/ablations/base_encode_decode_3maps_random_masking.py launch base_encode_decode_3maps_random ai2/jupiter \
    --launch.clusters='[ai2/jupiter,ai2/titan-cirrascale]' \
    --launch.priority=high \
    --trainer.callbacks.wandb.project=2026_01_26_cvpr_ablations
