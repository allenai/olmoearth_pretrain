#!/bin/bash
# Scaling sweep: ViT-H 3-bandset + ViT-H single-bandset + ViT-L proj768
# All use vec loss, urgent priority, same wandb project.
set -e

WANDB="--trainer.callbacks.wandb.project=2026_05_07_scaling_investigation"
# --- ViT-H 3-bandset: 150k steps (existing script, vec loss override) ---
python scripts/vnext/vit_h_fsdp.py launch vit_h_3bandset_150k ai2/jupiter \
    --launch.num_nodes=2 \
    --launch.num_gpus=8 \
    --launch.priority=urgent \
    --trainer.max_duration.value=150000 \
    --trainer.max_duration.unit=steps \
    --train_module.loss_config.loss_config.type=modality_patch_discrimination_vec \
    $WANDB

# --- ViT-H single bandset: 150k steps ---
python scripts/vnext/vit_h_single_bandset.py launch vit_h_sb_150k ai2/jupiter \
    --launch.num_nodes=2 \
    --launch.num_gpus=8 \
    --launch.priority=urgent \
    $WANDB

# --- ViT-L single bandset: encoder projects to 768 ---
python scripts/vnext/large_single_bandset_proj768.py launch large_sb_proj768 ai2/jupiter \
    --launch.num_gpus=8 \
    --launch.priority=urgent \
    $WANDB
