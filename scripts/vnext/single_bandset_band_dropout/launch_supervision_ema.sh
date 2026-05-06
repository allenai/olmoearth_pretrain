#!/bin/bash
set -e

SCRIPT_DIR="scripts/vnext/single_bandset_band_dropout"

# 1) Pixel supervision + EMA, no VICReg
python3 ${SCRIPT_DIR}/supervision_ema.py launch supervision_ema_v1 ai2/jupiter \
  --launch.num_gpus=8 \
  --trainer.callbacks.wandb.project=2026_02_08_masked_neg

# 2) Pixel supervision + EMA + 0.1x VICReg + 0.1x PatchVC
python3 ${SCRIPT_DIR}/supervision_ema_vicreg_tenth.py launch supervision_ema_vicreg_tenth_v1 ai2/jupiter \
  --launch.num_gpus=8 \
  --trainer.callbacks.wandb.project=2026_02_08_masked_neg
