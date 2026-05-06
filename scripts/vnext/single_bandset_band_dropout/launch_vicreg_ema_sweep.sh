#!/bin/bash
set -e

SCRIPT_DIR="scripts/vnext/single_bandset_band_dropout"

# 1) 0.1x VICReg only (no patch varcov), EMA
python3 ${SCRIPT_DIR}/vicreg_ema_tenth.py launch vicreg_ema_tenth_v1 ai2/jupiter \
  --launch.num_gpus=8 \
  --trainer.callbacks.wandb.project=2026_02_08_masked_neg

# 2) 0.1x VICReg + 0.1x patch varcov, EMA
python3 ${SCRIPT_DIR}/vicreg_ema_tenth_pvc.py launch vicreg_ema_tenth_pvc_v1 ai2/jupiter \
  --launch.num_gpus=8 \
  --trainer.callbacks.wandb.project=2026_02_08_masked_neg

# 3) 0.02x both, EMA
python3 ${SCRIPT_DIR}/vicreg_ema_fiftieth.py launch vicreg_ema_fiftieth_v1 ai2/jupiter \
  --launch.num_gpus=8 \
  --trainer.callbacks.wandb.project=2026_02_08_masked_neg

# 4) 0.01x both, EMA (control)
python3 ${SCRIPT_DIR}/vicreg_ema_hundredth.py launch vicreg_ema_hundredth_v1 ai2/jupiter \
  --launch.num_gpus=8 \
  --trainer.callbacks.wandb.project=2026_02_08_masked_neg
