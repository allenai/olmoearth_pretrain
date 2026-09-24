#!/usr/bin/env bash
# Exact launch of the ERA5 encoder training run with the CY-Bench evals
# (era5enc_1306_halo75_nogate_cybench_lyo = Beaker 01M3918WPJ07SZGTVAFP70BMK1,
# W&B era5_encoder_v2_evals, launched 2026-09-24 from gabi/cy-bench-swtinput @ 471f45cc8).
# A clone of hadriens' era5enc_1306_halo75_nogate (01M28KK8ENQF6N8DAXHKPK2J9Q) plus the
# eight cybench_* eval tasks. Run from the repo root on a clean, pushed checkout; the
# launcher pins GIT_REF to HEAD. Drop the *_lyo_eval tasks to reproduce the 5+2-task run
# era5enc_1306_halo75_nogate_cybench (01M36VHFRFDS9MNHD9M33N82XV).
set -euo pipefail
RUN_NAME=${RUN_NAME:-era5enc_1306_halo75_nogate_cybench_lyo}
python scripts/era5_supervised/v0/base.py launch "$RUN_NAME" ai2/saturn \
  common.learning_rate=1e-5 common.max_steps=50000 common.global_batch_size=32 \
  'common.tasks=[era5enc_pretrain_ssl]' \
  'common.eval_tasks=[lfmc_woody_eval,burnrisk_canada_nbac_eval,landslide_era5_eval,cybench_maize_eval,cybench_wheat_eval,cybench_maize_US_lyo_eval,cybench_wheat_US_lyo_eval,cybench_maize_DE_lyo_eval,cybench_wheat_DE_lyo_eval,cybench_maize_AR_lyo_eval,cybench_wheat_AR_lyo_eval]' \
  common.enable_downstream_eval=True common.enable_supervised=False common.enable_reconstruction=True \
  common.eval_interval=1000 common.eval_on_startup=True common.eval_probe_seed=1202 \
  common.encoder_use_conv_stem=True common.encoder_pooling=mean \
  common.encoder_swt_input=True common.encoder_swt_input_include_approx=True \
  common.encoder_swt_input_stats_path=scripts/era5_supervised/v0/norm_configs/swt_input_stats.json \
  common.recon_swt_lambda=0.0 common.recon_raw_lambda=1.0 common.recon_mask_policy=swt_halo_span \
  'common.recon_span_num_spans=[4,10]' 'common.recon_span_days=[30,120]' 'common.recon_span_num_variables=[9,14]' \
  --model.reconstruction_objective.group_recon_mode.pressure=raw_plus_all_swt \
  --launch.num_gpus=1 --launch.num_nodes=1 --launch.priority=urgent \
  --trainer.callbacks.wandb.project=era5_encoder_v2_evals \
  --trainer.callbacks.wandb.name="$RUN_NAME"
