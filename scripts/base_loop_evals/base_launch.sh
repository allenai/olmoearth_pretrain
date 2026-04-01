#!/bin/bash
# Base training with loop evals only (tolbi_crop, canada_wildfire_sat_eval_split, yemen_crop,
# geo_ecosystem_annual_test, forest_loss_driver, nigeria_settlement every 5k steps).
# Same as official base_launch.sh except eval tasks.
python scripts/base_loop_evals/base.py launch phase2.0_base_loop_evals ai2/ceres-cirrascale  --train_module.optim_config.lr=0.0001 --train_module.optim_config.weight_decay=0.02   --launch.clusters='[ai2/jupiter-cirrascale-2,ai2/ceres,ai2/titan]' --launch.num_gpus=8
