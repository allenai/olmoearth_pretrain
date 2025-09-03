#!/bin/bash

#python scripts/v0.2_sweep/latent_mim_large.py launch v0.2_large_latent_mim_space_time_10e-4 ai2/jupiter-cirrascale-2 --launch.priority=urgent  --common.launch.num_gpus=8 --train_module.optim_config.lr=0.0001
#python scripts/v0.2_sweep/latent_mim_large.py launch v0.2_large_latent_mim_space_time_10e-5 ai2/jupiter-cirrascale-2 --launch.priority=urgent  --common.launch.num_gpus=8 --train_module.optim_config.lr=0.00001
#python scripts/v0.2_sweep/latent_mim_large_alldata.py launch v0.2_large_latent_mim_space_time_alldata_10e-4 ai2/ceres-cirrascale --launch.priority=urgent --common.launch.num_gpus=8 --train_module.optim_config.lr=0.0001
#python scripts/v0.2_sweep/latent_mim_large_alldata.py launch v0.2_large_latent_mim_space_time_alldata_10e-5 ai2/ceres-cirrascale --launch.priority=urgent --common.launch.num_gpus=8 --train_module.optim_config.lr=0.00001
python scripts/v0.2_sweep/latent_mim_large.py launch v0.2_large_latent_mim_space_time_10e-3 ai2/jupiter-cirrascale-2 --launch.priority=urgent  --common.launch.num_gpus=8 --train_module.optim_config.lr=0.001
python scripts/v0.2_sweep/latent_mim_large.py launch v0.2_large_latent_mim_space_time_0.0004 ai2/jupiter-cirrascale-2 --launch.priority=urgent  --common.launch.num_gpus=8 --train_module.optim_config.lr=0.0004
