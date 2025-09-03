#!/bin/bash

python scripts/v0.2_sweep/latent_mim_128.py launch v0.2_latent_mim_128_space_time_r1 ai2/titan-cirrascale --launch.priority=urgent --common.launch.num_gpus=1 --init_seed=1234 --data_loader.seed=1234
python scripts/v0.2_sweep/latent_mim_128.py launch v0.2_latent_mim_128_space_time_r2 ai2/titan-cirrascale --launch.priority=urgent --common.launch.num_gpus=1 --init_seed=5678 --data_loader.seed=5678
python scripts/v0.2_sweep/latent_mim_128.py launch v0.2_latent_mim_128_space_time_r3 ai2/titan-cirrascale --launch.priority=urgent --common.launch.num_gpus=1 --init_seed=91011 --data_loader.seed=91011
python scripts/v0.2_sweep/latent_mim_128.py launch v0.2_latent_mim_128_space_time_r4 ai2/titan-cirrascale --launch.priority=urgent --common.launch.num_gpus=1 --init_seed=121314 --data_loader.seed=121314
