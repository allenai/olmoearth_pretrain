#!/bin/bash

python scripts/v0.2_sweep/latent_mim_presto_osm.py launch v0.2_latent_mim_space_time_token_2250 ai2/jupiter-cirrascale-2 --launch.priority=urgent --data_loader.token_budget=2250 --common.launch.num_gpus=8
python scripts/v0.2_sweep/latent_mim_presto_osm.py launch v0.2_latent_mim_space_time_token_3250 ai2/jupiter-cirrascale-2 --launch.priority=urgent --data_loader.token_budget=3250 --common.launch.num_gpus=8
python scripts/v0.2_sweep/latent_mim_presto_osm.py launch v0.2_latent_mim_space_time_token_5000 ai2/jupiter-cirrascale-2 --launch.priority=urgent --data_loader.token_budget=5000 --common.launch.num_gpus=8
python scripts/v0.2_sweep/latent_mim_presto_osm.py launch v0.2_latent_mim_space_time_token_5000_bigger ai2/jupiter-cirrascale-2 --launch.priority=urgent --data_loader.token_budget=5000 --data_loader.sampled_hw_p_list=\[5,6,7,8,9,10,11,12,13,14,15,16\] --common.launch.num_gpus=8
