#!/bin/bash

python scripts/v0.2_sweep/latent_mim.py launch v0.2_latent_mim_space_time_data_all ai2/jupiter-cirrascale-2 --launch.priority=urgent  --common.launch.num_gpus=8
python scripts/v0.2_sweep/latent_mim_no_worldcover.py launch v0.2_latent_mim_space_data_noworldcover ai2/jupiter-cirrascale-2 --launch.priority=urgent --common.launch.num_gpus=8
