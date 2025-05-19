#!/bin/bash

#python scripts/v0_sweep/galileo.py launch test_sweep_gal ai2/titan-cirrascale --model.decoder_config.depth=4 --common.launch.num_gpus=8
#python scripts/v0_sweep/latent_mim.py launch test_sweep_gal ai2/titan-cirrascale --model.decoder_config.depth=4 --common.launch.num_gpus=8
#python scripts/v0_sweep/contrastive_latent_mim.py launch test_sweep_gal ai2/titan-cirrascale --model.decoder_config.depth=4 --common.launch.num_gpus=8

#python scripts/v0_sweep/galileo.py launch test_sweep_gal_no_mae ai2/jupiter-cirrascale --model.reconstructor_config=null --train_module.mae_loss_config=null --model.decoder_config.depth=4 --common.launch.num_gpus=8
python scripts/v0_sweep/galileo_st.py launch test_sweep_gal_st ai2/jupiter-cirrascale-2 --common.launch.num_gpus=8 --data_loader.token_budget=6000
python scripts/v0_sweep/galileo_st.py launch test_sweep_gal_st_no_mae ai2/jupiter-cirrascale-2 --model.reconstructor_config=null --train_module.mae_loss_config=null --common.launch.num_gpus=8 --data_loader.token_budget=6000
