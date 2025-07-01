#!/bin/bash

#python scripts/joe/latent_mim.py launch latent_mim_batch_contrastive_test ai2/titan-cirrascale --launch.priority=urgent --common.launch.num_gpus=1
#python scripts/joe/latent_mim.py launch latent_mim_batch_contrastive_tau_2 ai2/titan-cirrascale --launch.priority=urgent --common.launch.num_gpus=8 --train_module.regularizer_config.loss_config.tau=1.0
python scripts/joe/latent_mim.py launch latent_mim_no_batch_contrastive ai2/titan-cirrascale --launch.priority=high --common.launch.num_gpus=8 --train_module.regularizer_config.loss_config.weight=0
python scripts/joe/galileo.py launch galileo_all_disc_cross_space_time ai2/titan-cirrascale --launch.priority=high --common.launch.num_gpus=8
