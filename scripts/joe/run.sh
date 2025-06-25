#!/bin/bash

#python scripts/joe/latent_mim.py launch latent_mim_batch_contrastive_test ai2/titan-cirrascale --launch.priority=urgent --common.launch.num_gpus=1
python scripts/joe/latent_mim.py launch latent_mim_batch_contrastive_tau_1 ai2/titan-cirrascale --launch.priority=urgent --common.launch.num_gpus=8 --train_module.regularizer_config.loss_config.tau=1.0
