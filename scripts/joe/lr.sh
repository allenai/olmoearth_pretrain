#!/bin/bash

# Titan on 128
python scripts/joe/latent_mim.py launch latent_mim_lr.0001 ai2/titan-cirrascale --launch.priority=low --common.launch.num_gpus=1 --train_module.optim_config.lr=0.0001
python scripts/joe/latent_mim.py launch latent_mim_lr.0002 ai2/titan-cirrascale --launch.priority=low --common.launch.num_gpus=1 --train_module.optim_config.lr=0.0002
python scripts/joe/latent_mim.py launch latent_mim_lr.0004 ai2/titan-cirrascale --launch.priority=low --common.launch.num_gpus=1 --train_module.optim_config.lr=0.0004
python scripts/joe/latent_mim.py launch latent_mim_lr.001 ai2/titan-cirrascale --launch.priority=low --common.launch.num_gpus=1 --train_module.optim_config.lr=0.001
