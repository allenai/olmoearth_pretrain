#!/bin/bash

python scripts/joe/latent_mim.py launch latent_mim_batch_contrastive_test ai2/titan-cirrascale --launch.priority=urgent --common.launch.num_gpus=1
