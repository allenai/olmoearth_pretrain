#!/bin/bash

python scripts/joe/latent_mim_st.py launch latent_mim_base_st_contrastive_space_time ai2/jupiter-cirrascale-2 --launch.priority=low --common.launch.num_gpus=2
