#!/bin/bash

#python scripts/joe/galileo.py launch galileo_large_repro_noema_ ai2/jupiter-cirrascale-2 --launch.priority=urgent --common.launch.num_gpus=8 --train_module.ema_decay=\[1,1\]
#python scripts/joe/galileo.py launch galileo_large_repro_fewer_modalities ai2/jupiter-cirrascale-2 --launch.priority=urgent --common.launch.num_gpus=8 --train_module.ema_decay=\[1,1\] --common.training_modalities=\[sentinel2_l2a,sentinel1,worldcover\]
#python scripts/joe/galileo.py launch galileo_large_repro_noema_3modalities_fix ai2/jupiter-cirrascale-2 --launch.priority=urgent --common.launch.num_gpus=8 --common.training_modalities=\[sentinel2_l2a,sentinel1,worldcover\]
python scripts/joe/galileo.py launch galileo_base_repro_noema_3modalities_nan_workaround ai2/jupiter-cirrascale-2 --launch.priority=urgent --common.launch.num_gpus=8 --common.training_modalities=\[sentinel2_l2a,sentinel1,worldcover\]
