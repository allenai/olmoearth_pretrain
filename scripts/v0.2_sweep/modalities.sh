#!/bin/bash

#python scripts/v0.2_sweep/latent_mim_presto_osm.py launch v0.2_latent_mim_space_time_sen2 ai2/jupiter-cirrascale-2 --launch.priority=urgent --common.training_modalities=\[sentinel2_l2a,\] --common.launch.num_gpus=8
#python scripts/v0.2_sweep/latent_mim_presto_osm.py launch v0.2_latent_mim_space_time_sen2_sen1 ai2/jupiter-cirrascale-2 --launch.priority=urgent --common.training_modalities=\[sentinel2_l2a,sentinel1,\] --common.launch.num_gpus=8
#python scripts/v0.2_sweep/latent_mim_presto_osm.py launch v0.2_latent_mim_space_time_sen2_sen1_landsat_srtm_latlon ai2/jupiter-cirrascale-2 --launch.priority=urgent --common.training_modalities=\[sentinel2_l2a,sentinel1,landsat,srtm,latlon\] --common.launch.num_gpus=8

# Titan on 128
python scripts/v0.2_sweep/latent_mim_128.py launch v0.2_latent_mim_128_space_time_sen2 ai2/titan-cirrascale --launch.priority=urgent --common.training_modalities=\[sentinel2_l2a,\] --common.launch.num_gpus=1
python scripts/v0.2_sweep/latent_mim_128.py launch v0.2_latent_mim_128_space_time_sen2_sen1 ai2/titan-cirrascale --launch.priority=urgent --common.training_modalities=\[sentinel2_l2a,sentinel1,\] --common.launch.num_gpus=1
python scripts/v0.2_sweep/latent_mim_128.py launch v0.2_latent_mim_128_space_time_sen2_sen1_landsat_srtm_latlon ai2/titan-cirrascale --launch.priority=urgent --common.training_modalities=\[sentinel2_l2a,sentinel1,landsat,srtm,latlon\] --common.launch.num_gpus=1
python scripts/v0.2_sweep/latent_mim_128.py launch v0.2_latent_mim_128_space_time_sen2_sen1_landsat_srtm_latlon_worldcover_osm ai2/titan-cirrascale --launch.priority=urgent --common.training_modalities=\[sentinel2_l2a,sentinel1,landsat,srtm,latlon,worldcover,openstreetmap_raster\] --common.launch.num_gpus=1
