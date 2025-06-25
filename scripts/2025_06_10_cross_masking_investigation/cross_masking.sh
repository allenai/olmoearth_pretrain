#! /bin/bash

# Min max encoding bandsets investigation
# python3 scripts/2025_06_10_cross_masking_investigation/latent_mim_128_cross.py launch v0.2_latent_mim_128_cross_space_time_allow_overlap_minenc_2_alldec ai2/titan-cirrascale --launch.priority=high --common.launch.num_gpus=1 --train_module.masking_config.strategy_config.min_encoded_bandsets=2 --train_module.rank_microbatch_size=128
# python3 scripts/2025_06_10_cross_masking_investigation/latent_mim_128_cross.py launch v0.2_latent_mim_128_cross_space_time_allow_overlap_minenc_2_mindec_6 ai2/titan-cirrascale --launch.priority=high --common.launch.num_gpus=1 --train_module.masking_config.strategy_config.min_encoded_bandsets=2 --train_module.masking_config.strategy_config.min_decoded_bandsets=6 --train_module.rank_microbatch_size=128


# Decode only modalities
python3 scripts/2025_06_10_cross_masking_investigation/latent_mim_128_cross.py launch v0.2_latent_mim_128_cross_space_time_allow_overlap_minenc_2_alldec_only_dec_worldcover_osm ai2/titan-cirrascale --launch.priority=high --common.launch.num_gpus=1 --train_module.masking_config.strategy_config.min_encoded_bandsets=2 --train_module.rank_microbatch_size=128 --train_module.masking_config.strategy_config.only_decode_modalities=\[worldcover,openstreetmap_raster\]
python3 scripts/2025_06_10_cross_masking_investigation/latent_mim_128_cross.py launch v0.2_latent_mim_128_cross_space_time_allow_overlap_minenc_2_alldec_only_dec_srtm_worldcover_osm ai2/titan-cirrascale --launch.priority=high --common.launch.num_gpus=1 --train_module.masking_config.strategy_config.min_encoded_bandsets=2 --train_module.rank_microbatch_size=128 --train_module.masking_config.strategy_config.only_decode_modalities=\[srtm,worldcover,openstreetmap_raster\]
