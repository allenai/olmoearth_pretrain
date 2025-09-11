python scripts/2025_09_10_phase1/script.py launch 0_base_default ai2/ceres-cirrascale
python scripts/2025_09_10_phase1/script.py launch 0_base_contrastive0.05 ai2/ceres-cirrascale --train_module.contrastive_config.loss_config.weight=0.05
python scripts/2025_09_10_phase1/script.py launch 0_base_lr2e-4 ai2/ceres-cirrascale --train_module.optim_config.lr=0.0002
python scripts/2025_09_10_phase1/script.py launch 0_base_encode0.25 ai2/ceres-cirrascale --train_module.masking_config.strategy_config.encode_ratio=0.25 --train_module.masking_config.strategy_config.decode_ratio=0.75
# Encode ratio 0.75 needs smaller microbatch size.
python scripts/2025_09_10_phase1/script.py launch 0_base_encode0.75 ai2/ceres-cirrascale --train_module.masking_config.strategy_config.encode_ratio=0.75 --train_module.masking_config.strategy_config.decode_ratio=0.25 --train_module.rank_microbatch_size=32 --launch.clusters='[ai2/ceres-cirrascale,ai2/jupiter-cirrascale-2]'
python scripts/2025_09_10_phase1/script.py launch 0_base_hw1-12 ai2/ceres-cirrascale --data_loader.sampled_hw_p_list='[1,2,3,4,5,6,7,8,9,10,11,12]' --launch.clusters='[ai2/ceres-cirrascale,ai2/jupiter-cirrascale-2]'
python scripts/2025_09_10_phase1/script.py launch 0_base_budget2250 ai2/ceres-cirrascale --data_loader.token_budget=2250 --launch.clusters='[ai2/ceres-cirrascale,ai2/jupiter-cirrascale-2]' --train_module.rank_microbatch_size=32
python scripts/2025_09_10_phase1/script_chm.py launch 0_base_add_chm ai2/ceres-cirrascale --launch.clusters='[ai2/ceres-cirrascale,ai2/jupiter-cirrascale-2]' --data_loader.token_budget=1700 --train_module.rank_microbatch_size=32
# CDL needs bs 64 since otherwise it is rare enough that with bs32 some batches have no CDL and it crashes.
# We need to run it on Titan to have enough GPU memory though.
python scripts/2025_09_10_phase1/script_chm_cdl_worldcereal.py launch 0_base_add_chm_cdl_worldcereal ai2/titan-cirrascale --data_loader.token_budget=2000 --launch.priority=low
python scripts/2025_09_10_phase1/script.py launch 0_base_bs32 ai2/ceres-cirrascale --train_module.rank_microbatch_size=32 --launch.clusters='[ai2/ceres-cirrascale,ai2/jupiter-cirrascale-2]'
python scripts/2025_09_10_phase1/script_chm_cdl_worldcereal.py launch 0_base_add_chm_worldcereal ai2/ceres-cirrascale --launch.clusters='[ai2/ceres-cirrascale,ai2/jupiter-cirrascale-2]' --data_loader.token_budget=2000 --train_module.rank_microbatch_size=32 --common.training_modalities='[sentinel2_l2a,sentinel1,landsat,worldcover,srtm,openstreetmap_raster,wri_canopy_height_map,worldcereal]' --train_module.masking_config.strategy_config.only_decode_modalities='[worldcover,srtm,openstreetmap_raster,wri_canopy_height_map,worldcereal]'
