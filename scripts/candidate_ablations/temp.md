python3 scripts/candidate_ablations/run_candidate_ablation_single_bandset.py launch basev11_200k_solo_novelty_divp95 ai2/jupiter-cirrascale-2 \
    --candidate_columns novelty \
    --score_suffix diverse_score_p95 \
    --select_top 50000 \
    --candidate_parquet /weka/dfive-default/rslearn-eai/datasets/globe_land_grid/s50ix24_embeddings/_scores/combined_acquisition_scores.parquet \
    --candidate_h5py_dir /weka/dfive-default/helios/dataset/candidates/h5py_data_w_missing_timesteps_zstd_3_128_x_1/cdl_landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcereal_worldcover_wri_canopy_height_map/693942 \
    --trainer.load_path=/weka/dfive-default/helios/checkpoints/favyen/hidden1/step200000 \
    --train_module.optim_config.lr=0.0001 \
    --train_module.scheduler.warmup_steps=0 \
    --train_module.scheduler.alpha_f=0.1 \
    --train_module.scheduler.t_max=400000 \
    --trainer.max_duration.value=400000 \
    --trainer.max_duration.unit=steps \
    --launch.priority=urgent \
    --launch.num_gpus=8 \
    --launch.num_nodes=1 \
    --trainer.callbacks.wandb.project=20260513_candidate_datasets \
    --trainer.callbacks.wandb.name=basev11_200k_solo_novelty_divp95

python3 scripts/candidate_ablations/run_candidate_ablation_single_bandset.py launch basev11_200k_solo_xglobal_bridge_divp95 ai2/jupiter-cirrascale-2 \
    --candidate_columns xglobal_bridge \
    --score_suffix diverse_score_p95 \
    --select_top 50000 \
    --candidate_parquet /weka/dfive-default/rslearn-eai/datasets/globe_land_grid/s50ix24_embeddings/_scores/combined_acquisition_scores.parquet \
    --candidate_h5py_dir /weka/dfive-default/helios/dataset/candidates/h5py_data_w_missing_timesteps_zstd_3_128_x_1/cdl_landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcereal_worldcover_wri_canopy_height_map/693942 \
    --trainer.load_path=/weka/dfive-default/helios/checkpoints/favyen/hidden1/step200000 \
    --train_module.optim_config.lr=0.0001 \
    --train_module.scheduler.warmup_steps=0 \
    --train_module.scheduler.alpha_f=0.1 \
    --train_module.scheduler.t_max=400000 \
    --trainer.max_duration.value=400000 \
    --trainer.max_duration.unit=steps \
    --launch.priority=urgent \
    --launch.num_gpus=8 \
    --launch.num_nodes=1 \
    --trainer.callbacks.wandb.project=20260513_candidate_datasets \
    --trainer.callbacks.wandb.name=basev11_200k_solo_xglobal_bridge_divp95

python3 scripts/candidate_ablations/run_candidate_ablation_single_bandset.py launch basev11_200k_solo_sparse_infill_divp95 ai2/jupiter-cirrascale-2 \
    --candidate_columns sparse_infill \
    --score_suffix diverse_score_p95 \
    --select_top 50000 \
    --candidate_parquet /weka/dfive-default/rslearn-eai/datasets/globe_land_grid/s50ix24_embeddings/_scores/combined_acquisition_scores.parquet \
    --candidate_h5py_dir /weka/dfive-default/helios/dataset/candidates/h5py_data_w_missing_timesteps_zstd_3_128_x_1/cdl_landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcereal_worldcover_wri_canopy_height_map/693942 \
    --trainer.load_path=/weka/dfive-default/helios/checkpoints/favyen/hidden1/step200000 \
    --train_module.optim_config.lr=0.0001 \
    --train_module.scheduler.warmup_steps=0 \
    --train_module.scheduler.alpha_f=0.1 \
    --train_module.scheduler.t_max=400000 \
    --trainer.max_duration.value=400000 \
    --trainer.max_duration.unit=steps \
    --launch.priority=urgent \
    --launch.num_gpus=8 \
    --launch.num_nodes=1 \
    --trainer.callbacks.wandb.project=20260513_candidate_datasets \
    --trainer.callbacks.wandb.name=basev11_200k_solo_sparse_infill_divp95

python3 scripts/candidate_ablations/run_candidate_ablation_single_bandset.py launch basev11_200k_solo_xlocal_bridge_divp95 ai2/jupiter-cirrascale-2 \
    --candidate_columns xlocal_bridge \
    --score_suffix diverse_score_p95 \
    --select_top 50000 \
    --candidate_parquet /weka/dfive-default/rslearn-eai/datasets/globe_land_grid/s50ix24_embeddings/_scores/combined_acquisition_scores.parquet \
    --candidate_h5py_dir /weka/dfive-default/helios/dataset/candidates/h5py_data_w_missing_timesteps_zstd_3_128_x_1/cdl_landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcereal_worldcover_wri_canopy_height_map/693942 \
    --trainer.load_path=/weka/dfive-default/helios/checkpoints/favyen/hidden1/step200000 \
    --train_module.optim_config.lr=0.0001 \
    --train_module.scheduler.warmup_steps=0 \
    --train_module.scheduler.alpha_f=0.1 \
    --train_module.scheduler.t_max=400000 \
    --trainer.max_duration.value=400000 \
    --trainer.max_duration.unit=steps \
    --launch.priority=urgent \
    --launch.num_gpus=8 \
    --launch.num_nodes=1 \
    --trainer.callbacks.wandb.project=20260513_candidate_datasets \
    --trainer.callbacks.wandb.name=basev11_200k_solo_xlocal_bridge_divp95

python3 scripts/candidate_ablations/run_candidate_ablation_single_bandset.py launch basev11_200k_solo_prototypes_divp95 ai2/jupiter-cirrascale-2 \
    --candidate_columns prototypes \
    --score_suffix diverse_score_p95 \
    --select_top 50000 \
    --candidate_parquet /weka/dfive-default/rslearn-eai/datasets/globe_land_grid/s50ix24_embeddings/_scores/combined_acquisition_scores.parquet \
    --candidate_h5py_dir /weka/dfive-default/helios/dataset/candidates/h5py_data_w_missing_timesteps_zstd_3_128_x_1/cdl_landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcereal_worldcover_wri_canopy_height_map/693942 \
    --trainer.load_path=/weka/dfive-default/helios/checkpoints/favyen/hidden1/step200000 \
    --train_module.optim_config.lr=0.0001 \
    --train_module.scheduler.warmup_steps=0 \
    --train_module.scheduler.alpha_f=0.1 \
    --train_module.scheduler.t_max=400000 \
    --trainer.max_duration.value=400000 \
    --trainer.max_duration.unit=steps \
    --launch.priority=urgent \
    --launch.num_gpus=8 \
    --launch.num_nodes=1 \
    --trainer.callbacks.wandb.project=20260513_candidate_datasets \
    --trainer.callbacks.wandb.name=basev11_200k_solo_prototypes_divp95



# CANDIDATE ONLY

# Candidate-only from scratch (0 -> 500k steps, no base dataset, no checkpoint)
python3 scripts/candidate_ablations/run_candidate_ablation_single_bandset.py launch basev11_candidateonly_novelty_100k_divp95 ai2/jupiter-cirrascale-2 \
    --candidate_only \
    --candidate_columns novelty \
    --score_suffix diverse_score_p95 \
    --select_top 100000 \
    --candidate_parquet /weka/dfive-default/rslearn-eai/datasets/globe_land_grid/s50ix24_embeddings/_scores/combined_acquisition_scores.parquet \
    --candidate_h5py_dir /weka/dfive-default/helios/dataset/candidates/h5py_data_w_missing_timesteps_zstd_3_128_x_1/cdl_landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcereal_worldcover_wri_canopy_height_map/693942 \
    --data_loader.num_dataset_repeats_per_epoch=11 \
    --train_module.optim_config.lr=0.0001 \
    --train_module.scheduler.warmup_steps=8000 \
    --train_module.scheduler.alpha_f=0.1 \
    --train_module.scheduler.t_max=500000 \
    --trainer.max_duration.value=500000 \
    --trainer.max_duration.unit=steps \
    --launch.priority=urgent \
    --launch.num_gpus=8 \
    --launch.num_nodes=1 \
    --trainer.callbacks.wandb.project=20260513_candidate_datasets \
    --trainer.callbacks.wandb.name=basev11_candidateonly_novelty_100k_divp95

python3 scripts/candidate_ablations/run_candidate_ablation_single_bandset.py launch basev11_candidateonly_xglobal_bridge_100k_divp95 ai2/jupiter-cirrascale-2 \
    --candidate_only \
    --candidate_columns xglobal_bridge \
    --score_suffix diverse_score_p95 \
    --select_top 100000 \
    --candidate_parquet /weka/dfive-default/rslearn-eai/datasets/globe_land_grid/s50ix24_embeddings/_scores/combined_acquisition_scores.parquet \
    --candidate_h5py_dir /weka/dfive-default/helios/dataset/candidates/h5py_data_w_missing_timesteps_zstd_3_128_x_1/cdl_landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcereal_worldcover_wri_canopy_height_map/693942 \
    --data_loader.num_dataset_repeats_per_epoch=11 \
    --train_module.optim_config.lr=0.0001 \
    --train_module.scheduler.warmup_steps=8000 \
    --train_module.scheduler.alpha_f=0.1 \
    --train_module.scheduler.t_max=500000 \
    --trainer.max_duration.value=500000 \
    --trainer.max_duration.unit=steps \
    --launch.priority=urgent \
    --launch.num_gpus=8 \
    --launch.num_nodes=1 \
    --trainer.callbacks.wandb.project=20260513_candidate_datasets \
    --trainer.callbacks.wandb.name=basev11_candidateonly_xglobal_bridge_100k_divp95

python3 scripts/candidate_ablations/run_candidate_ablation_single_bandset.py launch basev11_candidateonly_sparse_infill_100k_divp95 ai2/jupiter-cirrascale-2 \
    --candidate_only \
    --candidate_columns sparse_infill \
    --score_suffix diverse_score_p95 \
    --select_top 100000 \
    --candidate_parquet /weka/dfive-default/rslearn-eai/datasets/globe_land_grid/s50ix24_embeddings/_scores/combined_acquisition_scores.parquet \
    --candidate_h5py_dir /weka/dfive-default/helios/dataset/candidates/h5py_data_w_missing_timesteps_zstd_3_128_x_1/cdl_landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcereal_worldcover_wri_canopy_height_map/693942 \
    --data_loader.num_dataset_repeats_per_epoch=11 \
    --train_module.optim_config.lr=0.0001 \
    --train_module.scheduler.warmup_steps=8000 \
    --train_module.scheduler.alpha_f=0.1 \
    --train_module.scheduler.t_max=500000 \
    --trainer.max_duration.value=500000 \
    --trainer.max_duration.unit=steps \
    --launch.priority=urgent \
    --launch.num_gpus=8 \
    --launch.num_nodes=1 \
    --trainer.callbacks.wandb.project=20260513_candidate_datasets \
    --trainer.callbacks.wandb.name=basev11_candidateonly_sparse_infill_100k_divp95

python3 scripts/candidate_ablations/run_candidate_ablation_single_bandset.py launch basev11_candidateonly_xlocal_bridge_100k_divp95 ai2/jupiter-cirrascale-2 \
    --candidate_only \
    --candidate_columns xlocal_bridge \
    --score_suffix diverse_score_p95 \
    --select_top 100000 \
    --candidate_parquet /weka/dfive-default/rslearn-eai/datasets/globe_land_grid/s50ix24_embeddings/_scores/combined_acquisition_scores.parquet \
    --candidate_h5py_dir /weka/dfive-default/helios/dataset/candidates/h5py_data_w_missing_timesteps_zstd_3_128_x_1/cdl_landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcereal_worldcover_wri_canopy_height_map/693942 \
    --data_loader.num_dataset_repeats_per_epoch=11 \
    --train_module.optim_config.lr=0.0001 \
    --train_module.scheduler.warmup_steps=8000 \
    --train_module.scheduler.alpha_f=0.1 \
    --train_module.scheduler.t_max=500000 \
    --trainer.max_duration.value=500000 \
    --trainer.max_duration.unit=steps \
    --launch.priority=urgent \
    --launch.num_gpus=8 \
    --launch.num_nodes=1 \
    --trainer.callbacks.wandb.project=20260513_candidate_datasets \
    --trainer.callbacks.wandb.name=basev11_candidateonly_xlocal_bridge_100k_divp95

python3 scripts/candidate_ablations/run_candidate_ablation_single_bandset.py launch basev11_candidateonly_prototypes_100k_divp95 ai2/jupiter-cirrascale-2 \
    --candidate_only \
    --candidate_columns prototypes \
    --score_suffix diverse_score_p95 \
    --select_top 100000 \
    --candidate_parquet /weka/dfive-default/rslearn-eai/datasets/globe_land_grid/s50ix24_embeddings/_scores/combined_acquisition_scores.parquet \
    --candidate_h5py_dir /weka/dfive-default/helios/dataset/candidates/h5py_data_w_missing_timesteps_zstd_3_128_x_1/cdl_landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcereal_worldcover_wri_canopy_height_map/693942 \
    --data_loader.num_dataset_repeats_per_epoch=11 \
    --train_module.optim_config.lr=0.0001 \
    --train_module.scheduler.warmup_steps=8000 \
    --train_module.scheduler.alpha_f=0.1 \
    --train_module.scheduler.t_max=500000 \
    --trainer.max_duration.value=500000 \
    --trainer.max_duration.unit=steps \
    --launch.priority=urgent \
    --launch.num_gpus=8 \
    --launch.num_nodes=1 \
    --trainer.callbacks.wandb.project=20260513_candidate_datasets \
    --trainer.callbacks.wandb.name=basev11_candidateonly_prototypes_100k_divp95



# ALL5

python3 scripts/candidate_ablations/run_candidate_ablation_single_bandset.py launch basev11_200k_all5_top50k_divp95 ai2/jupiter-cirrascale-2 \
    --candidate_columns novelty xglobal_bridge sparse_infill xlocal_bridge prototypes \
    --score_suffix diverse_score_p95 \
    --select_top 50000 \
    --candidate_parquet /weka/dfive-default/rslearn-eai/datasets/globe_land_grid/s50ix24_embeddings/_scores/combined_acquisition_scores.parquet \
    --candidate_h5py_dir /weka/dfive-default/helios/dataset/candidates/h5py_data_w_missing_timesteps_zstd_3_128_x_1/cdl_landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcereal_worldcover_wri_canopy_height_map/693942 \
    --trainer.load_path=/weka/dfive-default/helios/checkpoints/favyen/hidden1/step200000 \
    --train_module.optim_config.lr=0.0001 \
    --train_module.scheduler.warmup_steps=0 \
    --train_module.scheduler.alpha_f=0.1 \
    --train_module.scheduler.t_max=400000 \
    --trainer.max_duration.value=400000 \
    --trainer.max_duration.unit=steps \
    --launch.priority=urgent \
    --launch.num_gpus=8 \
    --launch.num_nodes=1 \
    --trainer.callbacks.wandb.project=20260513_candidate_datasets \
    --trainer.callbacks.wandb.name=basev11_200k_all5_top50k_divp95





# ATTEMPTS

### v0a  -  xlocal_bridge only (+50k)
python3 scripts/candidate_ablations/run_candidate_ablation_single_bandset.py launch v0a_basev11_xglobal_bridge50k ai2/jupiter-cirrascale-2 \
    --candidate_columns xglobal_bridge \
    --score_suffix normalized_score \
    --select_top 50000 \
    --candidate_parquet /weka/dfive-default/rslearn-eai/datasets/globe_land_grid/s50ix24_embeddings/_scores/combined_acquisition_scores.parquet \
    --candidate_h5py_dir /weka/dfive-default/helios/dataset/candidates/h5py_data_w_missing_timesteps_zstd_3_128_x_1/cdl_landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcereal_worldcover_wri_canopy_height_map/693942 \
    --train_module.optim_config.lr=0.0001 \
    --train_module.scheduler.warmup_steps=8000 \
    --train_module.scheduler.alpha_f=0.1 \
    --train_module.scheduler.t_max=500000 \
    --trainer.max_duration.value=500000 \
    --trainer.max_duration.unit=steps \
    --launch.priority=urgent \
    --launch.num_gpus=8 \
    --launch.num_nodes=1 \
    --trainer.callbacks.wandb.project=20260513_candidate_datasets \
    --trainer.callbacks.wandb.name=v0a_basev11_xglobal_bridge50k


### v0b  -  xlocal_bridge only (+100k)
python3 scripts/candidate_ablations/run_candidate_ablation_single_bandset.py launch v0b_basev11_xglobal_bridge100k ai2/jupiter-cirrascale-2 \
    --candidate_columns xglobal_bridge \
    --score_suffix normalized_score \
    --select_top 100000 \
    --candidate_parquet /weka/dfive-default/rslearn-eai/datasets/globe_land_grid/s50ix24_embeddings/_scores/combined_acquisition_scores.parquet \
    --candidate_h5py_dir /weka/dfive-default/helios/dataset/candidates/h5py_data_w_missing_timesteps_zstd_3_128_x_1/cdl_landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcereal_worldcover_wri_canopy_height_map/693942 \
    --train_module.optim_config.lr=0.0001 \
    --train_module.scheduler.warmup_steps=8000 \
    --train_module.scheduler.alpha_f=0.1 \
    --train_module.scheduler.t_max=500000 \
    --trainer.max_duration.value=500000 \
    --trainer.max_duration.unit=steps \
    --launch.priority=urgent \
    --launch.num_gpus=8 \
    --launch.num_nodes=1 \
    --trainer.callbacks.wandb.project=20260513_candidate_datasets \
    --trainer.callbacks.wandb.name=v0b_basev11_xglobal_bridge100k

### v1a  -  all candidate samples (random with 700k budget > total available, uses all)
python3 scripts/candidate_ablations/run_candidate_ablation_single_bandset.py launch basev11_200k_random_all ai2/jupiter-cirrascale-2 \
    --random_seed 42 \
    --total_budget 700000 \
    --candidate_h5py_dir /weka/dfive-default/helios/dataset/candidates/h5py_data_w_missing_timesteps_zstd_3_128_x_1/cdl_landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcereal_worldcover_wri_canopy_height_map/693942 \
    --trainer.load_path=/weka/dfive-default/helios/checkpoints/favyen/hidden1/step200000 \
    --train_module.optim_config.lr=0.0001 \
    --train_module.scheduler.warmup_steps=0 \
    --train_module.scheduler.alpha_f=0.1 \
    --train_module.scheduler.t_max=400000 \
    --trainer.max_duration.value=400000 \
    --trainer.max_duration.unit=steps \
    --launch.priority=urgent \
    --launch.num_gpus=8 \
    --launch.num_nodes=1 \
    --trainer.callbacks.wandb.project=20260513_candidate_datasets \
    --trainer.callbacks.wandb.name=basev11_200k_random_all