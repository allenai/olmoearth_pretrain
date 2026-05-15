# Candidate ablations

## Strategies (pick any combination)
novelty, xglobal_bridge, sparse_infill, xlocal_bridge, prototypes

Candidates are selected by taking the top `--select_top` samples per strategy
(ranked by `{strategy}_normalized_score` in the raw parquet), then unioning
across strategies. Only samples present in the h5py directory are considered.

Use `--total_budget` instead of `--select_top` to keep the total candidate
pool size constant across ablations (budget is divided evenly across strategies).


# Run ablations -- single-bandset recipe

## Baseline: all 5 strategies, top 50k each (~250k candidates)
```shell
python3 scripts/candidate_ablations/run_candidate_ablation_single_bandset.py launch basev11_200k_all5_top50k ai2/jupiter-cirrascale-2 \
    --candidate_columns novelty xglobal_bridge sparse_infill xlocal_bridge prototypes \
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
    --trainer.callbacks.wandb.name=basev11_200k_all5_top50k
```

## Drop-one ablation (e.g. drop novelty, size-controlled via --total_budget)
```shell
python3 scripts/candidate_ablations/run_candidate_ablation_single_bandset.py launch basev11_200k_drop_novelty ai2/jupiter-cirrascale-2 \
    --candidate_columns xglobal_bridge sparse_infill xlocal_bridge prototypes \
    --total_budget 250000 \
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
    --trainer.callbacks.wandb.name=basev11_200k_drop_novelty
```

## Solo strategy (e.g. novelty only)
```shell
python3 scripts/candidate_ablations/run_candidate_ablation_single_bandset.py launch basev11_200k_solo_novelty ai2/jupiter-cirrascale-2 \
    --candidate_columns novelty \
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
    --trainer.callbacks.wandb.name=basev11_200k_solo_novelty
```


# Run ablations (v1) -- original multi-bandset recipe

```shell
python3 scripts/candidate_ablations/run_candidate_ablation.py launch base200k_all5 ai2/jupiter-cirrascale-2 \
    --candidate_columns novelty xglobal_bridge sparse_infill xlocal_bridge prototypes \
    --select_top 50000 \
    --candidate_parquet /weka/dfive-default/rslearn-eai/datasets/globe_land_grid/s50ix24_embeddings/_scores/combined_acquisition_scores.parquet \
    --candidate_h5py_dir /weka/dfive-default/helios/dataset/candidates/h5py_data_w_missing_timesteps_zstd_3_128_x_1/cdl_landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcereal_worldcover_wri_canopy_height_map/693942 \
    --trainer.load_path=/weka/dfive-default/helios/checkpoints/joer/phase2.0_base_lr0.0001_wd0.02/step200000 \
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
    --trainer.callbacks.wandb.name=base200k_all5
```


# BASE Annealing (v1)

```shell
    python3 scripts/official/base.py launch base200k_v1 ai2/jupiter-cirrascale-2 \
    --data_loader.global_batch_size=512 \
    --trainer.load_path=/weka/dfive-default/helios/checkpoints/joer/phase2.0_base_lr0.0001_wd0.02/step200000 \
    --train_module.optim_config.lr=0.0001 \
    --train_module.scheduler.warmup_steps=0 \
    --train_module.scheduler.alpha_f=0.1 \
    --train_module.scheduler.t_max=400000 \
    --trainer.max_duration.value=400000 \
    --trainer.max_duration.unit=steps \
    --launch.num_gpus=8 \
    --launch.num_nodes=1 \
    --launch.priority=urgent \
    --trainer.callbacks.wandb.project=20260513_candidate_datasets \
    --trainer.callbacks.wandb.name=base200k_v1
```

# BASE Annealing (v1.1)

```shell
python3 scripts/archived/2026_04_22_add_hidden_layer_to_initial_projection/hidden1.py launch base200k_v11 ai2/jupiter-cirrascale-2 \
    --data_loader.global_batch_size=512 \
    --trainer.load_path=/weka/dfive-default/helios/checkpoints/favyen/hidden1/step200000 \
    --train_module.optim_config.lr=0.0001 \
    --train_module.scheduler.warmup_steps=0 \
    --train_module.scheduler.alpha_f=0.1 \
    --train_module.scheduler.t_max=400000 \
    --trainer.max_duration.value=400000 \
    --trainer.max_duration.unit=steps \
    --launch.num_gpus=8 \
    --launch.num_nodes=1 \
    --launch.priority=urgent \
    --trainer.callbacks.wandb.project=20260513_candidate_datasets \
    --trainer.callbacks.wandb.name=base200k_v11
```