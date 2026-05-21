# Candidate ablations

## Strategies (pick any combination)
novelty, xglobal_bridge, sparse_infill, xlocal_bridge, prototypes

Candidates are selected by taking the top `--select_top` samples per strategy
(ranked by `{strategy}_{score_suffix}` in the parquet), then unioning
across strategies. Only samples present in the h5py directory are considered.

Use `--total_budget` instead of `--select_top` to keep the total candidate
pool size constant across ablations (budget is divided evenly across strategies).

## Score suffix (`--score_suffix`, required)
Controls which column is used to rank candidates. The resolved column name is
`{strategy}_{score_suffix}` — e.g. `--score_suffix diverse_score_p95` looks up
`novelty_diverse_score_p95`, `xglobal_bridge_diverse_score_p95`, etc.

Available suffixes:
- `normalized_score` — original min-max normalized strategy scores
- `diverse_score_p95` — greedy max-score with cosine-similarity exclusion (p95 threshold)

Future thresholds (e.g. `diverse_score_p90`) will follow the same naming pattern.


## Random baseline (`--random_seed`)

Instead of selecting top scorers by strategy, randomly sample from all available
h5 samples. Use `--random_seed` to activate this mode:
- `--random_seed <int>` — seed for reproducibility (activates random mode)
- `--total_budget <int>` — how many samples to select (required with `--random_seed`)
- `--candidate_h5py_dir` — source of available samples (required)
- `--candidate_columns`, `--score_suffix`, `--candidate_parquet` are ignored

If `--total_budget` exceeds the number of available samples, all samples are used.

```shell
python3 scripts/candidate_ablations/run_candidate_ablation_single_bandset.py launch basev11_200k_randcand_s42_50k ai2/jupiter-cirrascale-2 \
    --random_seed 42 \
    --total_budget 50000 \
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
    --trainer.callbacks.wandb.name=basev11_200k_randcand_s42_50k
```


# Run ablations -- single-bandset recipe

## Baseline: all 5 strategies, top 50k each (~250k candidates)
```shell
python3 scripts/candidate_ablations/run_candidate_ablation_single_bandset.py launch basev11_200k_all5_top50k ai2/jupiter-cirrascale-2 \
    --candidate_columns novelty xglobal_bridge sparse_infill xlocal_bridge prototypes \
    --score_suffix normalized_score \
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
    --score_suffix normalized_score \
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
    --score_suffix normalized_score \
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

## Diverse scoring (p95 threshold): all 5 strategies, top 50k each
```shell
python3 scripts/candidate_ablations/run_candidate_ablation_single_bandset.py launch basev11_200k_all5_diverse_p95_top50k ai2/jupiter-cirrascale-2 \
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
    --trainer.callbacks.wandb.name=basev11_200k_all5_diverse_p95_top50k
```


# Run ablations (v1) -- original multi-bandset recipe

```shell
python3 scripts/candidate_ablations/run_candidate_ablation.py launch base200k_all5 ai2/jupiter-cirrascale-2 \
    --candidate_columns novelty xglobal_bridge sparse_infill xlocal_bridge prototypes \
    --score_suffix normalized_score \
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
