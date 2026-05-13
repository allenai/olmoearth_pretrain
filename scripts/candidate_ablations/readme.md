# Run ablations (v1)


## combinations (pick any or any combination)
in_top_combined, in_top_solo_novelty, in_top_solo_xglobal_bridge,
in_top_solo_sparse_infill, in_top_solo_xlocal_bridge, in_top_solo_prototypes,
in_top_drop_novelty, in_top_drop_xglobal_bridge, in_top_drop_sparse_infill,
in_top_drop_xlocal_bridge, in_top_drop_prototypes


```shell
python3 scripts/candidate_ablations/run_candidate_ablation.py launch base_solo_novelty ai2/jupiter-cirrascale-2 \
    --candidate_columns in_top_solo_novelty \
    --candidate_parquet /weka/dfive-default/rslearn-eai/datasets/globe_land_grid/s50ix24_embeddings/_scores/selection_top250000.parquet \
    --candidate_h5py_dir /weka/dfive-default/helios/dataset/candidates/h5py_data_w_missing_timesteps_zstd_3_128_x_1/cdl_landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcereal_worldcover_wri_canopy_height_map/693942 \
    --trainer.load_path=/weka/dfive-default/helios/checkpoints/joer/phase2.0_base_lr0.0001_wd0.02 \
    --train_module.optim_config.lr=0.00008 \
    --train_module.scheduler.warmup_steps=0 \
    --train_module.scheduler.alpha_f=0.125 \
    --train_module.scheduler.t_max=200000 \
    --trainer.max_duration.value=200000 \
    --trainer.max_duration.unit=steps \
    --launch.priority=urgent \
    --launch.num_gpus=8 \
    --launch.num_nodes=1 \
    --trainer.callbacks.wandb.project=20260513_candidate_datasets \
    --trainer.callbacks.wandb.name=base_solo_novelty
```


# BASE Annealing (v1)

```shell
    python3 scripts/official/base.py launch base_masked_all ai2/jupiter-cirrascale-2 \
    --data_loader.global_batch_size=512 \
    --trainer.load_path=/weka/.../checkpoint/step_XXXXX \
    --train_module.optim_config.lr=0.0008 \
    --train_module.scheduler.warmup_steps=0 \
    --train_module.scheduler.alpha_f=0.125 \
    --train_module.scheduler.t_max=200000 \
    --trainer.max_duration.value=200000 \
    --trainer.max_duration.unit=steps \
    --launch.num_gpus=8 \
    --launch.num_nodes=1 \
    --launch.priority=urgent \
    --trainer.callbacks.wandb.project=20260513_candidate_datasets \
    --trainer.callbacks.wandb.name=base_sigreg_w005_s256_cont0_masked_all \
    --train_module.loss_config.loss_config.type=modality_patch_discrimination_masked_negatives \
    --train_module.loss_config.loss_config.tau=0.1 \
    --train_module.loss_config.loss_config.mask_negatives_for_modalities='["WORLDCOVER","SRTM","OPENSTREETMAP_RASTER","WRI_CANOPY_HEIGHT_MAP","CDL","WORLDCEREAL"]' \
    --train_module.loss_config.loss_config.same_target_threshold=0.999
```


# Run ablations (v1.1)

python3 scripts/candidate_ablations/run_candidate_ablation_single_bandset.py launch <run_name> <cluster> \
    --candidate_columns in_top_solo_novelty \
    --candidate_parquet /path/to/scored_candidates.parquet \
    --trainer.load_path=...
