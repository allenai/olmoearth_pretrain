These are for testing 8-bit, 4-bit, 2-bit, and 1-bit quantization when using the
distribution of embeddings computed from the pre-training dataset to compute the
quantization buckets, along with the medians to de-quantize to.

## Compute quantile boundaries from pre-training embeddings

Extract embeddings from the pre-training dataset and compute percentile boundaries:

```bash
python scripts/archived/2026-01-28_percentile_quant_experiment/compute_embedding_quantiles.py \
    --checkpoint_path=/weka/dfive-default/helios/checkpoints/joer/phase2.0_base_lr0.0001_wd0.02/step667200 \
    --h5py_dir=/weka/dfive-default/helios/dataset/osm_sampling/h5py_data_w_missing_timesteps_zstd_3_128_x_4/cdl_gse_landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcereal_worldcover_worldpop_wri_canopy_height_map/1138828 \
    --output=scripts/archived/2026-01-28_percentile_quant_experiment/quantiles.h5 \
    --num_samples=10000 \
    --batch_size=32
```

`quantiles.h5` will need to be committed so it can be used in the eval jobs.

## Run the eval sweep

Launch baseline and quantized eval runs on beaker:

```bash
./scripts/archived/2026-01-28_percentile_quant_experiment/run_percentile_quant_sweep.sh
```

## Fetch results from wandb

After runs complete, pull metrics and generate comparison CSV:

```bash
python scripts/archived/2026-01-28_percentile_quant_experiment/compare_percentile_quant.py
```

Outputs `percentile_quant_comparison.csv` with per-task metrics for each quantization level.
