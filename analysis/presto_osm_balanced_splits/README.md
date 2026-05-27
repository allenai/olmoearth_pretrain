# Presto OSM Balanced Eval Splits

This directory contains candidate OpenStreetMap raster eval splits built from the
Presto H5 corpus:

`/weka/dfive-default/helios/dataset/presto/h5py_data_w_missing_timesteps_zstd_3_128_x_4/cdl_landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcereal_worldcover_wri_canopy_height_map/469728`

The splits are generated from cached per-sample OSM label metadata and are wired
into `TILING_DIAG_TASKS` in `olmoearth_pretrain/internal/all_evals.py`.

## Variants

- `osm_base_balanced`: base class-balanced OSM eval. Anchors on classes with at
  least 500 tiles present. Current size: 6144 train / 3072 valid / 3072 test.
- `osm_diverse_context`: context-heavy eval. Requires at least 3 OSM classes per
  tile and normalized entropy >= 0.5. Current size: 6144 train / 2093 valid /
  2089 test.
- `osm_rare_class_focused`: rare-class diagnostic eval. Anchors on classes with
  more than 50 tiles present. Highway/building caps apply to filler/context tiles
  but rare-anchor tiles are exempt. Current size: 1290 train / 200 valid / 175 test.

Each variant contains:

- `train.csv`, `valid.csv`, `test.csv`: selected H5 `sample_index` values.
- `*_class_summary.csv`: class presence and pixel distribution for that split.

## Eval Command

Run only these three OSM balanced evals with:

```bash
TILING_DIAGNOSTICS_ONLY=1 \
TRAIN_SCRIPT_PATH=scripts/official/v1_1/base.py \
CHECKPOINT_DIR=/weka/dfive-default/helios/checkpoints/favyen/hidden1 \
CHECKPOINT_STEPS=665000 \
torchrun olmoearth_pretrain/internal/checkpoint_sweep_evals.py \
  evaluate v1_1_hidden1_osm_balanced_tiling_diag_step665k local \
  --trainer.callbacks.wandb.project=2026_05_27_tiling_diag_rope
```

## Supporting Artifacts

- `analysis/presto_osm_balancing/`: corpus-level OSM class presence and
  interesting-tile analysis.
- `analysis/presto_osm_balanced_split_distributions/`: class distribution plots
  for each generated split variant.
- `analysis/presto_osm_balanced_split_geographies/`: geography plots and
  lat/lon-joined CSVs for judging geographic holdout viability.

The large per-sample metadata caches are not required to run the evals; they are
only needed to regenerate or redesign splits.
