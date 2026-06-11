# Presto OSM Balanced Eval Splits

This directory is the canonical in-repo copy of the OpenStreetMap raster eval
splits built from the Presto H5 corpus:

`/weka/dfive-default/helios/dataset/presto/h5py_data_w_missing_timesteps_zstd_3_128_x_4/cdl_landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcereal_worldcover_wri_canopy_height_map/469728`

The splits are generated from cached per-sample OSM label metadata and are wired
into `EVAL_TASKS` in `olmoearth_pretrain/internal/all_evals.py` as the three
`presto_osm_*` probe tasks. Filter to just these with `--task-names`.

## Variants

- `osm_base_balanced`: populous-class OSM segmentation eval. Anchors on classes
  with at least 500 tiles present in the Presto corpus. Current size: 6144 train
  / 3072 valid / 3072 test. Eval remaps the 12 populous OSM classes to
  contiguous labels and ignores all other OSM pixels.
- `osm_diverse_context`: context-heavy eval. Requires at least 3 OSM classes per
  tile and normalized entropy >= 0.5. Current size: 6144 train / 2093 valid /
  2089 test. Eval task predicts 30-way multi-label OSM class presence for each
  tile.
- `osm_rare_class_focused`: rare-class diagnostic eval. Anchors on classes with
  more than 50 tiles present. Highway/building caps apply to filler/context tiles
  but rare-anchor tiles are exempt. Current size: 1227 train / 200 valid / 175 test.
  Eval remaps the feasible rare OSM classes (`fountain`, `generator_wind`,
  `storage_tank`, `taxiway`) to contiguous per-pixel segmentation labels and
  ignores all other OSM pixels.

Each variant contains:

- `train.csv`, `valid.csv`, `test.csv`: selected H5 `sample_index` values.
- `*_class_summary.csv`: class presence and pixel distribution for that split.
- `class_support.json`: precomputed labeled-class ids per split for eval scoring.
  Uses `tile_presence` for tile-classification tasks and `pixels` for segmentation.

## Eval Command

Validate config generation first:

```bash
python -m olmoearth_pretrain.internal.full_eval_sweep \
  --dry_run \
  --cluster=ai2/saturn-cirrascale \
  --defaults_only \
  --module_path=scripts/official/v1_1/base.py \
  --checkpoint_path=/weka/dfive-default/helios/checkpoints/favyen/hidden1/step665000 \
  --task-names=presto_osm_populous12_seg_probe_sentinel2_l2a,presto_osm_diverse_context_probe_sentinel2_l2a,presto_osm_rare4_seg_probe_sentinel2_l2a \
  --project_name=2026_06_01_osm_balanced_eval
```

Launch to Beaker (push branch first):

```bash
python -m olmoearth_pretrain.internal.full_eval_sweep \
  --cluster=ai2/saturn-cirrascale \
  --defaults_only \
  --module_path=scripts/official/v1_1/base.py \
  --checkpoint_path=/weka/dfive-default/helios/checkpoints/favyen/hidden1/step665000 \
  --task-names=presto_osm_populous12_seg_probe_sentinel2_l2a,presto_osm_diverse_context_probe_sentinel2_l2a,presto_osm_rare4_seg_probe_sentinel2_l2a \
  --project_name=2026_06_01_osm_balanced_eval
```

Local run (single GPU):

```bash
python -m olmoearth_pretrain.internal.full_eval_sweep \
  --cluster=local \
  --defaults_only \
  --module_path=scripts/official/v1_1/base.py \
  --checkpoint_path=/weka/dfive-default/helios/checkpoints/favyen/hidden1/step665000 \
  --task-names=presto_osm_populous12_seg_probe_sentinel2_l2a,presto_osm_diverse_context_probe_sentinel2_l2a,presto_osm_rare4_seg_probe_sentinel2_l2a \
  --project_name=2026_06_01_osm_balanced_eval
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
