# Open-Set Segmentation Pretraining Dataset

This module turns the **open-set label bank** (~300 diverse remote-sensing label
datasets, ingested into a consistent on-disk format) into an **OlmoEarth Pretrain
dataset**, pairing each label sample with satellite imagery.

It differs from the grid-based pretraining pipeline in `dataset_creation/`:

- Windows are **128×128 at 10 m/pixel**, **centered on each label sample** (not snapped
  to a global grid), with **per-sample time ranges**.
- Materialized data uses rslearn's **`PerLayerStorageFactory`** (one file per layer).
- Each window carries a combined **`open_set`** label layer (single-band uint16, globally
  unique class ids, nodata `65535`) or, for regression datasets, a two-band
  **`open_set_regression`** layer (band 0 = 1-based dataset id, band 1 = value linearly
  remapped to `[1, 65535]`, `0` = nodata).

> Requires the newest `rslearn` (`window.data` / `PerLayerStorageFactory` /
> `data_factory`). Single-window classification and regression training are implemented;
> paired pre/post change training remains a separate follow-up.

## Background

- **Label bank format**: see [`../../data/open_set_segmentation_data/AGENT_SUMMARY.md`](../../data/open_set_segmentation_data/AGENT_SUMMARY.md).
  The bulk outputs live on weka under `OUTPUT_ROOT` (see [manifest.py](manifest.py)):
  `datasets/{slug}/metadata.json`, dense `datasets/{slug}/locations/{id}.tif`+`.json`, or
  sparse `datasets/{slug}/points.geojson`.
- **Registry / status**: [`../../data/open_set_segmentation_data/registry.json`](../../data/open_set_segmentation_data/registry.json).
  Only datasets with status `completed` are used.

## Key modules

| File | Purpose |
|------|---------|
| [assemble_classes.py](assemble_classes.py) | Build the global class-id space + regression registry → `class_mapping.json`. |
| [pretrain_constants.py](pretrain_constants.py) | Window geometry, layer names, nodata values, excluded eval slugs. |
| [`../dataset_creation/create_windows/from_open_set.py`](../dataset_creation/create_windows/from_open_set.py) | Create centered windows and write the label layers. |
| [`../dataset_creation/create_windows/generate_eval_exclusion_geojson.py`](../dataset_creation/create_windows/generate_eval_exclusion_geojson.py) | Build the val/test exclusion GeoJSON (PASTIS, yemen_crop). |
| [`../dataset_creation/rslearn_to_olmoearth/open_set.py`](../dataset_creation/rslearn_to_olmoearth/open_set.py) | Convert a label layer to the OlmoEarth Pretrain format. |

## Eval contamination handling

To avoid pretraining on held-out evaluation data:

- **eurosat** and **so2sat_lcz42** are dropped at the dataset level (`EXCLUDED_SLUGS` in
  [pretrain_constants.py](pretrain_constants.py)) — they have no reliable per-sample
  geocoordinates.
- **PASTIS** and **yemen_crop** are excluded *geographically*: their val/test extents are
  written to an exclusion GeoJSON and any pretraining window whose footprint intersects a
  polygon is skipped.
- **MADOS** is not in the label bank (rejected) and has no geocoordinates, so it needs no
  handling.

## Steps to run

All commands run from the repo root with the `olmoearth_pretrain` venv active.
Steps 1–2 can run locally if the label bank is mounted; window creation, materialization,
and conversion need weka access.

### 1. Freeze the global class mapping

The checked-in `data/open_set_segmentation_data/class_mapping.json` is the authoritative
mapping for the dataset currently being built. Its numeric IDs are embedded directly in
the label rasters, so **do not regenerate or overwrite it** after window creation starts.
The matching `class_mapping.sha256` is verified by the official training configuration.

For a future, deliberately versioned dataset build, generate a candidate at a different
path first:

```bash
python -m olmoearth_pretrain.open_set_segmentation_data.assemble_classes \
  --datasets_root /weka/dfive-default/helios/dataset_creation/open_set_segmentation/datasets \
  --output /tmp/class_mapping.candidate.json
```

Only use a changed candidate with a fresh label/H5 build. The assembly command refuses to
replace an existing output unless `--overwrite` is passed explicitly.

### 2. (Optional) Build the eval exclusion GeoJSON

```bash
python -m olmoearth_pretrain.dataset_creation.create_windows.generate_eval_exclusion_geojson \
    --pastis_metadata /path/to/PASTIS-R/metadata.geojson \
    --yemen_ds_path   /weka/dfive-default/olmoearth/eval_datasets/yemen_crop
```

Writes `data/open_set_segmentation_data/eval_exclusion.geojson` (WGS84; folds 4/5 of
PASTIS + val/test windows of yemen_crop). yemen_crop stores its split under the
`eval_split` tag (the generator's default `--yemen_split_tag_key`).

### 3. Initialize the rslearn dataset

```bash
mkdir open_set_dataset
cp data/rslearn_dataset_configs/config_open_set.json open_set_dataset/config.json
```

### 4. Create windows + write label layers

```bash
python -m olmoearth_pretrain.dataset_creation.create_windows.from_open_set \
    --ds_path open_set_dataset \
    --class_mapping data/open_set_segmentation_data/class_mapping.json \
    --exclude_geojson data/open_set_segmentation_data/eval_exclusion.geojson \
    --paired_change_policy skip \
    --workers 32
```

Creates one window per label sample (single group `open_set`) and writes the `open_set` or
`open_set_regression` label layer through `window.data`. Use `--slugs a,b,c` to restrict to
  specific datasets (useful for a small verification run). This single-window build cannot
  represent paired pre/post change samples: the default policy is `error`; the explicit
  `skip` above excludes them and reports `skipped_paired_change` in the final counts.

### 5. Materialize imagery

`config_open_set.json` is a single consolidated config containing every v1.2 base
modality, so materialize once (no per-modality config copying). The multitemporal layers
(`sentinel2_l2a`, `sentinel1`, `landsat`) are each a single `MOSAIC` layer with
`period_duration=30d` + `include_partial_periods`, so every window gets one mosaic per
~30-day period **of its own time range** — a sub-30-day label → one mosaic, a 3-month
label → 3, an annual label → 12.

```bash
export DATASET_PATH=./open_set_dataset
rslearn dataset prepare     --root $DATASET_PATH --group open_set --workers 64
rslearn dataset ingest      --root $DATASET_PATH --group open_set --workers 64
rslearn dataset materialize --root $DATASET_PATH --group open_set --workers 64
```

Per-layer `ingest` flags are honored (worldcereal/openstreetmap ingest; the imagery
layers direct-materialize). The `open_set` / `open_set_regression` label layers were
written in step 4 and need no materialization.

### 6. Convert to the OlmoEarth Pretrain format

Multitemporal imagery (period mosaics) — one run per modality:

```bash
export DATASET_PATH=/weka/dfive-default/helios/dataset_creation/open_set_dataset
export OLMOEARTH_PATH=/weka/dfive-default/helios/dataset/open_set_dataset

for m in sentinel2_l2a sentinel1 landsat; do
  python -m olmoearth_pretrain.dataset_creation.rslearn_to_olmoearth.open_set_imagery \
      --ds_path $DATASET_PATH --olmoearth_path $OLMOEARTH_PATH --modality $m
done
```

Static raster modalities reuse their existing converters, selecting the open-set group
explicitly:

```bash
for m in worldcover srtm cdl worldcereal wri_canopy_height_map; do
  python -m olmoearth_pretrain.dataset_creation.rslearn_to_olmoearth.$m \
      --ds_path $DATASET_PATH --olmoearth_path $OLMOEARTH_PATH --group open_set
done
```

OpenStreetMap is first written as per-window vector GeoJSON, then rasterized. Because
open-set windows are 128 px and centered on each sample (their `col`/`row` are absolute
pixel coordinates, not grid-tile indices), the rasterize step needs
`--window_size 128 --pixel_coord_windows`:

```bash
python -m olmoearth_pretrain.dataset_creation.rslearn_to_olmoearth.openstreetmap \
  --ds_path $DATASET_PATH --olmoearth_path $OLMOEARTH_PATH --group open_set
```

Label layers:

```bash
python -m olmoearth_pretrain.dataset_creation.rslearn_to_olmoearth.open_set \
  --ds_path $DATASET_PATH --olmoearth_path $OLMOEARTH_PATH --layer open_set
python -m olmoearth_pretrain.dataset_creation.rslearn_to_olmoearth.open_set \
  --ds_path $DATASET_PATH --olmoearth_path $OLMOEARTH_PATH --layer open_set_regression
```

Create the per-modality metadata summaries. OpenStreetMap must be summarized before it is
rasterized because rasterization reads each centered window's CRS, center pixel, and
`example_id` from this CSV.

```bash
for m in sentinel2_l2a sentinel1 landsat; do
  python -m olmoearth_pretrain.dataset_creation.make_meta_summary \
    --olmoearth_path $OLMOEARTH_PATH --modality $m --time_span year
done

for m in worldcover srtm cdl worldcereal wri_canopy_height_map openstreetmap open_set open_set_regression; do
  python -m olmoearth_pretrain.dataset_creation.make_meta_summary \
    --olmoearth_path $OLMOEARTH_PATH --modality $m
done

python -m olmoearth_pretrain.dataset_creation.rslearn_to_olmoearth.rasterize_openstreetmap \
  --olmoearth_path $OLMOEARTH_PATH --window_size 128 --pixel_coord_windows
```

Examples are keyed by `example_id` (`{slug}_{sample_id}`), consistent across all
modalities.

### 7. Create H5s

Finally, convert the OlmoEarth Pretrain dataset into the per-sample H5 files used during
training (better optimized for reads). We use the same compression settings as the v1.2
base script (`zstd`, level 3, `tile_size=128`), but the open-set dataset differs from the
grid pipeline in two ways that require extra flags:

- **`--image_tile_size=128`** — the grid pipeline stores 256×256 tiles and splits each into
  four 128×128 H5 samples. Open-set windows are **already 128×128**, so we set the source
  tile size to 128; with `tile_size=128` this yields **one H5 sample per window** (no
  splitting). The output directory is suffixed `..._128_x_1` (vs `..._128_x_4` for the grid
  pipeline).
- **`--pixel_coord_windows=true`** — open-set windows are centered on each sample, so their
  `col`/`row` are absolute pixel coordinates rather than grid-tile indices. This flag makes
  the per-sample latlon be computed from those pixel coordinates.

```bash
python -m olmoearth_pretrain.internal.run_h5_conversion \
    --tile_path=$OLMOEARTH_PATH \
  --supported_modality_names='[cdl,landsat,open_set,open_set_regression,openstreetmap_raster,sentinel1,sentinel2_l2a,srtm,worldcereal,worldcover,wri_canopy_height_map]' \
    --compression=zstd --compression_opts=3 \
    --tile_size=128 --image_tile_size=128 --pixel_coord_windows=true
```

Unlike the grid pipeline, there is **no full-year / 12-timestep requirement**: each window
keeps exactly the period mosaics of its own label time range (1 for a sub-30-day label, up
to 12 for an annual one). The training data loader pads every sample to 12 months and masks
the absent timesteps, so variable-length samples are handled without any training-side
change.

### 8. Train

The supervised probe pools visible online-encoder tokens by spatial patch. Classification
labels are majority-pooled to that patch grid and use exact cross-entropy over the source
dataset's allowed classes; presence-only overlap conflicts are excluded as target-specific
negatives. Regression labels use the valid-pixel patch mean and a dataset-specific linear
head. Label modalities are loaded as supervision but are never tokenized by the encoder.

Two official launch paths are provided:

```bash
# Open-set dataset only.
python scripts/official/v1_2/open_set_only.py launch open_set_only ai2/jupiter \
    --launch.num_gpus=8

# Existing osm_sampling corpus plus the open-set corpus, sampled by dataset length.
python scripts/official/v1_2/open_set_osm.py launch open_set_osm ai2/jupiter \
    --launch.num_gpus=8
```

The configured mapping hash must match `class_mapping.sha256`; a mismatch fails before the
probe is built. Regression datasets with invalid ranges in the frozen mapping are ignored
by the supervised loss rather than reinterpreting already-encoded uint16 labels.
