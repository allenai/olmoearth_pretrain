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
> `data_factory`). Training-side concerns (masked-softmax loss, per-class vectors, model
> wiring) are out of scope here — this only produces the dataset + class mapping.

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

### 1. Assemble the global class mapping

```bash
python -m olmoearth_pretrain.open_set_segmentation_data.assemble_classes \
    --datasets_root /weka/dfive-default/helios/dataset_creation/open_set_segmentation/datasets
```

Writes `data/open_set_segmentation_data/class_mapping.json` (global class list,
per-training-dataset global-id subsets, presence-only group, regression registry). If any
dataset is described as presence-only in its summary but has a background class, it is
printed as a `WARNING` for manual review.

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
    --workers 32
```

Creates one window per label sample (single group `open_set`) and writes the `open_set` or
`open_set_regression` label layer through `window.data`. Use `--slugs a,b,c` to restrict to
specific datasets (useful for a small verification run).

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
for m in sentinel2_l2a sentinel1 landsat; do
  python -m olmoearth_pretrain.dataset_creation.rslearn_to_olmoearth.open_set_imagery \
      --ds_path open_set_dataset --olmoearth_path /path/to/olmoearth_output --modality $m
done
```

Static modalities reuse their existing converters unchanged, e.g.:

```bash
python -m olmoearth_pretrain.dataset_creation.rslearn_to_olmoearth.worldcover \
    --ds_path open_set_dataset --olmoearth_path /path/to/olmoearth_output
# ...and srtm, cdl, worldcereal, wri_canopy_height_map, rasterize_openstreetmap
```

Label layers:

```bash
python -m olmoearth_pretrain.dataset_creation.rslearn_to_olmoearth.open_set \
    --ds_path open_set_dataset --olmoearth_path /path/to/olmoearth_output --layer open_set
python -m olmoearth_pretrain.dataset_creation.rslearn_to_olmoearth.open_set \
    --ds_path open_set_dataset --olmoearth_path /path/to/olmoearth_output --layer open_set_regression
```

Examples are keyed by `example_id` (`{slug}_{sample_id}`), consistent across all
modalities.
