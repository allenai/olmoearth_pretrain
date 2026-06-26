# ERA5 Climate-Stratified Sampling Dataset

Global ERA5-Land daily sampling dataset with climate-stratified spatial coverage.
Replaces OSM-biased sampling with a globally diverse set stratified by
Köppen-Geiger climate class × elevation band × latitude band.

## Overview

- **250k primary windows** + ~62.5k temporal-overlap secondary windows
- **Rare-class oversampling**: classes below 5k windows get temporal oversampling
  (up to 8 disjoint 448-day slots per cell, capped at 5k per class)
- **Spatial**: 10 km adaptive land grid, one point per ERA5-Land 0.1° cell
- **Temporal**: 448-day windows uniformly in [2016-01-01, 2026-05-31]
- **Stratification**: Power allocation (α=0.5) across Köppen×elev×lat bins
- **Overlap pairs**: 25% of primaries get a second window at the same ERA5 cell
  with 100-day temporal overlap (offset 5 km for future S2 diversity)

## Paths

| What | Path |
|------|------|
| Scripts | `olmoearth_pretrain/dataset_creation/era5_sample/` |
| Metadata / intermediates | `/weka/dfive-default/helios/dataset/era5enc_pretrain/metadata` |
| rslearn dataset (windows + config) | `/weka/dfive-default/helios/dataset/era5enc_pretrain/rslearn_dataset` |

## Run Order

### Step 1: Download stratification sources

```bash
python -m olmoearth_pretrain.dataset_creation.era5_sample.fetch_stratification_sources \
    --output /weka/dfive-default/helios/dataset/era5enc_pretrain/metadata
```

Downloads:
- Köppen-Geiger 0.1° GeoTIFF (Beck et al. 2023)
- Natural Earth 10m land polygons
- ETOPO2022 60-second global elevation GeoTIFF

### Step 2: Build candidate grid

```bash
python -m olmoearth_pretrain.dataset_creation.era5_sample.build_candidate_grid \
    --metadata-dir /weka/dfive-default/helios/dataset/era5enc_pretrain/metadata
```

Outputs `candidates.parquet`: ~150k+ land cells with Köppen/elevation/latitude labels.

### Step 3: Stratified sampling

```bash
python -m olmoearth_pretrain.dataset_creation.era5_sample.stratified_sample \
    --metadata-dir /weka/dfive-default/helios/dataset/era5enc_pretrain/metadata \
    --target-count 250000 \
    --overlap-fraction 0.25 \
    --seed 42
```

Outputs `selected.parquet`: 250k primary + ~62.5k overlap secondary windows.

### Step 4: Generate rslearn windows

```bash
python -m olmoearth_pretrain.dataset_creation.era5_sample.era5_window_generation \
    --metadata-dir /weka/dfive-default/helios/dataset/era5enc_pretrain/metadata \
    --output /weka/dfive-default/helios/dataset/era5enc_pretrain/rslearn_dataset \
    --seed 42 \
    --workers 32 \
    --fresh
```

Use `--fresh` on first run to skip existence checks (much faster).
Omit `--fresh` on subsequent runs for safe resume after interruption.

### Step 4b: Oversample rare Köppen classes

Classes with very few land cells (e.g. Csc, Cwc, Dsd) get boosted by filling
all disjoint 448-day temporal slots at their existing locations (up to 8 per cell).

```bash
python -m olmoearth_pretrain.dataset_creation.era5_sample.oversample_rare_classes \
    --metadata-dir /weka/dfive-default/helios/dataset/era5enc_pretrain/metadata \
    --output /weka/dfive-default/helios/dataset/era5enc_pretrain/rslearn_dataset \
    --threshold 5000 \
    --max-per-class 5000 \
    --workers 32 \
    --fresh
```

This adds windows with `role: "rare_oversample"` into the same dataset. Any class
below `--threshold` total windows gets oversampled up to `--max-per-class`.

### Step 4c: Visualization smoke check

```bash
python -m olmoearth_pretrain.dataset_creation.era5_sample.visualize_windows \
    --ds-path /weka/dfive-default/helios/dataset/era5enc_pretrain/rslearn_dataset \
    --output /weka/dfive-default/helios/dataset/era5enc_pretrain/metadata/sampling_smokecheck.png
```

Produces a figure with:
- Global scatter map of window centers (rows 1-2), colored by role
- Histograms of windows per Köppen class, elevation band, and latitude band (row 3)

Use `--max-windows 50000` for a quick subset check.

### Step 5: rslearn prepare / ingest / materialize

The rslearn dataset config is at
`/weka/dfive-default/helios/dataset/era5enc_pretrain/rslearn_dataset/config.json`.

Requires `EARTHDATAHUB_TOKEN` environment variable (see `.secrets.env`).

```bash
export EARTHDATAHUB_TOKEN="<your-token>"
DS_PATH=/weka/dfive-default/helios/dataset/era5enc_pretrain/rslearn_dataset

# Prepare (match windows to data source items)
python -m rslearn.main prepare --ds_path "$DS_PATH" --group default --workers 32

# Ingest (download ERA5-Land daily data)
python -m rslearn.main ingest --ds_path "$DS_PATH" --group default --workers 16

# Materialize (write final rasters)
python -m rslearn.main materialize --ds_path "$DS_PATH" --group default --workers 32
```

Each step is resumable — rerun safely after interruption.

## Data Layer

**`era5d_448d_historical`**: ERA5-Land daily UTC, 14 bands, 448 timesteps per window.

| Band | Variable | Unit |
|------|----------|------|
| d2m | 2m dewpoint temperature | K |
| e | Total evaporation | m |
| pev | Potential evaporation | m |
| ro | Runoff | m |
| sp | Surface pressure | Pa |
| ssr | Surface net solar radiation | J/m² |
| ssrd | Surface solar radiation downwards | J/m² |
| str | Surface net thermal radiation | J/m² |
| swvl1 | Soil water layer 1 (0-7cm) | m³/m³ |
| swvl2 | Soil water layer 2 (7-28cm) | m³/m³ |
| t2m | 2m temperature | K |
| tp | Total precipitation | m |
| u10 | 10m U wind component | m/s |
| v10 | 10m V wind component | m/s |

Output shape per window: `(14, 448, 1, 1)` — 14 channels, 448 daily timesteps, 1×1 spatial.

## Stratification Details

- **Köppen-Geiger**: Beck et al. (2023), 30 classes, 0.1° resolution, 1991-2020
- **Elevation bands**: <250m / 250-750m / 750-1500m / 1500-3000m / >3000m
- **Latitude bands**: [-90,-60) / [-60,-30) / [-30,0) / [0,30) / [30,60) / [60,90]
- **Allocation**: `n_b ∝ count_b^0.5` with floor=5 per occupied bin

## References

Beck, H.E. et al. (2023). High-resolution (1 km) Köppen-Geiger maps for
1901–2099 based on constrained CMIP6 projections. *Scientific Data* 10, 724.
