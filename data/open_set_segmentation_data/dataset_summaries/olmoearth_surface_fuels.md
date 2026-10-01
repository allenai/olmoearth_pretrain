# OlmoEarth surface fuels

- **Slug**: `olmoearth_surface_fuels`
- **Status**: completed
- **Task type**: classification (sparse points)
- **num_samples**: 11017
- **Source**: local rslearn eval dataset `olmoearth_evals/surface_fuels`
  (`/weka/dfive-default/rslearn-eai/datasets/olmoearth_evals/surface_fuels`), `have_locally: true`.
- **License**: internal.

## What the source is

Wildfire surface-fuel-model segmentation eval derived from **LANDFIRE FBFM40** (Scott &
Burgan 2005, 40 Fire Behavior Fuel Models). The rslearn dataset has 14,000 windows
(train 7045 / val 3461 / test 3494), each a 64x64 tile at 10 m in local UTM, with a
Sentinel-2 12-band time series and a `label_raster` layer.

**Key finding — labels are single-pixel, not dense.** Although the manifest declares
`label_type: dense_raster`, inspection of every window's `label_raster/label/geotiff.tif`
showed **exactly one valid pixel** per 64x64 tile (the other 4095 pixels are nodata=255).
So each window is really a single 10 m FBFM40 point label, not a dense raster. Per spec
§2/§2a (1x1 labels are a (location, class) pair and must not be written as per-sample
GeoTIFFs — writing 14k near-empty tiles would waste weka), this was processed as a
**sparse-point classification dataset** written to one `points.geojson` table.

## Class mapping

29 distinct FBFM40 codes appear (one per class). The source `label_raster` encodes them as
class ids 0..28 in **ascending FBFM40-code order**; this was verified by scanning all 14k
windows (each window's `options.category` FBFM40 code maps 1:1 to its single label-raster
value). The same mapping is applied here. Class id / FBFM40 code / name:

| id | code | name | | id | code | name |
|----|------|------|-|----|------|------|
| 0 | 91 | NB1 Urban/Developed | | 15 | 162 | TU2 Moderate-load humid timber-shrub |
| 1 | 93 | NB3 Agricultural | | 16 | 163 | TU3 Moderate-load humid timber-grass-shrub |
| 2 | 98 | NB8 Open Water | | 17 | 165 | TU5 Very-high-load dry timber-shrub |
| 3 | 99 | NB9 Bare Ground | | 18 | 181 | TL1 Low-load compact conifer litter |
| 4 | 101 | GR1 Short sparse dry grass | | 19 | 182 | TL2 Low-load broadleaf litter |
| 5 | 102 | GR2 Low-load dry grass | | 20 | 183 | TL3 Moderate-load conifer litter |
| 6 | 103 | GR3 Low-load coarse humid grass | | 21 | 184 | TL4 Small downed logs |
| 7 | 121 | GS1 Low-load dry grass-shrub | | 22 | 185 | TL5 High-load conifer litter |
| 8 | 122 | GS2 Moderate-load dry grass-shrub | | 23 | 186 | TL6 Moderate-load broadleaf litter |
| 9 | 141 | SH1 Low-load dry shrub | | 24 | 187 | TL7 Large downed logs |
| 10 | 142 | SH2 Moderate-load dry shrub | | 25 | 188 | TL8 Long-needle litter |
| 11 | 143 | SH3 Moderate-load humid shrub | | 26 | 189 | TL9 Very-high-load broadleaf litter |
| 12 | 144 | SH4 Low-load humid timber-shrub | | 27 | 201 | SB1 Low-load activity fuel |
| 13 | 145 | SH5 High-load dry shrub | | 28 | 202 | SB2 Moderate-load activity fuel |
| 14 | 161 | TU1 Low-load dry timber-grass-shrub | | | | |

## Geo / time handling

- **Coordinates**: from `data.csv` (`latitude`, `longitude`, `fbfm40`, `task_name`),
  verified 1:1 with the 14,000 window directories (window name == `task_name`). Points fall
  in the US Sierra Nevada / Northern California region (lon -121.6..-119.5, lat 37.5..40.4).
- **Time range**: `data.csv` has a degenerate `start==end==2024-01-01`; window
  `metadata.json` gives `2024-01-01 .. 2024-12-31`. FBFM40 describes the 2024 fuel state, a
  seasonal/annual label -> assigned a **1-year window (2024)** (`io.year_range(2024)`),
  `change_time=null`. In the Sentinel era, so no pre-2016 filtering needed.

## Sampling

Balanced to <= 1000/class via `balance_by_class(..., per_class=1000)` with the default
`total_cap=25000`, which lowers the effective per-class limit to `25000 // 29 = 862`. Five
classes were capped at 862 (FBFM40 202/122/165/102/144, originally 1991/1671/1598/1135/898);
all others kept in full. Selected total: **11,017**. All source splits used. Several classes
are sparse (163 and 143: 5 each; 201: 9; 93: 11; 189: 28) — kept per spec §5 (downstream
assembly drops too-small classes; do not drop here).

## Outputs

- `datasets/olmoearth_surface_fuels/points.geojson` — FeatureCollection, 11017 Point
  features (WGS84), `properties.label` = class id 0..28.
- `datasets/olmoearth_surface_fuels/metadata.json` — 29-class map with FBFM40 descriptions.
- `raw/olmoearth_surface_fuels/SOURCE.txt` — pointer to the local rslearn dataset.

## Verification

- `points.geojson`: 11017 features, label range 0..28 (all 29 classes present), coordinates
  within the expected US region. Feature 000000 source_id `..._39.199973_-121.093169` matches
  its geometry (-121.0932, 39.2000).
- `metadata.json`: 29 classes, `num_samples=11017`, nodata 255.
- Deterministic (seeded `balance_by_class`) and atomic writes -> re-running reproduces the
  same table.

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.olmoearth_surface_fuels
```

## Caveats

- Manifest said `dense_raster` and time_range `2016-2022`, but the actual eval data is
  single-pixel point labels for **2024** — processed as sparse points accordingly.
- FBFM40 is a derived product; labels are point samples of the LANDFIRE map, so treat as
  map-derived reference, not in-situ ground truth.
