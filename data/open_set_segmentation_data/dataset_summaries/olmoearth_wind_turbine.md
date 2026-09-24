# OlmoEarth wind turbine

- **Registry slug:** `olmoearth_wind_turbine`
- **Status:** completed
- **Task type:** classification (object detection encoded as per-pixel classes)
- **Num samples:** 2000 (1000 turbine-positive tiles + 1000 background-only negative tiles)
- **Source:** olmoearth / Satlas — existing OlmoEarth wind-turbine detection eval
  (`have_locally: true`, not copied).
- **License:** ODbL/internal.

## Source

Local rslearn dataset (no raw copy; `raw/olmoearth_wind_turbine/SOURCE.txt` points at it):

```
/weka/dfive-default/rslearn-eai/datasets/wind_turbine/dataset_v1/20260122
```

Each `windows/{group}/{name}` is a small crop already in a **local UTM projection at
10 m/pixel** (~360–420 px square) with a Satlas seasonal-mosaic `time_range` (~90 days).
The vector `label` layer holds one `Point` feature per manually annotated turbine
(`properties.category == "turbine"`).

- **14,395 windows** in two groups: `label` (2,719) and `naip` (11,676). Both are used
  (splits are pretraining-agnostic, spec §5).
- **4,245 positive windows** (≥1 turbine) containing **38,081 turbine points** total;
  **10,150 turbine-free windows**.
- All windows are UTM @ 10 m (58 distinct UTM zones). All label years are **2017–2022**
  (post-2016; nothing to filter under the pre-2016 rule).

### Coordinate-system gotcha (important)

The two groups store the label GeoJSON coordinates **differently**:

- `label` group: coordinates are in the **window's projection pixel** units
  (FeatureCollection `properties.crs` = window UTM CRS, `x_resolution=10`), matching the
  window `bounds`.
- `naip` group: coordinates are **WGS84 lon/lat** (top-level GeoJSON `crs` = EPSG:4326).

The script detects the coordinate system per file (`_geojson_is_wgs84`: top-level `crs`
member, with a lon/lat magnitude fallback) and reprojects lon/lat into the window's
projection pixel coords via `STGeometry(WGS84).to_projection(window_proj)`. A first pass
that assumed pixel coords for all groups silently dropped ~74% of positives (the naip
lon/lat fell far outside every tile); this was caught in verification and fixed.

## Encoding (spec §4 — bboxes → detection)

The manifest `label_type` is `bboxes`, but the on-disk annotations are turbine-centroid
**points**, so we use the tunable detection encoding:

- One **64×64** context tile per turbine, **written in the window's own UTM projection** at
  10 m (exact georeferencing; no reprojection of the raster grid).
- The tile is centered on the turbine but **clamped to lie fully inside the source window**,
  so every turbine in the tile is known (turbines are only annotated within a window) and
  background pixels are true negatives (no unlabeled turbines leak in from outside).
- The turbine is a **1×1 positive** (class `1 = turbine`), ringed by a **10 px nodata (255)
  buffer** (centroids are not pixel-exact), all other pixels **background** (`0`). Every
  other annotated turbine of the same window falling inside the tile is also marked positive
  (dense wind farms yield multi-turbine tiles: 1–10 turbine pixels/tile, mean ≈1.9).
- **Negatives:** background-only 64×64 tiles sampled from turbine-free windows so the
  background class has spatially-meaningful negatives (spec §5 detection exception).

Parameters: `tile_size=64`, `positive_size=1`, `buffer_size=10`.

## Classes

| id | name | description |
|----|------|-------------|
| 0 | background | Non-turbine ground / sea surface within the tile. |
| 1 | turbine | Wind turbine (manually annotated centroid; onshore & offshore). |

`nodata_value = 255` (buffer ring / ignore).

## Sampling

Single object class → up to **1000 positive turbine tiles** + **1000 background negative
tiles** = 2000 samples (matching the `olmoearth_sentinel_2_vessels` detection precedent;
well under the 25k cap). One positive-tile candidate per annotated turbine (38,081
candidates); shuffled (seed 42) and truncated to 1000. Negatives: turbine-free windows
shuffled and truncated to 1000.

## Time range

Each sample uses its **source window's own Satlas seasonal-mosaic `time_range`** (~90 days,
≤ 1 year, spec §5). Turbines are persistent structures, so a seasonal observation window is
valid; `change_time = null`.

## Verification (spec §9)

- 2000 `.tif` + 2000 matching `.json`, all paired.
- All tiles: single band, **uint8**, **64×64**, resolution (10, −10), **UTM** CRS (50
  distinct zones in the selection).
- Pixel values ⊆ {0, 1, 255}; class map covers {0, 1}; nodata 255. All 1000 positives
  contain a turbine (1) + buffer (255); all 2000 contain background (0).
- All `time_range`s valid (0 < days ≤ 366); no change labels.
- **Spatial overlay:** for one positive from each group (`naip` and `label`), the source
  Sentinel-2 RGB was loaded at the tile CRS/bounds and the turbine pixels overlaid — the
  markers land on the bright turbine pads at road/ridgeline ends, confirming alignment
  (including the naip WGS84→UTM reprojection).

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.olmoearth_wind_turbine
```

Idempotent (skips already-written `{sample_id}.tif`). If the encoding/selection changes,
clear `datasets/olmoearth_wind_turbine/locations/` first (sample ids are reused).

## Caveats

- Individual turbines are ~point-scale at 10 m; the S2/S1 signal is the bright
  turbine pad + shadow, resolvable but small. The 10 px ignore buffer accommodates centroid
  imprecision.
- Only 1000 of 38,081 turbines are sampled (per-class cap); the dataset could support more
  if the downstream cap were raised.
