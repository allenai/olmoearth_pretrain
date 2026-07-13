# DENETHOR — crop-type field parcels (Brandenburg, Germany)

- **Slug:** `denethor`
- **Status:** completed
- **Task type:** classification (per-pixel crop type)
- **Samples written:** 3901 (`locations/{id}.tif` + `.json`)
- **Label type:** polygons → rasterized parcels

## Source

DENETHOR — "The DynamicEarthNET dataset for Harmonized, inter-Operable, analysis-Ready,
daily crop monitoring from space" (Kondmann et al., NeurIPS 2021 Datasets & Benchmarks).
Repo: https://github.com/lukaskondmann/DENETHOR

Crop-type field parcels for two spatially-separated 24 km × 24 km tiles in Brandenburg,
Germany, taken from **different years** to test out-of-year generalization: a train tile
(2018) and a test tile (2019). Field boundaries + crop ids are German-state (Brandenburg)
cadastral / CAP farmer-declaration data (GeoBasis-DE/LGB), harmonized from the raw 1–999
German crop code system into **9 high-level crop classes**.

Only the *label vector files* are needed for open-set segmentation (pretraining supplies
its own imagery), so the Planet Fusion / Sentinel-1 / Sentinel-2 time series were **not**
downloaded — only the two crop-parcel GeoJSONs (+ the documentation PDF).

### Access method (no credentials required)

The original Radiant MLHub collection `dlr_fusion_competition_germany` was retired; the
data now lives, unauthenticated, on **Source Cooperative** under the ESA "Fusion
Competition" project. Files were fetched over plain HTTPS (a browser `User-Agent` header
is required — the Cloudflare front end 403s the default urllib agent):

- `https://data.source.coop/esa/fusion-competition/br-18E-242N-crop-labels-train-2018.geojson` (train tile, 2018; 2534 parcels)
- `https://data.source.coop/esa/fusion-competition/br-17E-243N-crop-labels-test-2019.geojson` (test tile, 2019; 2064 parcels)
- `https://data.source.coop/esa/fusion-competition/Crops_GT_Brandenburg_Doc.pdf` (class documentation)

Raw copies + `SOURCE.txt` are under `raw/denethor/` on weka.

### License

DL-DE/BY-2.0 (Datenlizenz Deutschland – Namensnennung 2.0), © GeoBasis-DE/LGB (2018/19),
original data altered. Open for commercial and non-commercial use with attribution.

## Data format

Each GeoJSON feature is a `MultiPolygon` in **EPSG:25833** (ETRS89 / UTM 33N) with
properties `fid`, `crop_id` (1–9), `crop_name`, `SHAPE_AREA`, `SHAPE_LEN`. Parcel areas
span ~120 m² to ~1.9 M m².

## Class mapping

Source `crop_id` (1–9) → output class id (`crop_id − 1`, ids 0–8). All 9 classes retained
(none exceed the 254-class uint8 cap). No true background class — unlabeled land inside a
tile is nodata/ignore (255), matching the eurocrops recipe.

| id | name | description |
|----|------|-------------|
| 0 | Wheat | Wheat fields (CAP-declared) |
| 1 | Rye | Rye fields |
| 2 | Barley | Barley fields |
| 3 | Oats | Oats fields |
| 4 | Corn | Corn / maize fields |
| 5 | Oil Seeds | Oil-seed crops (rapeseed/canola, sunflower) |
| 6 | Root Crops | Root crops (sugar beet, potato); rare, retained to reflect real imbalance |
| 7 | Meadows | Meadows / permanent grassland |
| 8 | Forage Crops | Forage crops (legumes, other fodder) |

### Sample counts per class (after balancing)

```
Wheat 516, Rye 538, Barley 319, Oats 85, Corn 405, Oil Seeds 310,
Root Crops 38, Meadows 1000, Forage Crops 690        (total 3901)
```

Candidate parcels total ~4.6k (max 954 in one class before balancing). `balance_by_class`
(per_class=1000, total_cap=25000) caps only Meadows (1696 candidates → 1000); everything
else is kept in full. No 254-class or 25k truncation occurs. Root Crops (38) is genuinely
sparse in this region — kept per spec §5 (downstream assembly drops too-small classes).

## GeoTIFF / processing details

- Each parcel reprojected to WGS84 for its centroid → local UTM projection
  (**EPSG:32633** for this region) at **10 m/pixel**, then rasterized (`all_touched=True`)
  into a tile sized to the parcel footprint, **centered on the parcel, capped at 64×64**.
  Class id burned inside the polygon, **255 (nodata/ignore)** outside.
- Parcels larger than 640 m on a side are centered and cropped to a 64×64 window (a solid
  patch of that crop class), per spec §4 (sampled sub-window for large coverage).
- **dtype uint8**, nodata **255**. Single band, north-up UTM.
- 1 tiny sliver parcel rasterized empty and was skipped (3902 selected → 3901 written).

## Time range

Seasonal/annual crop labels → **1-year window** anchored on each tile's labeled year:
train tile → `[2018-01-01, 2019-01-01)`, test tile → `[2019-01-01, 2020-01-01)`. Both are
post-2016 (Sentinel era). Static seasonal labels → `change_time = null`.

## Verification (spec §9)

- Opened output tifs: all single-band **uint8**, CRS **EPSG:32633**, res 10 m, size ≤64×64.
- Pixel values across all 3901 tifs = {0..8, 255} — exactly the 9 class ids + nodata; the
  `metadata.json` class map covers every value present.
- Every `.tif` has a matching `.json` (3901/3901) with a 1-year `time_range` and
  `change_time = null`.
- Coordinate sanity: sampled tile centers reproject to lon ≈ 13.6–14.2°E, lat ≈ 52.4–52.8°N
  — squarely within Brandenburg, confirming georeferencing. (A rendered S2 overlay was not
  produced; labels are authoritative CAP cadastral polygons with exact georeferencing and
  the pipeline mirrors the validated eurocrops recipe.)
- Re-running the script is idempotent (second run: 3901 skip, 1 empty).

## Reproduce

```bash
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.denethor
```

Outputs: `datasets/denethor/{metadata.json, locations/{id}.tif,.json}` on weka; status in
`datasets/denethor/registry_entry.json` (completed, classification, 3901 samples).
