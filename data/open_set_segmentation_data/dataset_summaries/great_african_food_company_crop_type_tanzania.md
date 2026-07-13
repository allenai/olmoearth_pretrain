# Great African Food Company Crop Type Tanzania

- **slug**: `great_african_food_company_crop_type_tanzania`
- **status**: completed — **classification** (per-pixel crop type)
- **num_samples**: 392 label patches (6 classes)
- **source**: Source Cooperative `radiantearth/african-crops-tanzania-01`
  (Radiant Earth Foundation & Great African Food Company, 2018; DOI 10.34911/rdnt.5vx40r).
  License **CC-BY-4.0**.
- **region / time**: Tanzania (Arusha, Simiyu, ...); 2018 growing season (planting
  Jan–May 2018, mostly March; harvest Jul–Dec 2018).

## Source & access
Public, no credentials. Originally distributed via Radiant MLHub (now retired); an open
mirror lives on Source Cooperative and was read via the unsigned S3 proxy
`https://data.source.coop` (bucket `radiantearth`, prefix `african-crops-tanzania-01/`).
Ground reference collected in-field with the **Farmforce** app: a surveyor recorded a
point inside each field plus the field boundary and properties (Village, Region, Plot Area,
Planting Date, estimated Harvest Date, Crop). We pulled only the **vector labels** + docs
(`Tanzania_Documentation.pdf`, `Tanzania_properties.csv`) into
`raw/great_african_food_company_crop_type_tanzania/` (~436 KB). The bundled Sentinel-2
`imagery/` (~46k COGs) was **not** downloaded — pretraining supplies its own imagery.

The mirror ships the labels **twice** (identical 392 fields): 24 top-level STAC label items
`ref_african_crops_tanzania_01_tile_XXX.geojson` (WGS84) and per-imagery-chip
`label/{NN}/{NN}_label.geojson` (UTM). We use the **top-level WGS84 STAC tiles** (verified
byte-for-crop-count identical to the `label/` set) to avoid double-counting. All 392
geometries are valid `Polygon` field boundaries. Sample centroids confirmed to fall in
Tanzania (lon ≈ 35.6–35.9°E, lat ≈ −3.2 to −3.9°N).

## Classes
The manifest's guessed `classes` (`[wheat, maize, sorghum, vegetables]`) **do not match the
data**; the 6 crop types actually present in the `Crop` property are used instead. Class ids
0–5 assigned by descending field count. All classes kept (well under the 254 uint8 cap),
including sparse ones per spec §5 (downstream assembly handles rare-class filtering).

| id | name | fields | samples written |
|----|------|--------|-----------------|
| 0 | Bush Bean | 156 | 156 |
| 1 | Dry Bean | 137 | 137 |
| 2 | Sunflower | 51 | 51 |
| 3 | Safflower | 24 | 24 |
| 4 | White Sorghum | 15 | 15 |
| 5 | Yellow Maize | 9 | 9 |

## Label encoding
Polygon rasterization (EuroCrops/LEM/AgriFieldNet-style), one label patch per field. Each
field polygon is reprojected into its local UTM zone (EPSG:32736 here) at 10 m and
rasterized (`all_touched=True`) into a `≤64×64` tile centered on the polygon: the crop class
id is burned inside the polygon and **255 (nodata/ignore)** fills everything outside. Ground
truth exists only inside surveyed fields, so unlabeled land is ignore, not a background
class (no synthetic negatives; spec §5 positive-only handling). Fields are small smallholder
plots (max footprint ~39×33 px; typical tiles 3–8 px on a side; hard-capped 64).

- **Time range**: crop type is a seasonal label and each field carries a real **Planting
  Date**, so a per-field 1-year window `[Planting Date, Planting Date + 360 days]` is used —
  it spans the field's growing/harvest cycle (all within the 2018 season, consistent with
  the manifest `time_range` [2018, 2019]). `change_time = null`.
- **Sampling**: class-balanced, up to 1000 fields/class, 25k cap (`balance_by_class`). With
  only 392 fields, **all** are kept (no truncation).

## Judgment calls / caveats
- Manifest `classes` were a guess that didn't match the source; the actual `Crop` values
  (6 crops) are authoritative and were used verbatim.
- Two redundant label representations exist on the mirror; only the top-level WGS84 STAC
  tiles were processed to avoid duplicating the same 392 fields.
- Small dataset (392 fields) — several classes are sparse (Yellow Maize 9, White Sorghum
  15); kept per spec §5.
- Per-field planting-date-anchored windows chosen over a flat calendar-year window to
  better align imagery with each field's phenology; all fields fall in the 2018 season so
  this stays within the manifest's [2018, 2019] range.
- Georeferencing validated by reprojecting tile bounds to WGS84 and confirming centroids in
  Tanzania.

## Reproduce
```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.great_african_food_company_crop_type_tanzania
```
Idempotent (skips already-written `locations/{id}.tif`). Public unsigned S3 read from
`https://data.source.coop/radiantearth/african-crops-tanzania-01/` (no credentials).
