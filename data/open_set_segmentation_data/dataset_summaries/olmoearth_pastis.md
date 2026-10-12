# OlmoEarth PASTIS (`olmoearth_pastis`)

**Status:** completed · **Task:** classification (crop type, dense_raster) · **Samples:** 6001

## Source
PASTIS / PASTIS-R (Garnot & Landrieu, *Panoptic Segmentation of Satellite Image Time
Series*, ICCV 2021). A crop-type **semantic segmentation** benchmark over four Sentinel-2
tiles in France (UTM zones 30/31/32): 2433 patches of 128×128 px at 10 m. Labels are the
French **RPG** (Registre Parcellaire Graphique) farmer declarations for the **2019**
campaign.

- License: open for research.
- `have_locally: true`. Staged copy used directly:
  `/weka/dfive-default/rslearn-eai/artifacts/PASTIS-R/`
  - `ANNOTATIONS/TARGET_{id}.npy` — shape `(3,128,128)`; **channel 0** = semantic class.
  - `metadata.geojson` — per-patch footprint in **EPSG:2154 (Lambert-93)**, `TILE`
    (S2 tile id), `Fold`, and S2/S1 acquisition dates.
- Zenodo: https://zenodo.org/records/5735646 (PASTIS-R) / 5012942 (PASTIS).
- Only the **labels** are used; pretraining supplies its own S2/S1/Landsat imagery, so the
  `DATA_S2`/`DATA_S1*` arrays are ignored.

## Access / reproduction
Labels are local; no download. `raw/olmoearth_pastis/SOURCE.txt` records the source paths.
Reproduce with:
```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.olmoearth_pastis
```
Idempotent — skips already-written `locations/{id}.tif`.

## Georeferencing (important)
The pre-existing local **rslearn** PASTIS dataset
(`/weka/dfive-default/rslearn-eai/datasets/pastis/`) is **not usable for geolocation**: its
`convert.py` assigns a *dummy* `EPSG:3857` origin `(0,0)` ("difficult to get the actual
correct one"), which would place every patch off the coast of West Africa. We therefore
recover true geolocation from `metadata.geojson`:

- Each patch footprint, transformed from EPSG:2154 into its own Sentinel-2 UTM zone
  (derived from `TILE`, e.g. `t30uxv` → EPSG:32630), is an **exact 1280×1280 m
  axis-aligned square**, confirmed for all four tiles. So the 128×128 native grid maps
  **1:1** onto the UTM 10 m grid — **no resampling**. Origin snapped to the nearest 10 m
  pixel (sub-pixel <0.3 px offset from the 2154→UTM transform; negligible).
- Verified: 1201/1201 sampled tile centroids fall inside France; random samples round-trip
  onto their original EPSG:2154 patch footprints; output label arrays match the source
  TARGET quadrants exactly.

## Class mapping
Native PASTIS semantic ids are kept as-is (`uint8`), void → nodata:

| id | class | | id | class |
|----|-------|-|----|-------|
| 0 | background (non-crop, real negative) | | 10 | winter triticale |
| 1 | meadow | | 11 | winter durum wheat |
| 2 | soft winter wheat | | 12 | fruits/vegetables/flowers |
| 3 | corn | | 13 | potatoes |
| 4 | winter barley | | 14 | leguminous fodder |
| 5 | winter rapeseed | | 15 | soybeans |
| 6 | spring barley | | 16 | orchard |
| 7 | sunflower | | 17 | mixed cereal |
| 8 | grapevine | | 18 | sorghum |
| 9 | beet | | 19 | **void → 255 (nodata/ignore, dropped)** |

19 usable classes (0–18) + `nodata=255`; well under the 254-class uint8 cap. Class 0
(background) is a genuine observed non-crop class (kept, not fabricated), consistent with
the PASTIS semantic benchmark convention (train on background, mask out void).

## Processing
- **label_type = dense_raster.** Each 128×128 @ 10 m patch is split into four **64×64**
  UTM 10 m quadrant tiles (`≤64×64` cap). All-nodata quadrants are dropped.
- Single-band `uint8` GeoTIFFs written with exact rslearn georeferencing (local UTM,
  10 m, north-up), sidecar JSON per tile.
- **Sampling:** tiles-per-class balanced (`select_tiles_per_class`, `per_class=1000`,
  25k cap). A tile counts toward every class present; rare crops prioritized. 9730
  candidate tiles → **6001** selected.
- **Time range:** PASTIS labels are the 2019 RPG campaign; source imagery spans
  Sep 2018–Nov 2019 (>1 yr). Assigned a fixed **360-day 2019 growing-season window**
  `[2019-01-01, 2019-12-27)` (≤1 year, post-2016). `change_time = null` — this is
  crop-state classification, not a dated change event.

## Sample counts per class (tiles containing the class)
```
0 background      6001    10 winter triticale  1000
1 meadow          4669    11 winter durum wht   885
2 soft winter wht 2978    12 fruits/veg/flowers 1001
3 corn            3083    13 potatoes           501
4 winter barley   1524    14 leguminous fodder 1362
5 winter rapeseed 1031    15 soybeans           925
6 spring barley    830    16 orchard            989
7 sunflower        895    17 mixed cereal        784
8 grapevine       1000    18 sorghum            572
9 beet             719
```
(Background/meadow exceed 1000 because they co-occur in nearly every selected tile — a
tile counts toward all classes it contains; this is expected for tiles-per-class
balancing per spec §5. Rarer classes take all available tiles.)

## Verification (spec §9)
- 6001 `.tif` each with a matching `.json`; 0 corrupt/empty; all `time_range` ≤ 360 days;
  `change_time` null.
- Tifs: single-band `uint8`, CRS EPSG:32630/32631/32632, 10 m, max dim 64, pixel values
  ∈ {0..18, 255}. `metadata.json` class ids cover all values present.
- Centroids: 1201/1201 sampled in France; label arrays match source TARGET quadrants;
  geo round-trips onto EPSG:2154 footprints.

## Caveats
- The staged rslearn PASTIS dataset's geolocation is dummy; do not use it — true geo comes
  from `metadata.geojson` (handled here).
- ~0.3-pixel origin snap from the EPSG:2154→UTM transform; immaterial for co-location.
- Some classes are sparse (potatoes 501, sorghum 572, beet 719) — kept per spec §5;
  downstream assembly filters too-small classes.
