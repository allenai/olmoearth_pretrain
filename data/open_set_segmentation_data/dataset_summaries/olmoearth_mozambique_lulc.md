# OlmoEarth Mozambique LULC (+ crop type)

- **Slug:** `olmoearth_mozambique_lulc`
- **Status:** completed
- **Task type:** classification (sparse points → `points.geojson`, spec §2a)
- **Num samples:** 11,158 points (≤1000/class, 25k cap)
- **Family / region / license:** land_cover / Mozambique / internal

## Source

Internal OlmoEarth Mozambique LULC + crop-type project (manifest `url:
olmoearth_projects/projects/mozambique_lulc`, `have_locally: true`). The authoritative
labels are **manual field-survey reference points** collected across three provinces —
Gaza, Manica, Zambezia — distributed as GeoPackages at
`/weka/dfive-default/yawenz/datasets/mozambique/train_test_samples`:

- **LULC:** `{gaza,manica,zambezia}_{train,test}.gpkg` — column `class` (int 0–6). 7,721 points.
- **Crop type:** `{training,test}_gaza_zambezia_manica.gpkg` — column `crop1` (crop name string). 10,385 points.

Total 18,106 source points. A paired staged rslearn dataset exists at
`/weka/dfive-default/rslearn-eai/datasets/crop/mozambique_lulc/20251202` (built by the
project's `create_windows_for_lulc.py` with `--window_size 32`, then `create_label_raster.py`
which draws only the **centre 1×1 pixel** with the class — confirming the label is a single
10 m point). We read lon/lat directly from the source GPKGs instead of the staged windows.

## Format decision: points, not polygons

The manifest lists `label_type: polygons`, but **every source feature is a `Point`** (verified
with geopandas: `geom_type` is 100% Point in all 8 GPKGs). Each point is a single land-cover /
crop-type observation, so per spec §2 this is a **sparse-point** dataset: written as one
dataset-wide `points.geojson` (spec §2a), not per-sample GeoTIFFs.

## Georeferencing (PASTIS-style dummy-bounds check)

The sibling `olmoearth_pastis` found its local rslearn copy used a fake EPSG:3857 (0,0)
origin. Here we **avoid that risk entirely** by not reading the staged rslearn window bounds:
lon/lat come straight from the source GPKGs — LULC in **EPSG:4326**, crop type in
**EPSG:3036 (Moznet / UTM zone 36S)** reprojected to WGS84 with geopandas. All 18,106 points
land in Mozambique: **lon 31.32–39.03, lat −25.35 to −15.04** (Gaza in the south through
Manica/Zambezia in the centre-north). Geolocation is real; no recovery needed. A runtime
assertion enforces the Mozambique bbox.

## Unified class scheme (spec §5 multi-target → one dataset)

LULC and crop-type are combined into ONE dataset with a unified 14-class `uint8` map. LULC
keeps its native ids 0–6; the 7 crop types are appended as ids 7–13 (they refine the generic
`Cropland` LULC class with the surveyed crop). Well under the 254-class cap. `nodata=255`
(unused — every point carries a real class). Class names/definitions come from the project's
`create_windows_for_lulc.py` (`CLASS_MAP`) and `create_label_raster.py`.

| id | name | source | selected count |
|----|------|--------|----------------|
| 0 | Water | LULC | 874 |
| 1 | Bare Ground | LULC | 957 |
| 2 | Rangeland | LULC | 816 |
| 3 | Flooded Vegetation | LULC | 794 |
| 4 | Trees | LULC | 764 |
| 5 | Cropland | LULC | 1000 (of 2163) |
| 6 | Buildings | LULC | 1000 (of 1353) |
| 7 | corn | crop type | 1000 (of 4817) |
| 8 | cassava | crop type | 1000 (of 1019) |
| 9 | rice | crop type | 1000 (of 2023) |
| 10 | sesame | crop type | 767 |
| 11 | beans | crop type | 1000 (of 1573) |
| 12 | millet | crop type | 93 |
| 13 | sorghum | crop type | 93 |

`millet` and `sorghum` (93 each) are rare but kept in full (spec §5 — downstream assembly
drops classes below its minimum).

## Time range & change handling

Static/seasonal land-cover + crop labels for the surveyed growing season. All points use the
project's per-province window range **2024-10-23 → 2025-06-20 UTC** (~240 days, ≤1 year,
post-2016 Sentinel era). `change_time = null` (state classification, not a dated change event).

## Sampling

All source train/val/test splits used (spec §5: all windows are fair game). Balanced with
`sampling.balance_by_class(per_class=1000, total_cap=25000)`; 14 classes → cap stays 1000/class.
Selected 11,158 of 18,106.

## Verification (spec §9)

- `points.geojson`: `FeatureCollection`, `count=11158`, `task_type=classification`, 11,158 Point features.
- Labels span ids 0–13 (all 14 classes present); `nodata_value=255` in `metadata.json`.
- Per-feature `time_range` span = 240 days (≤ 1 year); `change_time=null`.
- Coordinates all within Mozambique (lon 31.32–39.03, lat −25.35 to −15.04). Geolocation is
  exact-by-construction (read from source lon/lat), so no imagery-overlay misalignment is expected.
- Idempotent: re-running rewrites `points.geojson` + `metadata.json` deterministically
  (seeded balancing, stable ordering).

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.olmoearth_mozambique_lulc
```

## Caveats

- Manifest `label_type` (`polygons`) and class count (`8 LULC classes`) are slightly off: the
  source features are Points and there are 7 LULC classes + 7 crop types (14 total). Followed
  the actual data.
- Crop-type and LULC points are largely at different survey locations; combining them adds
  distinct crop classes rather than overwriting the generic `Cropland` class.
