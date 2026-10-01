# OlmoEarth AWF land use/land cover

- **Slug:** `olmoearth_awf_land_use_land_cover`
- **Status:** completed
- **Task type:** classification (sparse points, spec §2a → `points.geojson`)
- **Num samples:** 1459
- **Region:** African Wildlife Foundation landscape, Kenya / Amboseli (East Africa; lon ~36.0–38.0, lat ~-1.66 to -3.38)
- **Source:** local rslearn dataset `have_locally: true` at
  `/weka/dfive-default/rslearn-eai/datasets/crop/awf_2023` (existing OlmoEarth eval, license internal).

## Source structure and access

Local rslearn dataset (no download; `raw/{slug}/SOURCE.txt` points at it). Three window
groups:

- `20250822` (1459 windows) — **the only group with ground-truth labels.** Each window is a
  32×32 (320 m) tile in EPSG:32737. Its `label` vector layer is a single window-covering
  polygon with property `lulc` (the class string), and its `label_raster` layer is
  background (0) everywhere except a single pixel at the center (16,16) carrying the
  reference class id (1–9). Window `metadata.json` `options` carry `lulc`, `latitude`,
  `longitude`, and a split. These are **manual reference points** — one labeled pixel per
  window.
- `amboseli` (7452) and `kenya` (437) — **unlabeled** prediction/eval tiles (only
  sentinel2/sentinel1/prediction layers, no `label`/`label_raster`). Excluded.

## Labels and class mapping

Treated as sparse-point classification: one `Point` feature per reference point at its
`(longitude, latitude)` (WGS84), `label` = class id. Verified the reported lon/lat matches
the window's center pixel to sub-pixel precision (~10 m).

Class ids assigned in manifest order (0-based). Source `label_raster` uses 1-based ids
which are not reused; `lulc` string is the source of truth.

| id | class | source count |
|----|-------|-------|
| 0 | Agriculture/Settlement | 288 |
| 1 | Grassland/barren | 320 |
| 2 | Herbaceous wetland | 49 |
| 3 | Lava forest | 18 |
| 4 | Montane forest | 59 |
| 5 | Open water | 55 |
| 6 | Shrubland/Savanna | 412 |
| 7 | Urban/dense development | 90 |
| 8 | Woodland forest (>40% canopy) | 168 |

All 9 classes kept; no class exceeds the 1000/class cap so all 1459 points are used
(`balance_by_class`, total cap 25000). The manifest's short name "Woodland forest" and the
source "Woodland forest (>40% canopy)" both map to id 8. Rare classes (Lava forest 18,
Herbaceous wetland 49, Open water 55) retained per §5 (downstream assembly filters
too-small classes).

## Time range

Seasonal/annual land cover from 2023 imagery → 1-year window `2023-01-01 .. 2024-01-01`
per point (§5). No change labels.

## Outputs

- `datasets/olmoearth_awf_land_use_land_cover/points.geojson` — FeatureCollection, 1459
  Point features, `task_type: classification`.
- `datasets/olmoearth_awf_land_use_land_cover/metadata.json` — class map + counts.
- `raw/olmoearth_awf_land_use_land_cover/SOURCE.txt`.
- `datasets/olmoearth_awf_land_use_land_cover/registry_entry.json` — status.

## Verification

- `points.geojson`: 1459 features, labels 0–8 present, coords within the Kenya/Amboseli
  bbox.
- Spatial sanity: point lon/lat reprojects to the labeled window's center pixel to
  sub-pixel precision, confirming label–geometry co-registration (imagery and labels share
  the source window in the existing eval).

## Caveats

- Only 1459 labeled points exist (the `20250822` group); the larger `amboseli`/`kenya`
  groups have no ground truth and are excluded.
- Labels are single reference pixels; the 32×32 source window was context only, so labels
  are stored as points (not dense tiles).

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.olmoearth_awf_land_use_land_cover
```
Idempotent (rewrites the single `points.geojson`/`metadata.json`).
