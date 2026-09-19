# OlmoEarth Canada crops (fine)

- **Slug:** `olmoearth_canada_crops_fine`
- **Status:** completed
- **Task type:** classification (sparse point segmentation)
- **Num samples:** 9,951
- **Source:** local rslearn eval dataset
  `/weka/dfive-default/rslearn-eai/datasets/olmoearth_evals/canada_crops_fine`
  (`source: olmoearth`, `license: internal`, `have_locally: true`).

## What the source is

Fine-grained crop-type point labels over Canadian farmland, derived from the AAFC Annual
Crop Inventory (ACI). The rslearn dataset has 14,566 windows across two groups
(`train`=5,565, `test`=9,001). Each window is a single labeled 10 m point: the class name,
lon/lat, and a ~1-year `time_range` live in the window `metadata.json` `options` (mirrored
in the `label` vector layer as a single `Point` feature, and in `label_raster` as one
class-0 pixel surrounded by 255 nodata in a 32x32 tile). This is therefore a pure
sparse-point dataset.

## Access / processing

`have_locally: true`, so no download — `raw/olmoearth_canada_crops_fine/SOURCE.txt` points
at the source path. Window `metadata.json` files are scanned in parallel
(`multiprocessing.Pool(64)`) to build flat `(lon, lat, label, year, source_id)` records.

Per spec §2a/§4, sparse points are written to a single dataset-wide GeoJSON point table
`datasets/olmoearth_canada_crops_fine/points.geojson` (no per-point GeoTIFFs). Each feature
carries `label` (class id), a 1-year `time_range`, and `source_id` (`group/window`).

## Classes

24 fine classes, all observed in the data (matches manifest "24 fine classes"; a few names
are more specific than the manifest short list, e.g. `Blueberry (Undiff)`, `Barley
(Undiff)`, `Mixedwood`). Class ids assigned in **descending observed frequency** (0-23).
Per-class descriptions (AAFC ACI crop/land-cover definitions) are in `metadata.json`.

Selected counts (`balance_by_class`, <=1000/class, 25k total cap not binding at 24x1000):

| id | class | selected |
|----|-------|----------|
| 0 | Mixed Forage | 1000 |
| 1 | Soybeans | 1000 |
| 2 | Corn | 1000 |
| 3 | Pasture | 1000 |
| 4 | Winter Wheat | 650 |
| 5 | Mixedwood | 603 |
| 6 | Urban | 485 |
| 7 | Alfalfa | 429 |
| 8 | Unimproved Pasture | 417 |
| 9 | Shrubland | 416 |
| 10 | Wetland | 371 |
| 11 | Abandoned (Overgrown) | 280 |
| 12 | Abandoned (Shrubs) | 265 |
| 13 | Coniferous | 255 |
| 14 | Potatoes | 242 |
| 15 | Barren | 235 |
| 16 | Oats | 205 |
| 17 | Blueberry (Undiff) | 187 |
| 18 | Barley (Undiff) | 184 |
| 19 | Water | 178 |
| 20 | Spring Wheat | 165 |
| 21 | Pasture/Forage | 156 |
| 22 | Canola/Rapeseed | 114 |
| 23 | Native Grassland | 114 |

Only 4 classes exceed 1,000 (Mixed Forage, Soybeans, Corn, Pasture, all capped at 1000);
the remaining 20 keep all their samples. Several classes are sparse (Canola/Rapeseed,
Native Grassland at 114) — kept per spec §5 (downstream assembly drops too-small classes).

## Time range

Labels span years 2016-2021 (all post-2016, no pre-Sentinel filtering needed). Annual crop
labels, so each point gets a 1-year window anchored on the ACI labeled year via
`io.year_range(year)` (Jan 1 -> Jan 1). No change labels.

## Verification

- `points.geojson`: `task_type=classification`, `count=9951`, 9,951 `Point` features,
  WGS84 coords. 24 distinct label ids (0-23), all within the class map. All time ranges
  <= 1 year.
- `metadata.json`: 24 classes with descriptions, `num_samples=9951`, `class_counts`
  present, `nodata_value=255`.
- Spatial/temporal sanity: feature 000000 (`source_id train/sample_3799`) at
  (-64.916, 47.751) in New Brunswick, Canada, label id 11 — cross-checked against the
  source window `options.label = "Abandoned (Overgrown)"` and lon/lat: exact match.
- Idempotent: re-running rescans and atomically rewrites `points.geojson`/`metadata.json`.

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.olmoearth_canada_crops_fine
```

## Caveats

- Derived-product labels (AAFC ACI), not in-situ ground truth; each point is one ACI 10 m
  pixel. A coarse 9-class variant of this dataset also exists (separate slug).
- Sparse single-pixel labels; pretraining assembly supplies negatives from other datasets.
