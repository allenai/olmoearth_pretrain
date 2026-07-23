# OlmoEarth Africa crop mask (`olmoearth_africa_crop_mask`)

- **Status:** completed
- **Task type:** classification (sparse point segmentation)
- **num_samples:** 1318

## Source

Local rslearn eval dataset at
`/weka/dfive-default/rslearn-eai/datasets/olmoearth_evals/africa_crop_mask`
(`have_locally: true`, so nothing downloaded; `raw/.../SOURCE.txt` points at it).
Binary crop / not-crop reference mask for agricultural land in Africa, manually /
photo-interpreted from Sentinel-2. 2556 windows across two source splits (train: 400,
test: 2156).

Each window carries exactly one labeled point. The class name, lon/lat and a ~1-year
`time_range` live in the window `metadata.json` `options` block; they are mirrored in a
single-point `label` vector layer and a 32x32 `label_raster` (center pixel = class id,
rest = 255 nodata). Because the label is a single 10 m point, this is a pure sparse-point
dataset and is written as one dataset-wide `points.geojson` (spec §2a), not per-point
GeoTIFFs.

## Class mapping

Manifest ordering, confirmed against the source `label_raster` pixel values:

| id | name | source count | selected count |
|----|------|--------------|----------------|
| 0 | not_crop | 2238 | 1000 |
| 1 | crop | 318 | 318 |

`not_crop` was capped at the 1000-per-class limit (`balance_by_class`, random subset);
`crop` (318) kept in full. Total selected = 1318.

## Time range

Labels carry native 1-year windows: 2019 (2405 windows) or 2020 (151 windows), all within
the Sentinel era. Assigned as-is via `io.year_range(labeled_year)`. Static/annual crop
label, no change labels.

## Geography

Points span Africa: lon -12.0 to 41.2, lat -17.6 to 24.9.

## Outputs (weka)

`datasets/olmoearth_africa_crop_mask/`: `points.geojson` (1318 features),
`metadata.json`, `registry_entry.json`.

## Caveats

- Binary mask; `crop` is the minority class (318) but kept fully per §5 (rare-class
  retention). Downstream assembly supplies negatives and filters too-small classes.
- Both source splits (train+test) used as pretraining labels, per §5.

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.olmoearth_africa_crop_mask
```
Idempotent: rewrites `points.geojson`/`metadata.json` on each run.
