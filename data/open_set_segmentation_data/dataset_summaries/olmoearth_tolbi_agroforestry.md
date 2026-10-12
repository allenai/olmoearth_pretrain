# OlmoEarth Tolbi agroforestry

- **Slug**: `olmoearth_tolbi_agroforestry`
- **Status**: completed
- **Task type**: classification (sparse point segmentation)
- **Num samples**: 5000 (1000 per class x 5 classes)
- **Region**: Ivory Coast, West Africa
- **License**: internal (olmoearth)

## Source and access

The Tolbi project maps tropical tree-crops / agroforestry (cash crops such as cacao,
rubber, oil palm) over Ivory Coast. The manifest `url`
`/weka/dfive-default/rslearn-eai/datasets/tolbi` (a weka mirror of
`gs://rslearn-eai/datasets/tolbi`) is **not present on disk**. The same dataset is
materialized on weka as the eval dataset
`/weka/dfive-default/olmoearth/eval_datasets/tolbi_crop`, group `20251210`, produced by
`rslp/tolbi/create_windows.py` (44,661 windows). That materialized copy was used as the
label source (`raw/olmoearth_tolbi_agroforestry/SOURCE.txt` records this).

Provenance of the labels (from `rslp/tolbi/README.md`):
- **Positives** (cacao, rubber): points sampled from the Tolbi team's manually annotated
  ground-truth cash-crop polygons, reference year **2024**.
- **Negatives** (tree, shrub, others): reference point clusters from ESA WorldCover over
  Ivory Coast, reference year **2016**.

## Why a point table (not rasterized polygons)

The manifest lists `label_type: polygons`, but the on-disk materialized labels are
**single-pixel points**: each 31x31 window's `label_raster` carries the class id only at
the center pixel (rest = 0), and the class name lives in the window `options.category`.
The original ground-truth polygons are not on weka. A single 10 m pixel per label is a
sparse-point dataset, so per spec §2a it is written as one dataset-wide
`points.geojson` (FeatureCollection, WGS84 `[lon,lat]`), not per-sample GeoTIFFs.
lon/lat is computed from each window's center pixel in its native UTM projection
(EPSG:32629/32630) and verified against the lat/lon encoded in the window name.

## Classes

Manifest classes: cacao, palmoil, rubber, tree, shrub, others (6). The materialized
dataset contains **no `palmoil` samples**, so palmoil is dropped and documented. The 5
present classes (0-indexed uint8 ids):

| id | name | source | raw count | selected |
|----|------|--------|-----------|----------|
| 0 | cacao | Tolbi ground-truth polygons | 8467 | 1000 |
| 1 | rubber | Tolbi ground-truth polygons | 6194 | 1000 |
| 2 | tree | ESA WorldCover reference | 10000 | 1000 |
| 3 | shrub | ESA WorldCover reference | 10000 | 1000 |
| 4 | others | ESA WorldCover reference | 10000 | 1000 |

nodata/ignore value = 255.

## Sampling and time range

- Balanced to **1000 per class** via `balance_by_class` (5 classes x 1000 = 5000, well
  under the 25k cap). Rare-class truncation not triggered.
- **Time range** = reference year as a 1-year window (`io.year_range`): positives -> 2024,
  negatives -> 2016. Both are in the Sentinel era (2016+), so no pre-2016 rejection.
  No change labels.
- All source train/val splits used (no split filtering).

## Verification

- `points.geojson`: 5000 features, exactly 1000 per class id 0-4; year split 2024=2000,
  2016=3000; lon in [-8.56, -2.68], lat in [4.55, 10.52] (matches the Ivory Coast bbox in
  the Tolbi README). `metadata.json` class ids cover all label values present.
- Coordinate correctness confirmed: computed lon/lat matches the lat/lon embedded in each
  window name (e.g. `20251210/5017_7.4792_-6.6793` -> (-6.6793, 7.4792)). A full
  Sentinel-2 overlay eyeball was not performed; coordinates are verified against the
  source window georeferencing and the labels are the project's own ground truth.

## Caveats

- Labels are single-pixel points (not polygon footprints); the source polygons were not
  available on weka.
- Negatives (tree/shrub/others) are WorldCover-derived and, per the Tolbi README, may
  include some mislabeled cash crops (e.g. WorldCover "tree" that is actually oil
  palm/rubber). This is a known source data-quality issue.
- `palmoil` (manifest class) has zero materialized samples and is omitted.

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.olmoearth_tolbi_agroforestry
```
Idempotent: re-running rewrites `points.geojson` / `metadata.json` deterministically
(seeded balancing).
