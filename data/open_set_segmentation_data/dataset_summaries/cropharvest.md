# CropHarvest

- **Slug:** `cropharvest`
- **Task:** classification (sparse point segmentation)
- **Status:** completed — 9,974 samples, 11 classes
- **Source:** CropHarvest (Tseng et al., NeurIPS 2021 Datasets & Benchmarks).
  Zenodo record 7257688, file `labels.geojson`. Also mirrored at
  https://github.com/nasaharvest/cropharvest
- **License:** CC-BY-SA-4.0
- **Region:** Global (strong Africa/Asia coverage), years 2016–2022.

## Source

`labels.geojson` (70 MB) holds 95,186 harmonized agricultural samples aggregating 25
source datasets. Each feature has a representative `lon`/`lat`, an `is_crop` flag, a
harmonized FAO crop group in `classification_label` (present for ~33k crop samples; the
manifest's "33k points have multiclass crop labels"), a raw free-text `label` (350 messy
values, many non-English — not used), and an `export_end_date` (end of the ~1-year EO
window the label describes). Geometry is Point or Polygon, but `lon`/`lat` give the
representative point for every record, so we use those directly (point dataset). Only the
labels file was downloaded; the 21 GB `eo_data.tar.gz` and feature tarballs are not needed
since we only require coordinates + labels.

## Class mapping (unified scheme)

Multiclass crop labels are used where available (`classification_label`), with crop/non-crop
as the fallback for the ~62k records without a harmonized group:

| id | name | source rule |
|----|------|-------------|
| 0 | non_crop | `is_crop == 0` |
| 1 | cereals | `classification_label == 'cereals'` |
| 2 | fruits_nuts | `classification_label == 'fruits_nuts'` |
| 3 | vegetables_melons | `classification_label == 'vegetables_melons'` |
| 4 | leguminous | `classification_label == 'leguminous'` |
| 5 | oilseeds | `classification_label == 'oilseeds'` |
| 6 | beverage_spice | `classification_label == 'beverage_spice'` |
| 7 | sugar | `classification_label == 'sugar'` |
| 8 | root_tuber | `classification_label == 'root_tuber'` |
| 9 | other_crop | `classification_label == 'other'` (crop, no FAO group) |
| 10 | crop_unspecified | `is_crop == 1`, no harmonized group |

`classification_label` is fully consistent with `is_crop` (all named crop groups have
`is_crop==1`; `'non_crop'` has `is_crop==0`), so no conflicts arise. 1 feature with null
geometry/lat/lon was dropped.

Raw usable counts (before balancing): non_crop 29,735; cereals 9,287; fruits_nuts 3,730;
vegetables_melons 2,927; leguminous 1,368; oilseeds 1,230; beverage_spice 1,004; sugar 525;
root_tuber 449; other_crop 7,537; crop_unspecified 37,393 (total 95,185).

## Sampling & time

- Balanced to ≤ 1000 per class (`balance_by_class`, seed 42), well under the 25k cap.
  Final counts: all classes 1000 except sugar 525 and root_tuber 449 (their full pools).
  Total 9,974.
- Output is one dataset-wide point table `points.json` (spec §2a), not per-point tifs
  (pure 1×1 sparse points).
- **Time range:** 1-year window ending at each label's `export_end_date` — i.e.
  `[export_end - 365 days, export_end]`. This matches CropHarvest's own convention of
  exporting the 12 months of EO data up to `export_end_date`. All windows are within the
  Sentinel era (2016+). No change labels.
- All source splits used (both `is_test` true and false).

## Caveats

- Locations are representative points; some source polygons are field-sized, but we treat
  every label as a single 10 m point per the manifest `label_type: points`.
- `crop_unspecified` and `other_crop` are coarse "crop but unknown type" buckets; keep this
  in mind if using the fine crop classes only.
- Coordinate precision varies by source dataset (aggregated field surveys/declarations).

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.cropharvest
```
Idempotent: the labels download skips if present; the point table is regenerated
deterministically (seeded balancing).
