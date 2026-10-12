# Digital Earth Africa Cropland Reference Data

- **Slug:** `digital_earth_africa_cropland_reference_data`
- **Status:** completed
- **Task type:** classification (sparse point segmentation, 2 classes)
- **Num samples:** 2000 (1000 cropland + 1000 non-cropland)
- **Family / region:** cropland / Africa (continent-wide)
- **License:** CC-BY-4.0

## Source & access

Digital Earth Africa **crop-mask** GitHub repo
(`https://github.com/digitalearthafrica/crop-mask`). The reference labels are open and live
directly in the repo — no credentials required. We use the project's merged, continent-wide
reference set:

```
https://raw.githubusercontent.com/digitalearthafrica/crop-mask/main/testing/combined_training_data.geojson
```

Downloaded to `raw/digital_earth_africa_cropland_reference_data/combined_training_data.geojson`
(~10 MB, 30,448 features). This file is the output of the project's
`Reference_data_merge` step, which merges the crop/non-crop reference samples that were
manually photo-interpreted via **Collect Earth Online** across the DE Africa
agro-ecological zones (Central, Eastern, Indian Ocean, Northern, Sahel, South-Eastern,
Southern, Western), together with a few pre-existing crop reference sets. Each feature is a
small **field-scale Polygon** carrying a single attribute `Class` (`1` = cropland,
`0` = non-cropland). No per-feature date attribute is present.

Raw class pool: `0` (non-cropland) = 18,026; `1` (cropland) = 12,422. Coverage spans the
whole continent (lon −17.5…57.8, lat −34.6…37.2).

## Label / class mapping

Binary crop mask. Class ids are kept identical to the source `Class` value so provenance is
exact:

| id | name | source `Class` |
|----|------|----------------|
| 0 | non-cropland | 0 |
| 1 | cropland | 1 |

(The manifest lists the classes as `["cropland", "non-cropland"]`; we use source-native ids
`0=non-cropland, 1=cropland`, which also matches the sibling `olmoearth_africa_crop_mask`.)

## Representation decision (points vs polygons)

`label_type` is "points/polygons": the samples are Collect-Earth-Online interpreted
locations whose footprints are small field-scale polygons. Per the manifest instruction and
spec §2a, these sparse reference samples are represented as **points** — one `Point` at each
polygon's **centroid**, labeled `0/1` — and written to a single dataset-wide
`points.geojson` (FeatureCollection), rather than as per-feature GeoTIFFs. Pretraining
projects each point onto the S2 grid as a 1×1 label. 56 empty/invalid geometries were
repaired via `buffer(0)` where possible and skipped only if still empty (none were lost in
practice; all 30,448 produced a valid centroid).

## Time range

Cropland is a seasonal/annual state and the source carries no per-sample date, so each point
is assigned a **1-year window anchored on 2019** (`2019-01-01 … 2020-01-01`), the DE Africa
crop-mask reference campaign period and squarely within the manifest's declared 2018–2020
valid range. `change_time` is null (not a change/event dataset).

## Sampling / balancing

`balance_by_class(records, "label", per_class=1000)` (spec §5 default: up to 1000
locations/class, 25k dataset cap). With 2 classes this yields **1000 cropland + 1000
non-cropland = 2000 samples**, far under the 25k cap. Selection is seeded/deterministic.

## Verification (§9)

- `points.geojson`: `FeatureCollection`, `task_type=classification`, `count=2000`; every
  feature is a `Point` with WGS84 `[lon, lat]`, `properties.label ∈ {0,1}`, and a valid
  ≤1-year `time_range`. Label distribution 1000/1000. No out-of-range coords, no
  over-length time ranges.
- `metadata.json`: classification schema with both classes (ids 0,1) and per-class
  descriptions; `nodata_value=255`; `num_samples=2000`; class ids cover all label values
  present.
- `registry_entry.json`: `completed`, `task_type=classification`, `num_samples=2000`.
- Spatial sanity: all centroids fall within the African landmass extent; centroids are
  computed directly from the validated source field polygons. A full Sentinel-2 image
  overlay was not rendered (point-only labels from an authoritative, validated reference
  set); coordinates were validated against the source geometries instead.

## Caveats

- Points are polygon centroids; the original field-scale footprint is not preserved (labels
  are 1×1 as required for sparse point datasets).
- No per-sample acquisition date in the source → a single representative 2019 window is used
  for all points. This is appropriate for a persistent annual cropland state.
- This is the **preferred reference** product; prefer it over the derived DE Africa cropland
  *map*.

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.digital_earth_africa_cropland_reference_data
```

Idempotent: re-downloading skips the existing raw file and re-writing overwrites the outputs
atomically. Outputs land in
`/weka/dfive-default/helios/dataset_creation/open_set_segmentation/datasets/digital_earth_africa_cropland_reference_data/`.
