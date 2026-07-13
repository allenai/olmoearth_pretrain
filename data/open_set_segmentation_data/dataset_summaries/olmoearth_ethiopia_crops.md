# olmoearth_ethiopia_crops

**Status:** completed · classification · 1,453 samples (point table)

## Source
Local rslearn eval `/weka/dfive-default/rslearn-eai/datasets/olmoearth_evals/ethiopia_crops`
(OlmoEarth internal eval; `have_locally: true`, not copied). Each window is one manually
field-surveyed crop-type point over Ethiopia. The class name, lon/lat, split, and a
growing-season time range live in window `metadata.json` `options` (`lon`, `lat`, `label`,
`split`) and are duplicated in the `label` vector layer (`layers/label/data.geojson`,
property `label`). Groups: `train` (574) + `test` (1,956) = 2,530 labeled points; all
splits used. All points carry lon/lat and post-2016 time ranges.

## Processing
- Parallel-scanned all window `metadata.json` (`Pool(64)`) to collect
  (lon, lat, label, time_range, source_id).
- Classes mapped in manifest order → ids 0–3: wheat, barley, maize, teff (short
  staple-cereal definitions in `metadata.json` `classes[].description`).
- Sparse point segmentation (label_type `points`, single 10 m pixel) → **point table**
  (`points.geojson`, spec §2a) via `io.write_points_table`, not per-point GeoTIFFs.
  Each feature: `Point [lon, lat]`, `properties.label=class_id`, `time_range`, `source_id`.
- **Time range:** each point keeps its source ~1-year growing-season window verbatim
  (anchored on the labeled 2019/2020 season; e.g. 2019-10-30 → 2020-10-30) rather than
  snapping to a Jan–Jan calendar year — this preserves the phenologically-correct window
  used by the eval. All windows span 366 days (2020 leap year).
- Balanced to ≤1000 per class (seeded shuffle, `balance_by_class`). 25k total cap not
  binding (1,453 « 25,000).

## Output
- `datasets/olmoearth_ethiopia_crops/points.geojson` — 1,453 point features.
- `datasets/olmoearth_ethiopia_crops/metadata.json` — class map + counts.
- `raw/olmoearth_ethiopia_crops/SOURCE.txt` — pointer to source (local; not copied).

## Class counts
Source distribution: wheat 2,077 · teff 255 · barley 102 · maize 96.
Selected (≤1000/class): wheat 1000 (truncated from 2,077) · teff 255 · barley 102 ·
maize 96 = 1,453. Rare classes (barley/maize/teff) kept in full per spec §5 — none dropped.

## Reproduce
`python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.olmoearth_ethiopia_crops`
(idempotent; deterministic seed, atomically rewrites `points.geojson`).

## Notes / judgment calls
- 1×1 point labels carry no spatial context by design; paired with S2/S1/Landsat at
  pretraining time by lon/lat + time overlap.
- Preserved source per-window growing-season `time_range` instead of `io.year_range()`.
  Each is 366 days (leap year), marginally over the nominal 360-day "≤1 year" guidance but
  consistent with a single crop season and with the lcmap worked-example precedent
  (`year_range` likewise yields 365/366 days).
- No fabricated negatives; no background class (positive-only reference points, spec §5) —
  assembly supplies negatives from other datasets.
- Verified: FeatureCollection with 1,453 unique zero-padded ids, all coords inside the
  Ethiopia bbox, label ids 0–3 matching `metadata.json`, task_type=classification.
