# OlmoEarth vessel attributes (type)

- **slug**: `olmoearth_vessel_attributes_type`
- **task_type**: classification (object detection **by vessel type** encoded as per-pixel classes)
- **status**: completed
- **num_samples**: 8000 (1000 positive tiles for each of 8 populated vessel types)
- **classes**: `0 = background` (open water), `1 = cargo`, `2 = tanker`, `3 = passenger`,
  `4 = service`, `5 = tug`, `6 = pleasure`, `7 = fishing`, `8 = enforcement`, `9 = sar`;
  `255 = nodata/ignore` (buffer ring). Class ids 1–9 follow the eval order
  (`rslp.vessel_attribute.train.SHIP_TYPE_CATEGORIES`), offset by +1 so background is 0.

## Source

Local rslearn dataset (`have_locally=true`, not copied — see
`raw/olmoearth_vessel_attributes_type/SOURCE.txt`):

```
/weka/dfive-default/rslearn-eai/datasets/sentinel2_vessel_attribute/dataset_v1/20250205
```

This is the existing OlmoEarth **vessel-attribute** eval (distinct from the plain
vessel-*detection* eval `olmoearth_sentinel_2_vessels`). Two window groups —
`detections_bigtable` (106,054) and `detections_jan_470k` (478,378), **584,432 windows
total**. Each window is a **128×128 per-vessel crop** in a **local UTM projection at
10 m/pixel** with a **~2-hour Sentinel-2 acquisition `time_range`**, centered on **one**
AIS-matched vessel.

- Label layer `info` (vector GeoJSON, `layers/info/data.geojson`): a single `Point`
  feature at the vessel, in the window's projection (pixel) coordinates — the vessel is
  essentially at the window center (verified offset < 1 px). Its `properties.type` is the
  vessel category, **present only for the 9 valid categories** (via
  `rslp.vessel_attribute.ship_types.VESSEL_CATEGORIES`, which also merges "other
  pleasure"→pleasure and "other fishing"→fishing). Windows whose vessel type is
  `unknown`/`other` carry **no** `type` property and are skipped — matching the eval, which
  marks those examples invalid.

## Label type note

Manifest `label_type` is `points`. The point marks a single vessel and carries its **type
attribute**, so this is processed with the tunable **detection** encoding (spec §4), where
the positive pixel's class id is the **vessel type** (not a single "vessel" class). This is
the type-classification target; the length/width/course/speed attributes are regression and
are **excluded per the manifest** ("length is regression, excluded").

## Encoding (detection by type, spec §4)

- One **32×32** (`DET_TILE`) context tile per vessel, centered on the vessel pixel, written
  in the window's **own UTM projection** (source already local UTM @ 10 m → georeferencing
  is exact, no reprojection).
- The vessel is a **1×1 positive** whose class id is its **vessel type** (1–9), ringed by a
  **10 px nodata (255) buffer** (centroids are not pixel-exact), all other pixels
  **background** (id 0). With `positive_size=1`, `buffer_size=10` the ignore ring is 21×21,
  leaving ample background in the 32×32 tile.

### Negatives — judgment call

Unlike `olmoearth_sentinel_2_vessels` (which had dedicated vessel-free `train-bg`/`valid-bg`
windows and fully-annotated scenes), this dataset has **no vessel-free windows** (every
window is centered on a vessel) and **only the central target vessel is annotated** —
neighboring vessels within a 128×128 crop are unlabeled. Sampling background-only "negative"
tiles from window corners would therefore be **unreliable** (they could contain real,
unlabeled vessels mislabeled as background). So **no separate negative tiles are emitted**.
The background class (0) is still abundantly and spatially-meaningfully represented **within
every positive tile** (the open water outside the 21×21 ignore ring). Downstream assembly
supplements cross-dataset negatives (spec §5).

Shared code reused: `sampling.balance_by_class`, `sampling.encode_detection_tile`,
`io.centered_bounds`, `io.write_label_geotiff`, `io.write_sample_json`,
`io.write_dataset_metadata`, `manifest.write_registry_entry`,
`rslearn.utils.mp.star_imap_unordered`.

## Sampling / counts (spec §5)

Typed vessels available (of 584,432 windows; the rest are `unknown`/`other`, skipped):
**433,196 total** —

| type | available | selected |
|------|-----------|----------|
| cargo | 178,473 | 1000 |
| tanker | 87,193 | 1000 |
| passenger | 26,205 | 1000 |
| service | 46,325 | 1000 |
| tug | **0** | 0 |
| pleasure | 41,325 | 1000 |
| fishing | 48,882 | 1000 |
| enforcement | 2,566 | 1000 |
| sar | 2,227 | 1000 |

Class-balanced to **1000 positive tiles per type** (`balance_by_class(key="type_id",
per_class=1000)`; 8 populated types × 1000 = **8000 tiles**, far under the 25k cap). All
splits used (pretraining-agnostic). Seeded, deterministic selection.

- **`tug` has 0 samples**: the manifest/eval scheme lists `tug` as a class, but no vessel in
  this data release maps to the `tug` category (the `VESSEL_CATEGORIES` map has a `tug` key
  but no source `vessel_category` value resolves to it here). `tug` is kept in the class map
  (id 5, so ids stay aligned with the eval) but has no tiles. Per spec §5, sparse/empty
  classes are retained, not dropped; downstream filtering removes too-small classes.

## Time range (spec §5, specific-image)

Each sample uses its **source window's own ~2-hour S2 acquisition `time_range`** (uniformly
7200 s across the samples checked — well within the specific-image / ≤ 1-year budget). No
`change_time`.

## Judgment calls

- **Detection-by-type, not single-class detection.** The positive pixel carries the vessel
  *type* (class 1–9); background = 0. This makes the dataset a multi-class type classifier,
  consistent with the OlmoEarth eval target.
- **`unknown`/`other` vessels skipped.** They are not valid type labels (the source omits
  `type` for them, and the eval treats them as invalid); ~26% of windows.
- **No separate negative tiles** (reasoning above) — differs from the
  `olmoearth_sentinel_2_vessels` precedent because there are no clean vessel-free windows and
  neighboring vessels are unlabeled. Background is still learned from each positive tile.
- **`tug` empty class retained** (id 5) to keep ids aligned with the eval.
- **Length/width/course/speed excluded** (regression attributes, out of scope per manifest).

## Verification (spec §9)

- 8000 `.tif`, each paired with a `.json`; all single-band **uint8**, **32×32**, local UTM
  CRS at **10 m**, nodata **255** (sampled 40; 27 distinct UTM zones). Positive tiles contain
  {0, type_id, 255}; union of values across the sample = {0,1,2,3,4,6,7,8,9,255} (no 5=tug,
  as expected).
- All `time_range`s exactly 2 h (≤ 1 yr), `change_time=null`; `metadata.json` class ids
  0–9 cover all values appearing in the tiles.
- **Spatial sanity**: over 15 positive samples, the brightest source-S2 **B08 (NIR)** pixel
  within 3×3 of each annotation is a **median 2.88×** (all > 1.1×) brighter than surrounding
  water, confirming positives land on vessels — no gross misalignment.
- **Idempotent by construction**: selection is deterministic (`_stable_order` + seeded
  shuffle → fixed `sample_id` assignment) and `_write_positive` skips any existing
  `{sample_id}.tif`, so re-running skips all writes.

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.olmoearth_vessel_attributes_type --workers 64
```
(Scan of 584k windows takes ~14 min; writing 8000 tiles ~1 min.)
