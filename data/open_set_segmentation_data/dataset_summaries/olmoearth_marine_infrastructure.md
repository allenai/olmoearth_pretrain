# OlmoEarth marine infrastructure

- **Slug:** `olmoearth_marine_infrastructure`
- **Status:** completed
- **Task type:** classification (object detection encoded as per-pixel classes)
- **Num samples:** 3000
- **Family / region:** infrastructure / global oceans
- **License:** ODbL/internal

## Source

Existing OlmoEarth / Satlas offshore marine-infrastructure detection eval, a **local
rslearn dataset** (`have_locally: true`, not copied):

```
/weka/dfive-default/rslearn-eai/datasets/marine_infra/dataset_v1/20250605
```

`raw/olmoearth_marine_infrastructure/SOURCE.txt` points at this path (no raw copy).

Single window group `label` (7197 windows). Each window is a specific-image crop already
in a **local UTM projection at 10 m/pixel** (~855×855 px) with a **~220-day** monthly-
composite `time_range`. The label layer `label` is a vector GeoJSON with one `Point`
feature per manually annotated object; `properties.category` ∈
{`platform`, `turbine`, `vessel`, `power`, `aerialway`} (config `class_property_name`
= `category`, declared `class_names` = `[unknown, platform, turbine]`). Point coordinates
are in the window's projection (pixel) coordinates, matching the window `bounds`, so no
reprojection is needed. `metadata.options.has_objects` flags object-bearing windows
(True: 3022, False: 4175).

Scan totals (all 7197 windows): projection 10 m for all; feature counts —
turbine 8791, vessel 5354, platform 4459, power 189, aerialway 4; splits train 6289 /
val 908; start years 2016–2022 (**all post-2016**); `time_range` span uniformly 220 days
(none > 1 year). 2060 windows contain ≥1 platform/turbine target; 4175 are object-free.

## Class scheme

The manifest `label_type` is `bboxes` but the on-disk annotations are object-centroid
**points**, so this is processed as a **detection** dataset (spec §4). Unified two-target
class map (spec §5, multi-target → one class map):

| id | name | description |
|----|------|-------------|
| 0 | background | open water / non-infrastructure ocean surface within the tile |
| 1 | platform | offshore platform (oil/gas platform, offshore substation) |
| 2 | turbine | offshore wind turbine |
| 255 | nodata/ignore | detection buffer rings + non-target annotated objects |

Non-target categories (`vessel`, `power`, `aerialway`, and any `unknown`) are **not**
targets of this dataset; where they fall inside a tile they are written as **nodata (255)
ignore** (with the same buffer) rather than being called background — so the model is
neither penalized for them nor taught a wrong class. Windows whose only objects are
non-targets are dropped (not usable positives, and unsafe as negatives since they do
contain annotated objects).

## Encoding, time range, sampling

- **Detection encoding** (`sampling.encode_detection_tile`): one **32×32** UTM-10 m
  context tile per platform/turbine detection, centered on it, written in the window's own
  UTM projection (source already local UTM @ 10 m — no reprojection). The detection is a
  **1×1** positive of its class id, ringed by a **10 px nodata (255) buffer** (centroids
  are not pixel-exact); all other pixels are background (0). Every other platform/turbine
  of the same window falling inside the tile is also marked; non-target objects in the tile
  are marked nodata.
- **Negatives:** background-only tiles from object-free windows (`has_objects == false`),
  so the background class has spatially-meaningful negatives (spec §5 detection exception).
- **Time range:** each sample keeps its window's own ~220-day monthly-composite
  `time_range` (< 1 year; marine infrastructure is static across that window, spec §5).
  `change_time` = null (not a change dataset). All labels post-2016.
- **Splits:** all splits used (pretraining-agnostic, spec §5).
- **Balancing:** `sampling.balance_by_class(per_class=1000)` on each tile's center class →
  up to 1000 platform tiles + 1000 turbine tiles, plus 1000 background negatives. Well
  under the 25k cap.

## Sample counts (final)

- Total: **3000** samples (all 32×32 uint8, nodata 255).
- platform positive tiles: **1000**; turbine positive tiles: **1000**; background
  negatives: **1000**.
- Tiles containing each class (a tile can contain both): platform in 1000, turbine in
  1001, background in all 3000, nodata buffer in 2000 positive tiles.

## Verification (spec §9)

- 3000 `.tif` + 3000 matching `.json`, all paired. Every tif: single band, **uint8**,
  **32×32**, projected UTM CRS, resolution **(10, −10)**. Pixel values ∈ {0, 1, 2, 255}
  only; `metadata.json` class ids {0,1,2} cover all non-nodata values.
- All `time_range`s are 220 days (≤ 1 year); all `crs` are `EPSG:*`.
- **Spatial sanity:** for 6 positive samples, read the source Sentinel-2 (B08/NIR) at the
  tile's CRS/bounds and compared NIR at labeled pixels vs. background median — labeled
  pixels are consistently brighter (ratios 1.1–4.2×), confirming labels sit on real
  offshore structures against dark water. No misalignment observed.

## Caveats

- Annotations are centroid points, not full footprints; the 1×1 positive + 10 px ignore
  buffer is the intended detection encoding for such point-like offshore structures.
- Non-target categories (vessel/power/aerialway) are ignored, not modeled; a dedicated
  vessel dataset (`olmoearth_sentinel_2_vessels`) covers ships.
- 2060 windows contain targets but only 1000/class are kept after balancing; the remainder
  are available if a larger sample is desired later.

## Reproduce

From repo root `.` (idempotent; skips existing outputs):

```bash
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.olmoearth_marine_infrastructure
```
