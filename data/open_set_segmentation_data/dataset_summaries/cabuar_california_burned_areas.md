# CaBuAr (California Burned Areas)

- **Slug**: `cabuar_california_burned_areas`
- **Status**: **completed** — task_type = **classification** (dense per-pixel, binary), **1617 samples**
- **Family / region**: fire / California, USA
- **Source**: CaBuAr — Rege Cambrin, Colomba, Garza 2023, *IEEE Geoscience and Remote
  Sensing Magazine*, doi:10.1109/MGRS.2023.3292467. HuggingFace dataset
  `DarthReca/california_burned_areas`
  (https://huggingface.co/datasets/DarthReca/california_burned_areas).
- **License**: CDLA-Permissive-2.0 (README frontmatter; the HF loader script/manifest also
  say "OpenRAIL" — either is permissive for research use).
- **Access**: public HuggingFace, no credentials.

## What the dataset is

Sentinel-2 pre/post-fire acquisitions over California wildfires with **binary burned-area
masks** derived from **CAL FIRE** (California Dept. of Forestry and Fire Protection) fire
perimeters, mapped onto the imagery. We used the pre-patched file
`raw/patched/512x512.hdf5`: **534 patches** of 512×512 px at **20 m/pixel** (the Sentinel-2
20 m grid), each keyed `{uuid}_{patch}` and holding `post_fire` (12-band uint16), optional
`pre_fire`, and a `mask` (uint16, values {0,1}; **1 = burned**). Only patches containing at
least one burned pixel are present in this file. Per-patch georeferencing (EPSG CRS + x/y
pixel-center coordinate arrays + post-fire acquisition timestamp) comes from the companion
`metadata.parquet` (`post==True` rows; keyed on uuid+patch). All acquisitions are 2018–2022
(entirely in the Sentinel era; nothing filtered on the pre-2016 rule).

## Class scheme (dense per-pixel classification)

| id | name     | definition |
|----|----------|------------|
| 0  | unburned | mask == 0, among observed pixels (outside the CAL FIRE perimeter) |
| 1  | burned   | mask == 1 (inside the CAL FIRE perimeter, mapped onto post-fire S2) |
| 255| nodata   | all-12-band-zero fill at Sentinel-2 tile edges (present in ~14% of patches) |

The manifest's two classes ("burned", "unburned") map directly. `unburned` (0) is the
background class; both are retained.

## Processing (label_type = dense_raster)

- Source patches are already in **local UTM at 20 m**. Each 512×512 patch is cut into
  32×32 (20 m) blocks and **upsampled 2× with nearest resampling** to a **64×64 tile at
  10 m** (categorical label → nearest, never bilinear). 512 is divisible by 32, so the grid
  is exact (16×16 = 256 candidate tiles/patch).
- **Nodata**: pixels where the post-fire image is all-zero (S2 tile-edge fill) are set to
  255 so padding is not mislabeled as `unburned`. Tiles with > 50 % nodata are skipped.
- Georeferencing per tile is computed directly from the block's UTM pixel-center coords
  (`x_left = x0 + c0·20 − 10`, `y_top = y0 − r0·20 + 10`), giving integer 10 m pixel bounds
  in the patch CRS (EPSG:32610 / 32611).
- **Sampling**: tiles-per-class balanced (spec §5), ≤ 1000 tiles/class, rarer class
  (burned) filled first. Result: **1617 tiles** — **1024 contain burned**, **1000 contain
  unburned** (most burned tiles also contain unburned; pure-unburned tiles are drawn from
  unburned blocks within the same fire patches). Well under the 25k cap.
- **Time / change**: the burn is an **event/change** label. `change_time` = the post-fire
  Sentinel-2 acquisition timestamp (a **post-event** date — the fire occurred shortly
  before it, between the pre- and post-fire acquisitions). Instead of one centered window
  we now emit **two independent six-month windows** via
  `io.pre_post_time_ranges(change_time, pre_offset_days=90)`: a **`post_time_range`** that
  starts at `change_time` and runs ~6 months (≤183 days) forward, and a **`pre_time_range`**
  that **ends 90 days before `change_time`** (a guard offset, since the fire precedes the
  acquisition) and spans ~6 months (≤183 days) backward from there — placing the pre window
  entirely before the true fire. `time_range` is `null`. The mask marks *where* the burn
  occurred; pretraining pairs a "before" stack with an "after" stack and probes on their
  difference.

## Verification

- Sampled `.tif`s: single-band uint8, EPSG:326xx UTM at 10 m, 64×64, values ⊆ {0,1} with
  255 declared nodata; every `.tif` has a matching `.json` with `time_range` null, a
  `pre_time_range` and `post_time_range` (each ≤183 days), and `change_time` set.
- All sampled tile centers fall inside California (lon −125…−114, lat 32…42).
- **Round-trip check**: 8/8 written tiles exactly equal the 2×-upsampled source block they
  came from (label content + geometry consistent).
- Georeferencing is exact because coordinates come straight from the Sentinel-2 product
  geotransforms in `metadata.parquet`; a live S2 image overlay was not run (coords are
  authoritative S2-grid UTM).

## Judgment calls / caveats

- Used the pre-patched `512x512.hdf5` (534 burned patches) rather than the full-tile
  `raw/complete/california_*.hdf5` (only 0–9 available in raw): the patched file gives clean
  per-patch georeferencing via `metadata.parquet` and every patch is guaranteed to contain
  burn. Enough for the ≤1000/class target.
- Source resolution is **20 m**; upsampled 2× (nearest) to the pipeline's 10 m grid. The
  effective label detail is native 20 m.
- `change_time` anchored on the post-fire acquisition (not the exact ignition date, which is
  not in the file). The 90-day pre-window guard offset pushes the "before" window back so it
  sits entirely before the fire, so this is not ill-posed.
- Only burn-containing patches exist in the source, so there are no fully-clean "far from
  any fire" negative scenes; downstream assembly adds cross-dataset negatives (spec §5).

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.cabuar_california_burned_areas
```

Raw inputs on weka: `raw/cabuar_california_burned_areas/{512x512.hdf5, metadata.parquet}`
(downloaded from HuggingFace). Script is idempotent (skips already-written `{id}.tif`).
