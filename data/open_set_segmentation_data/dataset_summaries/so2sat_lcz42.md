# So2Sat LCZ42

- **Slug:** `so2sat_lcz42`
- **Status:** completed
- **Task type:** classification (17 Local Climate Zones)
- **Samples:** 15,721 uniform-class 32×32 GeoTIFF tiles
- **Source:** So2Sat LCZ42 v4.2 "data with geolocation", TUM mediaTUM record
  [1836598](https://mediatum.ub.tum.de/1836598), doi:10.14459/2025mp1836598.002.
  License **CC-BY-4.0**. Annotation: manual (expert-labeled LCZ). Family:
  `local_climate_zone`.

## What the source is

So2Sat LCZ42 (Zhu et al. 2020) is ~400k co-registered Sentinel-1/2 32×32 image patches
over 42+ global cities, each hand-labeled by experts into one of the 17
[Local Climate Zones](https://doi.org/10.1175/BAMS-D-11-00019.1) (Stewart & Oke 2012). It
is distributed as ML-ready HDF5 patch stacks (`sen1`, `sen2`, one-hot `label`). Early
versions **stripped geocoordinates** (a common fast-reject case for this bank). The **v4.2
release adds per-patch geolocation** (`*_geo.h5`: EPSG code + a worldfile `tfw` affine +
`city`), which is exactly what lets each 320 m patch be placed on the S2 grid.

## Triage decision: ACCEPT

- **Georeferencing recoverable** via the v4.2 `*_geo.h5` files → not a no-geocoordinates
  reject.
- **Post-2016**: labels/imagery are 2017 (manifest range 2017–2018).
- **Scene/patch label → uniform-class tile (spec §4 scene-level).** An LCZ label is a
  patch-level class, but So2Sat patches are sampled from **expert-delineated homogeneous
  LCZ polygons**, i.e. genuinely coherent land-cover / urban-morphology patches. The LCZ
  classes (dense trees, water, low plants, compact high-rise, …) are real land-cover
  types, so we emit a **uniform-class 32×32 tile** per patch (all pixels = the LCZ id)
  rather than rejecting as patch classification. 32×32 @ 10 m = the native 320 m footprint
  (≤ 64 cap).

## Access / download (label-only; NO imagery pulled)

- **Geolocation**: the small corrected v4.2 geo files are downloaded fully from the
  Hugging Face mirror `zhu-xlab/So2Sat-LCZ42` (`v4/{split}_geo.h5`, ~0.24 MB each for
  val/test). Verified: EPSG codes are identical to the mediaTUM originals for val/test and
  the corner coordinates are consistent; the correction mainly re-ordered `tfw` into proper
  worldfile order `[A,D,B,E,C,F]` (x_res=10, y_res=−10, C/F = upper-left corner) and added
  `city`. City→EPSG matches the official `save_geotiff.py` table (e.g. jakarta=32748,
  moscow=32637, nairobi=32737, munich=32632, tehran=32639).
- **Labels**: the LCZ one-hot `label` array lives only inside the big `sen1/sen2/label`
  HDF5. mediaTUM serves those **uncompressed** with HTTP Range support, so a new shared
  helper (`download.read_remote_h5_dataset` / `download.HttpRangeFile`) reads **only the
  contiguous `label` dataset** via a byte-range read (~3 MB per split). The ~3.5 GB of
  Sentinel-1/2 imagery per split is never fetched. `argmax` of the one-hot → class id,
  cached to `raw/so2sat_lcz42/{split}_labels.npy` for idempotent re-runs.

## Splits used and the excluded training set

**Used: validation + testing** — 48,307 patches over **10 cities across continents**
(guangzhou, jakarta, moscow, mumbai, munich, nairobi, sanfrancisco, santiago, sydney,
tehran).

**Excluded: `training.h5` (42 cities, 352,366 patches).** The mediaTUM data server would
**not serve HTTP Range requests on that single 52 GB file** within a workable time budget
(even a 1-byte probe did not return in > 6 min, whereas the 3.5 GB val/test files each read
in ~3 min). This is a **source-server throughput limit, not a data problem**, and it is
retryable. To add the training patches later: place a
`raw/so2sat_lcz42/training_labels.npy` (uint8 `argmax` of the `training.h5` one-hot label —
e.g. once the server serves the range, or by downloading + `gunzip`-ing the 16 GB HF
`v4/training.h5.gz`) and add `"training"` to `SPLITS`; everything else is idempotent.

## Labels / class scheme

17 LCZ classes, in the So2Sat one-hot column order (built types LCZ 1–10, then natural
types LCZ A–G), class ids 0–16, uint8, nodata=255 (unused — every tile is a single valid
class). Full Stewart & Oke definitions are stored per-class in `metadata.json`.

| id | name | LCZ | selected |
|----|------|-----|----------|
| 0 | compact_high_rise | LCZ 1 | 522 |
| 1 | compact_mid_rise | LCZ 2 | 1000 |
| 2 | compact_low_rise | LCZ 3 | 1000 |
| 3 | open_high_rise | LCZ 4 | 1000 |
| 4 | open_mid_rise | LCZ 5 | 1000 |
| 5 | open_low_rise | LCZ 6 | 1000 |
| 6 | lightweight_low_rise | LCZ 7 | 977 |
| 7 | large_low_rise | LCZ 8 | 1000 |
| 8 | sparsely_built | LCZ 9 | 1000 |
| 9 | heavy_industry | LCZ 10 | 1000 |
| 10 | dense_trees | LCZ A | 1000 |
| 11 | scattered_trees | LCZ B | 815 |
| 12 | bush_scrub | LCZ C | 1000 |
| 13 | low_plants | LCZ D | 1000 |
| 14 | bare_rock_or_paved | LCZ E | 407 |
| 15 | bare_soil_or_sand | LCZ F | 1000 |
| 16 | water | LCZ G | 1000 |

All 17 classes are present. Four rare built classes (LCZ 1, 7, B, E) fall short of the
1000/class target from val+test alone; per spec §5 sparse classes are kept as-is and the
downstream assembly step drops any that end up too small. Adding the excluded training set
later would raise every class toward the cap.

## Sampling / tile / time

- **Tile:** 32×32 @ 10 m, single band, uint8, **local UTM reused directly from the source**
  (no reprojection — the patches are already UTM at 10 m). Upper-left corner from `tfw`
  snapped to the 10 m pixel grid (sub-metre off-grid rounding, < 0.1 px).
- **Sampling:** `balance_by_class(per_class=1000, total_cap=25000)` (17×1000 < 25k, so the
  full 1000/class target applies). Deterministic/idempotent (seeded, stable ordering; skips
  existing `.tif`s).
- **Time range:** uniform **2017** 1-year window `[2017-01-01, 2018-01-01)` — So2Sat S2
  imagery is 2017 and there is no per-patch acquisition date (spec §5 seasonal/annual).
  `change_time=null` (LCZ is a static land-cover/morphology label, not a change label).

## Verification (spec §9)

- 15,721 `.tif` + 15,721 matching `.json`; `metadata.json` covers class ids 0–16.
- Sampled tiles: single band, `uint8`, UTM CRS at 10 m, 32×32, each a single valid class
  id, `time_range` = 1 year.
- **Spatial sanity:** reprojecting sampled tile centers to WGS84 lands them in the correct
  city — e.g. `EPSG:32643 → (73.12, 19.06)` Mumbai, `32639 → (51.42, 35.79)` Tehran,
  `32737 → (36.73, −1.41)` Nairobi, `32610 → (−122.43, 37.87)` San Francisco Bay,
  `32637 → (37.35, 55.86)` Moscow. Georeferencing is correct.

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.so2sat_lcz42 --workers 64
```

Outputs: `datasets/so2sat_lcz42/{metadata.json, registry_entry.json,
locations/{id}.tif,.json}` on weka; raw geo files + cached label `.npy` under
`raw/so2sat_lcz42/`.
