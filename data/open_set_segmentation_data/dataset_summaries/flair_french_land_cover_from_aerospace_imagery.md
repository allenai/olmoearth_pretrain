# FLAIR (French Land cover from Aerospace ImageRy)

- **Slug**: `flair_french_land_cover_from_aerospace_imagery`
- **Status**: **completed**
- **Task type**: classification (dense per-pixel land cover)
- **Num samples**: 6,225 label tiles (`locations/{id}.tif` + `.json`)
- **Family / region**: land_cover / Metropolitan France
- **License**: Etalab Open Licence 2.0 (permissive open data; use permitted)

## Source & access

IGN France's FLAIR dataset, distributed on Hugging Face as
[`IGNF/FLAIR-1-2`](https://huggingface.co/datasets/IGNF/FLAIR-1-2). The release ships 60
per-département ZIP archives under `data/{train-val, flair#1-test, flair#2-test}/D0XX_YYYY.zip`
(121 GB total). Each archive bundles, per 512×512 patch: a 0.2 m 5-band aerial image
(`aerial/…/IMG_*.tif`), a Sentinel-2 time series (`sentinel/…`), and a **0.2 m land-cover
mask** (`labels/Z*/MSK_*.tif`, uint8, 19 classes). Only the masks are needed here.

**We do not download the 121 GB of archives.** The masks are tiny; we stream only the
`MSK_*.tif` members out of each remote ZIP over HTTP range reads via
`huggingface_hub.HfFileSystem` + Python `zipfile` (~15 ms/mask). No credentials required
(public dataset). Scanned tile records are cached to
`raw/{slug}/scan_cache.pkl`; `raw/{slug}/SOURCE.txt` documents the access method.

- Masks are georeferenced in **RGF93 / Lambert-93 (EPSG:2154)**, but stored with an
  unlabeled `LOCAL_CS` WKT (no embedded EPSG code). We assign EPSG:2154 explicitly on read
  (verified: coordinates match the Lambert-93 France envelope, and reprojected centroids
  land in metropolitan France).
- All 93,462 patches (train/val + both official test sets — all splits are fair game as
  pretraining labels) were scanned.
- The département folder suffix carries the aerial **acquisition year** (`D041_2021` → 2021).
  All patches are 2018–2021, i.e. fully within the Sentinel era.

## VHR → 10 m handling

FLAIR masks are 0.2 m / 512×512 (a ~102.4 m footprint) — VHR-native, far finer than the
10 m S2 grid. Per the task spec (§4), each mask is **reprojected from EPSG:2154 to a local
UTM zone at 10 m/pixel using MODE resampling** (categorical majority; never bilinear), via
`rasterio.warp.reproject`. Each 102.4 m patch collapses to **one ~11×11 tile** (sizes range
11–12 px on each axis; all ≤64). The local UTM projection is chosen from each patch
centroid's lon/lat (`get_utm_ups_projection`); output tiles are written with rslearn's
`GeotiffRasterFormat` (single-band uint8, north-up, 10 m). Time range = the patch's
acquisition-year 1-year window (`year_range`).

## Class mapping (FLAIR 13-class baseline nomenclature)

FLAIR has 19 source classes but recommends a **13-class baseline** that merges the seven
rare/fine classes into "other". We adopt exactly that scheme (it is also what 10 m mode
resampling supports — see below):

| out id | name | FLAIR source class(es) | train/val pixel % |
|---|---|---|---|
| 0 | building | 1 | 8.14 |
| 1 | pervious surface | 2 | 8.25 |
| 2 | impervious surface | 3 | 13.72 |
| 3 | bare soil | 4 | 3.47 |
| 4 | water | 5 | 4.88 |
| 5 | coniferous | 6 | 2.74 |
| 6 | deciduous | 7 | 15.38 |
| 7 | brushwood | 8 | 6.95 |
| 8 | vineyard | 9 | 3.13 |
| 9 | herbaceous vegetation | 10 | 17.84 |
| 10 | agricultural land | 11 | 10.98 |
| 11 | plowed land | 12 | 3.88 |
| 12 | other | 13–19 (merged) | ~0.6 combined |

Source→output is a uint8 LUT: source 1..12 → 0..11; source 13..19 → 12; source 0 (should
not occur) → nodata 255. Nodata sentinel = **255**.

### Classes merged/coarsened at 10 m (folded into "other", id 12)

- **13 swimming pool** (0.01 %) and **18 greenhouse** (0.12 %): individual objects are a
  few metres across — **unresolvable as distinct classes at 10 m**.
- **14 snow** (0.15 %), **15 clear cut** (0.15 %), **16 mixed** (0.05 %), **17 ligneous**
  (0.01 %), **19 other** (0.14 %): all extremely rare and/or semantically thin at 10 m.

Grouping these (rather than dropping) preserves them as a real catch-all class and matches
IGN's published baseline. No source class is discarded outright; the 12 main classes are
all kept and individually resolvable at 10 m.

## Sample counts (tiles-per-class; a tile counts toward every class it contains)

Selection: one tile per patch; **tiles-per-class balanced to ≤1000/class, rarest-class
first, ≤25,000 total** (`sampling.MAX_SAMPLES_PER_DATASET`). 6,225 tiles selected from
93,462 candidate patches.

| id | name | tiles | | id | name | tiles |
|---|---|---|---|---|---|---|
| 0 | building | 2164 | | 7 | brushwood | 2649 |
| 1 | pervious surface | 2890 | | 8 | vineyard | 3890 |
| 2 | impervious surface | 2894 | | 9 | herbaceous vegetation | 5258 |
| 3 | bare soil | 1000 | | 10 | agricultural land | 1639 |
| 4 | water | 1087 | | 11 | plowed land | 1114 |
| 5 | coniferous | 1274 | | 12 | other | 2279 |
| 6 | deciduous | 4218 | | | | |

Counts exceed 1000 for common classes because tiles are multi-label: the greedy rarest-first
selection stops adding a tile only once *every* class it contains is already ≥1000, so
co-occurring common classes accumulate past 1000 while rare-ish classes (bare soil hit
exactly its 1000 target) drive selection. Every class comfortably clears the downstream
minimum, so no rare-class handling was needed.

## Caveats / judgment calls

- **13-class merge vs full 19 classes**: chose IGN's 13-class baseline. The seven merged
  classes are either sub-10 m objects (pool, greenhouse) or vanishingly rare; keeping them
  separate would add noise, not signal, at 10 m. This is the main modelling decision.
- **CRS is an unlabeled LOCAL_CS**: masks lack an embedded EPSG; we hard-assign EPSG:2154.
  Verified correct via France-envelope coordinate check and reprojected-centroid check
  (5 sampled tiles all land inside metropolitan France).
- **~11×11 tiles**: each FLAIR patch is only ~102 m, so tiles are small (well under 64).
  This is expected for a VHR-native product resampled to 10 m; it means one patch = one
  co-location sample.
- **pervious vs impervious surface kept separate** (source 2 vs 3), unlike the manifest's
  "pervious/impervious surface" shorthand — the source annotates them distinctly and both
  are resolvable as broad urban cover at 10 m.
- **Mode resampling** picks the majority 0.2 m class in each 10 m cell; sub-dominant fine
  features within a cell are dropped by design.
- Spatial sanity check: verified tile CRS/bounds are UTM @ 10 m and centroids fall in
  France; a full S2 overlay was not rendered (no misalignment observed in the geometry).

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.flair_french_land_cover_from_aerospace_imagery
```

Idempotent: the remote scan is cached to `raw/{slug}/scan_cache.pkl` and already-written
`locations/{id}.tif` are skipped. `--workers` controls remote-read parallelism (default 16),
`--write-workers` the GeoTIFF write pool (default 64).
