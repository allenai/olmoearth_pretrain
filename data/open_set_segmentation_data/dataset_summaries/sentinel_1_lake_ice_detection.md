# Sentinel-1 Lake Ice Detection

- **Slug:** `sentinel_1_lake_ice_detection`
- **Source:** ETH Zurich PRS — GitHub repo [`prs-eth/sentinel_lakeice`](https://github.com/prs-eth/sentinel_lakeice), accompanying Tom et al., *"Lake Ice Detection from Sentinel-1 SAR with Deep Learning"* (ISPRS Annals, 2020).
- **License:** MIT (labels).
- **Task type:** classification (`dense_raster`), binary lake-ice vs open-water.
- **Region / time:** four Swiss Alpine lakes, winters 2016–17 and 2017–18 (Sentinel era).
- **Status:** completed — **2000 samples** (1000 frozen + 1000 water).

## What the source provides and what we use

The repository ships the ground-truth *labels* and lake outlines as small text/vector files
(all inside the repo); the Sentinel-1 SAR rasters themselves are hosted on an ETH polybox
share that is **dead / HTTP 404** as of processing. That does not block us: the S1 rasters are
only input *imagery*, which OlmoEarth pretraining supplies independently — we only need the
labels + georeferencing, both of which are in the repo.

Files used (downloaded to `raw/sentinel_1_lake_ice_detection/`):
- `data/gt/{2016_17,2017_18}/{sihl,sils,silvaplana,stmoritz}.txt` — per-lake, **per-day
  whole-lake state** from daily webcam observation (semi-automated GT). Each day the whole
  lake gets one code. Only the three unambiguous ("clean") codes are used:
  - `s` = snow-on-ice (~90–100% frozen) → **frozen (0)**
  - `i` = ice (~90–100% frozen) → **frozen (0)**
  - `w` = water (~90–100% open) → **non-frozen (1)**
  - `ms`/`mi`/`mw` (60–90% partial), `c` (cloud/fog), `u` (unclear), `n` (no data), and any
    composite code → **excluded** (ambiguous).
  The whole-lake state is propagated to every pixel inside the lake polygon — exactly how the
  paper builds its per-pixel ground truth.
- `data/shapes/UTM32N.shp` — lake-outline polygons (EPSG:32632). The four labelled lakes:
  `stmoritz`→"Lej da San Murezzan" (0.75 km²), `silvaplana`→"Lej da Silvaplauna" (2.66 km²),
  `sils`→"Lej da Segl" (4.09 km²), `sihl`→"Sihlsee" (10.49 km²). (Other polygons in the
  shapefile — Lac de Joux, Greifensee, etc. — have no gt and are ignored.)

## Label / class mapping

| id | name | meaning |
|----|------|---------|
| 0 | frozen (ice) | lake frozen ~90–100% (bare ice or snow-on-ice) |
| 1 | non-frozen (water) | open water ~90–100% |
| 255 | nodata/ignore | pixels outside the lake polygon |

Each sample is a ≤64×64, 10 m, UTM (EPSG:32632) tile covering part of one lake on one
clean-state day: inside-polygon pixels carry that day's class, outside pixels are 255. There
is no explicit background class (only the lake surface is labelled) — per spec §5, negatives
are supplied downstream at assembly time; we do **not** fabricate them.

## Tiling and sampling

Each lake is covered by a fixed grid of 64×64 tiles over its polygon bbox; only tiles with
≥5% lake coverage are kept (tiles-per-lake: sihl 40, sils 17, silvaplana 11, stmoritz 4).
A selected lake-day contributes **all** its kept tiles.

To keep the four lakes balanced (spec §5, tiles-per-class balanced, ≤1000/class): allocate
~250 samples per (lake, class), so `n_days = ceil(250 / n_tiles_for_lake)` clean days are
drawn per lake, **evenly spaced** across that lake's sorted clean-day list. A final
`balance_by_class(per_class=1000)` caps each class at 1000 while preserving cross-lake /
cross-date spread.

Clean-day availability (both winters merged): frozen — sihl 28, sils 97, silvaplana 101,
stmoritz 129; water — sihl 285, sils 197, silvaplana 220, stmoritz 229.

Resulting per-lake / per-class counts (total 2000):

| lake | frozen | water |
|------|-------:|------:|
| sihl | 264 | 272 |
| sils | 243 | 246 |
| silvaplana | 248 | 243 |
| stmoritz | 245 | 239 |

## Time range and change handling

Lake-ice presence is a specific-date / seasonal **STATE**, so each sample gets a **tight
1-day** window `[obs_day 00:00 UTC, obs_day+1 00:00 UTC)` anchored on the webcam observation
date, with `change_time = null` (a per-date state, not a dated change event). The webcam
observation date is used as the source acquisition date because the exact per-scene S1
acquisition timestamps only existed in the now-unavailable polybox raster share. The frozen /
open state persists across neighbouring days, so a 1-day anchor is temporally coherent;
downstream assembly pairs the label with whatever S1 scene falls in the window.

## Judgment calls / caveats

- **S1 rasters not needed / unavailable.** The authors' polybox share (the only S1-raster
  source) returns 404. Labels are fully reconstructable from the repo, and pretraining
  supplies S1 imagery, so the dataset is processed normally (not a temporary_failure).
- **Whole-lake state → per-pixel.** Labels are per-day whole-lake observations, so within a
  given lake-day every lake pixel has the same class; spatial variety comes from the shoreline
  (polygon boundary within edge tiles) and across lakes/dates. This matches the source's own
  GT construction.
- **Ambiguous/partial days dropped** (ms/mi/mw/c/u/n and composites) to avoid mixed-state
  labels — only clearly-frozen and clearly-open days are kept.
- **Observation date vs S1 pass.** A 1-day window may not always contain an S1 acquisition;
  such samples are simply unused downstream. This is the tightest honest window given no
  recoverable per-scene timestamps.
- **Spatial sanity check:** tile centres for all four lakes reproject to within ~1–2 km of the
  known lake locations (Sils, Silvaplana, St. Moritz in the Engadine; Sihlsee near Zurich).
- **sensors_relevant:** `["sentinel1"]` — the dataset is designed for SAR (winter Alpine
  optical is frequently cloud/snow obscured).

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.sentinel_1_lake_ice_detection --workers 64
```
Idempotent: existing `locations/{id}.tif` are skipped. Downloads gt+shapefiles (<200 KB) from
the GitHub raw endpoint into `raw/sentinel_1_lake_ice_detection/`.
