# RADD Forest Disturbance Alerts

- **Slug**: `radd_forest_disturbance_alerts`
- **Task type**: classification (dense_raster, **dated CHANGE** dataset)
- **Family / region**: deforestation / pan-tropical (South America, Congo Basin/Africa, insular SE Asia)
- **Source**: Wageningen University (WUR) RADD (RAdar for Detecting Deforestation) alerts,
  contributed to WRI Global Forest Watch. Accessed via Google Earth Engine collection
  `projects/radar-wur/raddalert/v1`.
- **URL**: https://data.globalforestwatch.org/datasets/gfw::deforestation-alerts-radd/about
- **License**: CC-BY-4.0
- **Num samples**: **2910** (2299 disturbance tiles + 611 stable-forest background tiles)

## What the source is

RADD provides near-real-time forest-disturbance alerts for the humid tropics at **10 m**,
derived from cloud-penetrating **Sentinel-1** C-band radar. Each geography's latest
(cumulative) alert image has two bands:

- `Alert` — `2` = unconfirmed (low confidence), `3` = confirmed (high confidence).
- `Date` — date of first detected disturbance, encoded **YYDOY**:
  `year = 2000 + (value // 1000)`, `day-of-year = value % 1000`.
  e.g. `24184 → 2024, DOY 184 (2024-07-02)`; `22230 → 2022, DOY 230 (2022-08-18)`.
  This is **day-precise**, well within the spec's ~1–2 month change-timing requirement.

A per-geography `forest_baseline` image (band `constant` = 1 over the primary-forest baseline
extent, masked elsewhere; baseline year 2019 for `sa`/`asia`, 2018 for `africa`) delimits the
valid forest area.

Three RADD geographies are used: `sa` (Amazon / South America), `africa` (Congo Basin /
Africa), `asia` (insular Southeast Asia).

## Access method

Earth Engine service-account credentials from `.env`
(`TEST_GEE_SERVICE_ACCOUNT_*`; spec §8 authorizes `.env` creds — the GFW data-lake S3 mirror
`gfw-data-lake` is requester-pays and unusable anonymously). **No bulk download** of the
pan-tropical tiles: candidate tile centers are sampled with EE `stratifiedSample` over a grid
of 2° cells across each geography (scale 30 m for fast discovery), then each ≤64×64 label
patch is fetched directly, **reprojected to the local UTM zone at 10 m (nearest-neighbour)**,
via `ee.data.computePixels`. Candidate points are cached to `raw/{slug}/candidates_{region}.json`
so re-runs skip the sampling phase; existing `locations/{id}.tif` are skipped (idempotent).

## Label scheme (uint8, single band, local UTM 10 m, ≤64×64)

| id  | name                 | meaning |
|-----|----------------------|---------|
| 0   | `stable_forest`      | forest-baseline pixel with no alert (background/negative) |
| 1   | `forest_disturbance` | confirmed alert (`Alert==3`) whose decoded date lies within the tile's event window |
| 255 | nodata / ignore      | outside forest baseline; low-confidence alerts (`Alert==2`); confirmed alerts of a *different* date than this tile's event |

Only **confirmed** (high-confidence) alerts define the positive class; low-confidence alerts
are ignored (255) to keep the change mask clean.

## Change handling (spec §5)

This is a genuine dated CHANGE dataset (forest → disturbed). Each **disturbance** tile is
built to represent a single **temporally-coherent event**:

- A candidate center is a sampled confirmed-alert pixel with seed date `D`.
- Pixels labeled `1` are confirmed alerts whose decoded YYDOY date lies within
  `D ± 45 days` (a 90-day event window). Confirmed alerts of any *other* date in the tile are
  set to nodata (255) so a different-dated disturbance is neither counted as change nor as
  stable background.
- `change_time` = **median decoded date of the in-window disturbed pixels** (day-precise,
  representative central date of the event).
- `time_range` = **change_time ± 180 days** (360-day window, ≤ 1 year), centered on
  `change_time`, as required for change labels.
- A disturbance tile is kept only if it has **≥ 20** in-window confirmed pixels (~0.2 ha) so
  the mask carries real signal (715 of 3014 selected candidates were dropped for isolated /
  temporally-incoherent centers).

**Background negatives**: stable-forest tiles (≤ 5 confirmed alert px, ≥ 20 forest px) with
`change_time = null` and a static representative 1-year window (`year_range(2022)`). These
give the change class spatial negatives from genuine forest; downstream assembly also adds
cross-dataset negatives (spec §5). 289 of 900 selected background candidates were dropped for
containing too many alerts / too little forest.

**Post-2016 rule**: RADD begins ~2018–2019, so every alert is inside the Sentinel era; no
pre-2016 filtering is needed. Realized `change_time` years span **2019–2026**.

## Counts

- Disturbance tiles per region: `sa` 768, `africa` 867, `asia` 664.
- Disturbance tiles per `change_time` year: 2019:123, 2020:358, 2021:329, 2022:314, 2023:309,
  2024:307, 2025:283, 2026:276.
- Background (stable-forest) tiles: 611.
- Tiles containing class `stable_forest` (0): most disturbance tiles + all background tiles;
  class `forest_disturbance` (1): 2299.

Sampling is round-robin across `(region, year)` (disturbance) and per-region random
(background), deduplicated to a ~640 m grid so tiles don't heavily overlap. Well under the
25k per-dataset cap and the 254-class uint8 cap.

## Verification (spec §9)

- 2910 `.tif` each with a matching `.json`; all single-band **uint8**, **UTM** (EPSG:326xx)
  at **10 m**, **64×64**, nodata **255**; pixel values ⊆ {0, 1, 255}.
- All `time_range`s ≤ 366 days; `change_time` set for every disturbance tile (null only for
  background negatives).
- Spatial sanity: 200 sampled centroids all fall in the tropical band (lat −16.5…11.2) and
  split across Americas / Africa / Asia as expected.

## Caveats

- Positives are RADD's own Sentinel-1-derived detections (a derived product, not in-situ
  reference). Low-confidence alerts are deliberately ignored rather than labeled.
- The `Date` band records the *first* detection date; a slowly-progressing clearing may have
  pixels spread over the 90-day event window, but `change_time` (window median) places the
  event confidently within the ±180-day pairing window.
- Non-forest and other-dated disturbances inside a tile are nodata, so a tile is not a
  wall-to-wall land-cover map — it is a where-mask for one disturbance event vs. stable forest.

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.radd_forest_disturbance_alerts --workers 32
```
(from repo root `.`, with EE service-account creds available at
`/etc/credentials/gcp_credentials.json`).
