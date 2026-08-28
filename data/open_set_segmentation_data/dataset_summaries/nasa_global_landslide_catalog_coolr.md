# NASA Global Landslide Catalog / COOLR

- **Slug**: `nasa_global_landslide_catalog_coolr`
- **Status**: completed
- **Task type**: classification (sparse points, spec §2a → one `points.geojson`)
- **num_samples**: 1169
- **Source**: NASA GSFC — Global Landslide Catalog (GLC), part of COOLR (Cooperative
  Open Online Landslide Repository). https://landslides.nasa.gov — public domain.

## Source & access

Downloaded the "Global Landslide Catalog Export" CSV from the NASA Open Data Portal
(the portal migrated Socrata → CKAN; the old `data.nasa.gov/resource/*.json` API is
gone). Direct label-only CSV, no credentials:

```
https://data.nasa.gov/docs/legacy/Global_Landslide_Catalog_Export/Global_Landslide_Catalog_Export_rows.csv
```

11,033 rows, one per documented landslide event, with `longitude`/`latitude`,
`event_date` (date+time, e.g. `05/19/2017 08:14:00 PM`), `location_accuracy`,
`landslide_category`, trigger/size/fatality/country fields, etc. Cached to
`raw/nasa_global_landslide_catalog_coolr/glc_export.csv` (+ `SOURCE.txt`). The GLC
export ends 2017-09 (later reporting migrated to the citizen-science COOLR stream).

## Label mapping

Sparse-point **classification** by `landslide_category`. Canonical class ids are ordered
by full-catalog frequency and stable (cover every GLC category, incl. ones no filter
keeps, so ids never shift):

| id | class | id | class |
|----|-------|----|-------|
| 0 | landslide | 7 | riverbank_collapse |
| 1 | mudslide | 8 | snow_avalanche |
| 2 | rock_fall | 9 | translational_slide |
| 3 | complex | 10 | lahar |
| 4 | debris_flow | 11 | earth_flow |
| 5 | other | 12 | creep |
| 6 | unknown | 13 | topple |

Positive-only presence points (no "no-landslide" class); per spec §5 no synthetic
negatives are fabricated — assembly supplies negatives from other datasets.

## Change / timing handling

Each landslide is a dated **event** → CHANGE label. `event_date` is precise to the day
for every row (far tighter than the ~1–2 month hard requirement), so `change_time` = the
event date, retained as the reference used to build the windows. Instead of a single
centered window, each sample emits two independent six-month windows: a `pre_time_range`
(the ≤183 days immediately **before** `change_time`) and a `post_time_range` (the ≤183
days immediately **after** it), with `time_range` set to null. The windows are adjacent and
split exactly at `change_time` (built via `io.pre_post_time_ranges(change_time, ...)`), so
pretraining pairs a "before" image stack with an "after" stack — seeing the slope before
and after the failure — and probes on their difference. Verified: all 1169 features have
change_time set with adjacent pre/post windows (each ≤183 days).

## Filters applied (and counts dropped)

Starting from 11,033 rows (all have valid coordinates):

- **Location accuracy (hard)**: kept only `exact` and `1km`. Coarser codes
  (`5km`/`10km`/`25km`/`50km`/`100km`/`250km`/`unknown`) place the point tens–hundreds of
  10 m pixels from the actual failure and are unusable on the S2 grid → **7,462 rows
  dropped**. (Full accuracy histogram: 5km 3178, 1km 2185, 25km 1470, 10km 1435, exact
  1386, 50km 794, unknown 542, 100km 25, 250km 16.)
- **Sentinel era (hard)**: kept only year ≥ 2016 → **8,595 pre-2016 rows dropped**.
- No date / no coords: 0 dropped (all rows populated).

**Net kept: 1,169** (2016: 454, 2017: 715; accuracy: exact 424, 1km 745). All classes
are below the 1000/class cap, so `balance_by_class` keeps every point.

Selected class counts: landslide 661, mudslide 255, rock_fall 194, debris_flow 21,
other 11, unknown 6, complex 5, snow_avalanche 5, riverbank_collapse 4,
translational_slide 4, earth_flow 2, topple 1. Sparse classes retained per spec §5
(downstream assembly drops too-small ones).

## Verification

- `points.geojson`: FeatureCollection, task=classification, count=1169, all `Point`
  geometries; label ids in 0–13 (all valid, covered by `metadata.json` classes); lon/lat
  span the globe (−179.7…178.5, −44.7…71.5).
- All samples have `time_range` null with adjacent pre/post windows (each ≤183 days) split
  at change_time (0 bad windows / 1169).
- Spatial sanity: these are the catalog's own event geocoordinates (point events, not
  pixel masks); spot-checked points land on plausible terrain (e.g. id 000000 at
  −4.857,56.227 in the Scottish Highlands). No per-pixel S2 overlay is meaningful for a
  point-presence catalog.

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.nasa_global_landslide_catalog_coolr
```

Idempotent: the raw CSV is cached (download skipped if present) and
`points.geojson`/`metadata.json` are rewritten atomically each run.
