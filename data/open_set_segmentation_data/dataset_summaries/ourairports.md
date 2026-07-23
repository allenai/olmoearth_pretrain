# OurAirports

- **Slug:** `ourairports`
- **Source:** [OurAirports](https://ourairports.com/data/) — a community-maintained
  (crowdsourced), public-domain global database of ~85k airfields.
- **Family / region:** transportation / global
- **Label type:** points (sparse point segmentation)
- **Task type:** classification
- **License:** public domain
- **have_locally:** false

## Source & access

OurAirports publishes flat CSV dumps at `https://ourairports.com/data/`. We download two
files with `download.download_http` into
`raw/ourairports/` (label-only; no imagery pulled):

- `airports.csv` (~12.7 MB, 85,729 rows) — one row per airfield with `id`, `ident`,
  `type`, `name`, `latitude_deg`, `longitude_deg`, country/region, etc. **This file
  carries everything we need** (a lon/lat and a `type` class per record).
- `runways.csv` (~4 MB) — downloaded for provenance/completeness only; not required for
  typing (the `type` field already classifies each airfield).

All 85,729 rows have valid `latitude_deg`/`longitude_deg` (0 missing/invalid). Because
each label is a single 10 m pixel with a class, this is a **pure sparse-point dataset**:
per spec §2a we write ONE dataset-wide GeoJSON point table
(`datasets/ourairports/points.geojson`) rather than per-point GeoTIFFs.

## Class mapping

Class ids follow the manifest ordering. The OurAirports `type` field maps directly:

| id | class name    | source `type`   | raw count in source |
|----|---------------|-----------------|---------------------|
| 0  | large airport | `large_airport` | 1,175               |
| 1  | medium airport| `medium_airport`| 4,101               |
| 2  | small airport | `small_airport` | 42,670              |
| 3  | heliport      | `heliport`      | 23,116              |
| 4  | seaplane base | `seaplane_base` | 1,274               |

**Dropped (non-target):** `closed` (13,332) — decommissioned/abandoned sites, dropped
per task instructions; `balloonport` (61) — not one of the target classes.

Total target-class records with valid coordinates: **72,336**.

## Sampling & counts

`balance_by_class(records, "label", per_class=1000)` (default `total_cap=25000`). With 5
classes the per-class limit stays at 1,000, well under the 25k cap. Every class has ≥1,175
candidates, so all five reach the full 1,000:

- large airport: 1,000
- medium airport: 1,000
- small airport: 1,000
- heliport: 1,000
- seaplane base: 1,000
- **Total selected: 5,000**

No class was truncated by the cap, and no class needed to be dropped for sparsity (all
target classes are well-populated; spec §5 assembly-time filtering handles any downstream
minimums).

## Time range

Airports are **static features**, so per spec §5 (static labels → representative 1-year
window in the Sentinel era) every point is anchored to a single window
**[2020-01-01, 2021-01-01)** (`io.year_range(2020)`), which sits inside the manifest
`time_range` of 2016–2026. `change_time` is null (not a change/event dataset).

## Caveats / observability

- Airfield `type` in OurAirports is inferred largely from runway length and
  infrastructure. **Large and medium airports** (long paved runways, terminals/aprons) are
  clearly resolvable at 10–30 m. **Small airports** are often a single short paved/unpaved
  strip and sit near the resolution limit. **Heliports** (a single helipad) and **seaplane
  bases** (a marked water area plus a dock) are frequently **sub-resolution** at 10–30 m —
  the point marks the site but the facility itself may not be individually visible.
- Per spec §5 we keep every class regardless of observability; downstream assembly filters
  rare/too-small classes and supplies negatives (this is a positive/presence-style point
  set with no fabricated background class).
- Coordinate precision is community-sourced and generally good (spot-checked large
  airports — Lisbon LPPT, Marseille LFML, Chiang Rai VTCT, Bata FGBT — match their true
  runway locations exactly), but individual small/heliport points may be offset by tens of
  metres; the label is still a valid presence point for the airfield site.

## Verification

- `points.geojson`: FeatureCollection, `count=5000`, 5,000 features; all coordinates in
  valid lon/lat range; all `time_range`s exactly the 1-year 2020 window; label ids
  {0,1,2,3,4} exactly match `metadata.json` classes.
- `metadata.json`: `task_type=classification`, `nodata_value=255`, class map + per-class
  descriptions, `class_counts` = 1,000 each.
- Spatial sanity: selected `source_id` idents cross-referenced back to source names/coords
  — large-airport points land on the correct international airports; seaplane-base points
  land on the correct water bodies.

## Reproduce

```bash
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.ourairports
```

Idempotent: downloads skip if present; re-running regenerates `points.geojson` and
`metadata.json` deterministically (seeded `balance_by_class`).
