# WRI Global Power Plant Database

- **Slug:** `wri_global_power_plant_database`
- **Status:** completed
- **Task type:** classification (sparse points, spec §2a)
- **Num samples:** 8,384 points (1x1) written to one `points.geojson`
- **Source:** WRI Global Power Plant Database **v1.3.0**, CC-BY-4.0.
  Portal: https://datasets.wri.org/datasets/global-power-plant-database

## What the source is

A curated, open global compilation of ~35k power plants in 167 countries. Each record is
one plant with an explicit `latitude`/`longitude` (WGS84), a `primary_fuel` (the label we
use), `capacity_mw`, and — for about half the records — a `commissioning_year`. The
v1.3.0 release is a single flat CSV (`global_power_plant_database.csv`).

## Access method (reproducible)

No credentials needed. The v1.3.0 CSV ships in a public S3 zip:

```
https://wri-dataportal-prod.s3.amazonaws.com/manual/global_power_plant_database_v_1_3.zip
```

The script downloads + extracts it to
`raw/wri_global_power_plant_database/` (see `SOURCE.txt` there). Reproduce with:

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.wri_global_power_plant_database
```

## Why a point (not a footprint)

This is a pure sparse-point classification dataset, so per §2a it is written as ONE
dataset-wide `points.geojson` (one `Point` feature per plant), NOT per-sample GeoTIFFs.
GPPD provides a single representative coordinate per plant whose precision varies
(`geolocation_source` ranges from exact plant coordinates to locality/centroid geocodes),
and plant footprints span orders of magnitude (a small gas peaker vs a multi-km solar or
hydro complex). Since we have neither a footprint polygon nor coordinate accuracy that
would justify fabricating one, a 1x1 10 m point marking plant presence at the reported
location is the honest representation. Pretraining projects these lon/lat onto the S2 grid.

## Class mapping (fuel type)

`primary_fuel` is mapped case-insensitively onto the manifest's 10 fuel classes
(id = manifest order):

| id | class | raw count | selected |
|----|-------|-----------|----------|
| 0 | coal | 2330 | 1000 |
| 1 | gas | 3998 | 1000 |
| 2 | oil | 2320 | 1000 |
| 3 | hydro | 7156 | 1000 |
| 4 | nuclear | 195 | 195 |
| 5 | solar | 10665 | 1000 |
| 6 | wind | 5344 | 1000 |
| 7 | biomass | 1430 | 1000 |
| 8 | geothermal | 189 | 189 |
| 9 | waste | 1068 | 1000 |

Balanced to ≤1000/class via `sampling.balance_by_class` (well under the 25k cap; effective
per-class limit stays 1000 since 10×1000 < 25000). `nuclear` (195) and `geothermal` (189)
are naturally sparse and kept in full — retained per §5 (downstream assembly drops
too-small classes if needed).

**Dropped:** 241 plants (0.7%) whose `primary_fuel` is outside the 10-class scheme —
`Storage` (135), `Other` (43), `Cogeneration` (41), `Petcoke` (12), `Wave and Tidal` (10).
These were dropped rather than force-mapped, to keep the class scheme aligned with the
manifest contract and avoid speculative fuel assignments.

## Time range

Power plants are persistent physical structures, so the fuel-type label is a static label
(§5). Each point gets a 1-year Sentinel-era window: default **2019**; if a
`commissioning_year` is known and later than 2019, the window is anchored on it (so we
never label a window before the plant was built), clamped to 2016–2021. Result: 8,364
points at 2019 and 20 points at 2020 (plants commissioned 2020). `change_time` is null
(not a change dataset). All windows are exactly 1 year (≤ 360-day cap verified).

## Verification (§9)

- `points.geojson`: `FeatureCollection`, `task_type=classification`, `count=8384` ==
  feature count. Labels are ids 0–9 (all valid uint8 class ids, ≤254 cap). No window
  exceeds 1 year. lon∈[-179.98, 179.39], lat∈[-45.75, 70.48] (plausible global spread).
- `metadata.json`: 10 classes with descriptions; `class_counts` and `raw_class_counts`
  recorded; `nodata_value=255`.
- **Spatial/provenance sanity:** labels and coordinates come directly from GPPD. Spot
  checks: Three Gorges Dam (30.823, 111.003) → hydro; Grand Coulee (47.958, -118.977) →
  hydro; Palo Verde (33.388, -112.862) → nuclear — all correct sites/fuels.
- Idempotent: re-running overwrites `points.geojson`/`metadata.json` atomically.

## Caveats

- Coordinate precision is heterogeneous (some plants geocoded to locality centroids); the
  point marks approximate plant location, adequate for 10 m point-segmentation labels.
- `capacity_mw` is carried as an auxiliary per-feature property (not used as a label).
- Generation columns cover 2013–2017; not used (we label fuel type, not generation).
