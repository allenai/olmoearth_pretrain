# WoSIS Soil Profiles — `wosis_soil_profiles`

**Status:** completed · **Task:** regression · **Samples:** 5000 · **Format:** point table (`points.json`, spec §2a)

## Source

ISRIC **WoSIS 2023 snapshot (December 2023)** — the "World Soil Information Service"
standardised compilation of legacy soil-profile point data (228k profiles, 174 countries).

- Download (open, no credential): `https://files.isric.org/public/wosis_snapshot/WoSIS_2023_December.zip` (~446 MB)
- DOI: https://doi.org/10.17027/isric-wdcsoils-20231130 · Paper: https://doi.org/10.5194/essd-16-4735-2024
- License: **CC-BY-4.0**
- Raw stored at `raw/wosis_soil_profiles/WoSIS_2023_December.zip` and extracted to
  `raw/wosis_soil_profiles/wosis_202312/WoSIS_2023_December/`.

The snapshot ships one TSV per standardised soil property. Each per-property TSV is a
layer-level (horizon) table that already carries the profile's lon/lat, sampling date,
positional uncertainty and standardised `value_avg` — no join to the profiles file needed.

## Regression target: topsoil pH in water (PHAQ / pH-H2O)

The manifest lists several candidate properties (organic carbon, pH, texture, CEC, bulk
density, classification). Per the task, one **primary continuous** property is emitted:
**pH-H2O**, chosen because it is the **most-populated** of the organic-carbon/pH
candidates — from `wosis_202312_observations.tsv`:

| property | code | profiles | layers |
|---|---|---|---|
| **pH in water** | **PHAQ** | **140,326** | 655,336 |
| Organic carbon | ORGC | 135,655 | 526,953 |
| Clay | CLAY | 153,319 | 652,347 |

(Clay has more profiles but is a texture fraction; the task specifically suggested organic
carbon or pH, and pH-H2O is the most-populated of those two.)

**Topsoil** = the shallowest sampled layer of each profile with `upper_depth < 30 cm`
(0–30 cm convention), one value per profile, using the ISRIC-standardised `value_avg`.

## Processing

1. Read `wosis_202312_phaq.tsv` (`profile_id, upper_depth, value_avg, longitude, latitude, positional_uncertainty, date`).
2. Drop rows missing value/coords; keep topsoil layers (`upper_depth < 30`); take the
   shallowest per profile → 137,452 profiles.
3. Quality filters: pH ∈ [2, 12] (drops 3 impossible outliers); keep only profiles located
   to ≲ 1 km (`positional_uncertainty` ∈ {"Circa 100 m", "100 m − 1 km"}); drop coarser
   "1 km − 10 km" / "Over 10 km" profiles (~18.5k) that can't be placed on a 10 m grid.
   → **118,921** profiles passed.
4. **Bucket-balance** across the pH range (`bucket_balance_regression`, 10 quantile
   buckets, seed 42) down to the **5000**-sample regression cap. pH is only moderately
   skewed but has long acidic/alkaline tails; bucketing gives even coverage of the full range.
5. Write `points.json` (spec §2a): `{id, lon, lat, label=<pH>, time_range, change_time=null, source_id=profile_<id>}`.

## Time range

WoSIS is **legacy in-situ** data: sampling dates median ≈ 1991, only ~100 profiles ≥ 2016,
~38% undated. Topsoil pH is a **quasi-static** property (SoilGrids and similar routinely
learn it from recent imagery over legacy profiles), so per spec §5 (static labels) every
point is anchored to a single **representative Sentinel-era 1-year window (2020-01-01 →
2021-01-01)** rather than its historical sample date. `change_time` is null.

## Value distribution (5000 selected samples)

pH range 3.0–10.8, mean 6.15. Histogram (1-unit bins):

| pH bin | 3–4 | 4–5 | 5–6 | 6–7 | 7–8 | 8–9 | 9–10 | 10–11 |
|---|---|---|---|---|---|---|---|---|
| count | 104 | 746 | 1470 | 1395 | 868 | 392 | 23 | 2 |

## Caveats

- **Legacy dates / static-year assumption** (see Time range) — the main caveat.
- **Positional uncertainty** ~100 m for most kept profiles (coarser than a 10 m pixel);
  treat points as approximately located. Coarser (km-scale) profiles were dropped.
- Point-only dataset → no per-sample GeoTIFFs (spec §2a); per-tif verification N/A.
  Coordinates spot-checked as plausible land locations (global spread, lon −164…166, lat −64…70).

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.wosis_soil_profiles
```

Idempotent: re-downloading skips the existing zip; the script recomputes `points.json`
deterministically (seed 42).

## Other properties

Emitting one dataset per property is possible (orgc, clay, silt, sand, cecph7, bulk
density, …) but per the task only the single most-populated property (pH-H2O) is produced
for now. The full snapshot remains in `raw/` for later per-property runs.
