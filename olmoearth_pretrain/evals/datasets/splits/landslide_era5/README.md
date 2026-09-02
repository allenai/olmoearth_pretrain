# Landslide ERA5 Eval Split

A leakage-safe train/val split of landslide events, built to probe whether the
ERA5 (weather) encoder can separate landslide-positive from negative windows.
Each event has a **positive** window (at the event date) and a **negative**
twin (same lat/lon, one year earlier) — so the only thing that differs is the
interannual weather forcing.

**Pipeline:** `dedup_windows.py` (build the leakage-safe universe) →
`eval_tagging.py` (filter to weather triggers, assign train/val, copy metadata).
Tagged dataset: `/weka/dfive-default/olmoearth/eval_datasets/landslide_era5/`
(`windows/<group>/<name>/metadata.json`, with `options.split` = train/val).

## Sources kept

| Source | Kept? | Why |
|---|---|---|
| `sen12_landslides` | yes | Global, real event dates, explicit triggers. |
| `glc` (Global Landslide Catalog) | yes | Real precise dates + explicit weather triggers; globally scattered. Its lack of polygons (Piper's reason to down-rate it) is irrelevant for point-based ERA5. |
| `icimod` (Nepal) | no | Post-Gorkha-earthquake inventory; `event_date`s are imagery/mapping dates, not failure dates → unusable for temporal alignment, and triggers are seismic. |
| `fwn_mtli` | no | All dates are a placeholder July-1 → cannot align to triggering weather. |

## Event types kept

Only **weather-triggered** events are retained (both source vocabularies):
`Rainfall`, `Hurricane`, `Cyclone`, `Tropical Storm`, `rain`, `continuous_rain`,
`downpour`, `tropical_cyclone`, `monsoon`, `flooding`, `freeze_thaw`,
`snowfall_snowmelt`.

`Earthquake` and unlabeled (`unknown`/`nan`, e.g. Kyrgyzstan) windows are
**dropped**: a seismic trigger has no systematic interannual weather signature,
so its positive/negative contrast is label noise (or worse, a spurious confound)
for a weather encoder.

## Deduplication (why it was necessary)

ERA5-Land is coarse (0.1° ≈ 11 km), far larger than a 640 m window. Many nearby
landslides — especially co-triggered clusters from one storm — fall in the
**same ERA5 cell at the same time and share an identical ERA5 input**. Without
dedup, neighbors split across train/val would leak.

Fix: group all windows by **(nearest ERA5-Land cell, ISO year-week of each
window's own time range)** and keep one window per bucket (one positive if any,
else one negative). This collapsed **151,104 → 6,549** windows. The ERA5 cell id
is the nearest 0.1° node (`round(lat/0.1)`, `round(lon/0.1)`), matching the
`ERA5LandDailyUTCv1` data-source registration (not the half-cell-offset
`build_candidate_grid` scheme).

## Train/val split (why ERA5-cell level)

The split unit is the **ERA5 cell**, not the individual window or pos/neg pair,
because (a) a positive and its negative twin share a cell and must stay together,
and (b) ~half the cells contain multiple distinct landslide locations that share
the same coarse ERA5 input. Assigning whole cells guarantees no two windows with
the same ERA5 forcing straddle the split. Cells are assigned greedily to mirror
the overall distribution across **event_type, event_year, and coarse geography**,
targeting a ~29.4% val fraction (≈ 300/125 pairs); curated-val windows are seeded
into val first.

## ERA5 fetch — window time semantics

Each window's `time_range` is **anchored at the event day as its START** and
runs ~60 days *forward* (it was built to find a post-event Sentinel-2 image):
positives = `[event, event+60d]`, negatives = the same shifted one year earlier
`[event−1yr, event−1yr+60d]`. So the event day is `time_range_start`, **not** the
center or end.

For ERA5 we want the daily sequence to **end exactly on the event day** and look
backward (antecedent precipitation / soil moisture). rslearn layer windows are
`[start+time_offset, start+time_offset+duration]`, so we set
`time_offset = -448d`, `duration = 448d` → `[event−448d, event]`, ending on the
event (negatives auto-align to the −1yr anniversary). This matches the burnrisk
448-day ERA5 sequence length but, unlike burnrisk's `-384d/448d` (which ends at
`+64d`), is cut at the event so no post-failure weather leaks in. Layer defined
in the dataset `config.json` as `era5d_448d_to_event` (`ERA5LandDailyUTCv1`, 14
bands, `spatial_size [1,1]`).

## Final breakdown — 852 windows (426 pos / 426 neg), 413 ERA5 cells

| | train | val | total |
|---|---|---|---|
| **positive** | 301 | 125 | 426 |
| **negative** | 300 | 126 | 426 |
| **total** | **601** | **251 (29.5%)** | **852** |

| event_type | train | val | | event_year | train | val |
|---|---|---|---|---|---|---|
| Rainfall | 228 | 94 | | 2017 | 349 | 145 |
| rain | 129 | 58 | | 2018 | 98 | 38 |
| continuous_rain | 97 | 34 | | 2019 | 32 | 16 |
| downpour | 79 | 37 | | 2023 | 122 | 52 |
| Hurricane / Cyclone | 36 | 16 | | | | |
| tropical_cyclone / Tropical Storm | 20 | 8 | | **group** | **train** | **val** |
| monsoon / flooding | 6 | 4 | | glc | 329 | 137 |
| freeze_thaw / snowfall | 6 | 0 | | sen12_landslides | 272 | 114 |
