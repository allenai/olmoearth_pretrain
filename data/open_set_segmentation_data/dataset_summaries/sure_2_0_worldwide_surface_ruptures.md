# SURE 2.0 (Worldwide Surface Ruptures)

- **Slug:** `sure_2_0_worldwide_surface_ruptures`
- **Status:** completed (accepted, filtered subset)
- **Task type:** classification (change / event segmentation, binary rupture-zone mask)
- **Family:** faults · **label_type:** lines (+ points, unused) · **region:** Global
- **Source:** Nurminen, F. et al. *SURE 2.0 – New release of the worldwide database of
  surface ruptures for fault displacement hazard analyses.* Sci Data 9, 729 (2022),
  https://doi.org/10.1038/s41597-022-01835-z
- **Data:** Zenodo record 7020265, https://doi.org/10.5281/zenodo.7020265 (CC-BY-4.0),
  no credentials.
- **Num samples:** 1312 tiles (64×64, UTM 10 m)

## What the source is

SURE 2.0 is a unified worldwide database of **coseismic surface-rupture traces** (per-event
line shapefiles, WGS84) and **slip observation points** for **50 crustal earthquakes,
1872–2019**, compiled by manual field mapping + georeferenced maps/satellite/LiDAR for fault
displacement hazard analysis. Each event ships as
`YYYYMMDD_EventName_SURE2.0_ruptures.shp` and a row in `SURE2.0_Earthquakes.xlsx`
(Year/Month/Day, Mw, focal mechanism), so **every rupture carries a day-precise earthquake
date**. Rupture attributes include a fault-ranking `Comp_rank` (1 = principal fault, 1.5/2/3/
21/22 = distributed rupturing). Total: 75,695 traces / 18,885 slip points.

## Accept / reject reasoning

Surface ruptures are produced by a **dated earthquake**, so a rupture trace is a genuine,
date-resolvable **change** signal (before→after surface break / scarp / deformation belt in
imagery) — but only for earthquakes in the Sentinel era, and only where the rupture zone is
observable at 10 m. Applying spec §2/§5/§8:

**Timing (pre-2016 rule + change-timing rule).** Of the 50 events, **42 are pre-2016** (32
in the 1900s; oldest 1872 Owens Valley, 1887 Sonora). They fail the pre-2016 change rule and
their surface expression is decades-eroded / re-vegetated / built-over — **dropped**. The 8
events from **2016+** have day-precise dates (≪ the ~1–2 month change-timing requirement), so
`change_time` = the earthquake date and `time_range` = ±180 d centered on it — a clean change
signal.

**Observability at 10 m.** Many surface ruptures are meter-scale offsets (sub-pixel at 10 m),
but **large earthquakes** produce wide deformation zones / continuous scarps visible at 10 m.
Of the 8 post-2016 events I additionally **drop 2019 Le Teil (Mw 4.9)** — ~cm offsets over a
~5 km rupture detected mainly by InSAR/field, effectively sub-pixel — via a `MIN_MW = 5.5`
filter. The **7 kept events are all Mw ≥ 6.0**, significant earthquakes whose rupture zones
(surface breaks, fault scarps, wide belts, e.g. the well-documented Ridgecrest breaks and the
Monte Vettore Norcia scarp) are plausibly observable at 10 m once the trace is buffered to a
zone.

**Georeferencing.** Vector lines in WGS84 lon/lat (verified: rupture-mask pixels land 18–29 m
median, ≤45 m max, from the source traces — consistent with the 30 m buffer). Accept.

## Kept events (Mw ≥ 6.0, ≥ 2016)

| Event (IdE) | Date | Mw | Mechanism | Region | traces | tiles |
|---|---|---|---|---|---|---|
| 20160415 Kumamoto | 2016-04-15 | 7.0 | strike-slip | Japan | 1145 | 446 |
| 20160520 Petermann | 2016-05-20 | 6.1 | reverse | Australia | 229 | 38 |
| 20160824 Amatrice | 2016-08-24 | 6.0 | normal | Italy | 120 | 17 |
| 20161030 Norcia | 2016-10-30 | 6.5 | normal | Italy | 732 | 202 |
| 20161201 Parina | 2016-12-01 | 6.2 | normal | Peru | 21 | 23 |
| 20190704 Ridgecrest 1 | 2019-07-04 | 6.4 | strike-slip | USA | 7074 | 191 |
| 20190705 Ridgecrest 2 | 2019-07-05/06 | 7.1 | strike-slip | USA | 10875 | 395 |
| **Total** | | | | | | **1312** |

**Dropped:** 42 pre-2016 events (pre-2016 rule + eroded expression); 2019 Le Teil Mw 4.9
(observability). Slip observation points (`points`) are not used — the buffered-line raster is
the preferred rupture representation (spec §4 lines).

## Label mapping (2-class, unified)

- `0 background` — no mapped rupture; genuine non-rupture context within the tile (the
  mapped footprint is authoritative, so off-trace pixels are background, not ignore).
- `1 surface_rupture` — rupture trace (principal **and all distributed ranks merged**)
  buffered to a **~30 m half-width (3 px @ 10 m → ~60 m wide zone)** so the surface break /
  scarp / deformation belt is resolvable at 10 m.

Both classes appear in all 1312 tiles. `nodata = 255` (unused). All tif values ∈ {0, 1}.

## Tiling / time handling

- Per event: reproject traces to the event's **local UTM zone** (from the centroid), convert
  to 10 m pixel space `(E/10, −N/10)`, buffer each line by 3 px, and grid the buffered
  footprint's pixel bbox into **non-overlapping 64×64 tiles**; keep tiles intersecting a
  buffered rupture. One kept tile = one sample.
- `change_time` = earthquake date (12:00 UTC); `time_range` = **±180 d centered** (360-day
  window, ≤ 1 year). Verified: all windows span exactly 360 d, all `change_time` ≥ 2016.

## Verification (§9)

- 1312 `.tif` + 1312 `.json`. Sampled tiles: single band, `uint8`, UTM (EPSG:326xx) at 10 m,
  64×64, nodata 255, values {0, 1}. Dataset-wide unique values = {0, 1}.
- Every `.json` has a 360-day `time_range` and a post-2016 `change_time`.
- Spatial sanity: rupture pixels lie 18–29 m median (≤45 m) from the source traces →
  georeferencing correct.
- Idempotent: re-running skips existing `locations/{id}.tif`.

## Caveats

- Distributed rupturing can be locally under-mapped, and the smallest kept events (Mw ~6,
  e.g. Amatrice/Parina) have narrow zones near the 10 m resolution limit — the 30 m buffer
  mitigates but the model should treat thin single-event ruptures as weak signal.
- Strike-slip ruptures (Kumamoto, Ridgecrest) dominate tile counts; normal/reverse events
  (Italy, Peru, Australia) are rarer but retained for kinematic diversity.

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.sure_2_0_worldwide_surface_ruptures
```
Downloads `SURE2.0_Ruptures.zip` + `SURE2.0_Earthquakes.xlsx` from Zenodo 7020265 to
`raw/sure_2_0_worldwide_surface_ruptures/`, then writes tiles to
`datasets/sure_2_0_worldwide_surface_ruptures/locations/`. Tunables in the script:
`BUF_PX` (buffer half-width, px), `MIN_MW` (5.5), `MIN_YEAR` (2016), `TILE` (64),
`HALF_WINDOW` (±180 d).
