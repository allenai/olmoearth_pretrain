# Smithsonian Global Volcanism Program (GVP)

- **Slug**: `smithsonian_global_volcanism_program`
- **Status**: completed
- **Task type**: classification (positive-only object **detection** encoding, multi-class by volcano type)
- **Num samples**: 3,049 (2,349 volcano-positive tiles + 700 background-only negative tiles)

## Source & access

Smithsonian Institution, Global Volcanism Program — *Volcanoes of the World* (VOTW)
database. Homepage: https://volcano.si.edu/ . License: free research use (cite GVP).

Accessed programmatically (no credentials) via the GVP OGC **WFS** GeoServer endpoint
`https://webservices.volcano.si.edu/geoserver/GVP-VOTW/wfs`, pulling two point layers as
GeoJSON into `raw/smithsonian_global_volcanism_program/`:
- `Smithsonian_VOTW_Holocene_Volcanoes` — 1,196 well-preserved Holocene edifices.
- `Smithsonian_VOTW_Pleistocene_Volcanoes` — 1,451 older Pleistocene edifices.

Each feature is **one POINT at the volcano's summit** with attributes `Primary_Volcano_Type`,
`Volcano_Number`, `Volcano_Name`, `Country`, `Elevation`, `Geological_Summary`, and (Holocene
only) `Last_Eruption_Year`.

## Triage / accept decision

**Accepted.** Volcano edifices are large landforms clearly discernible at 10–30 m from
S2/S1/Landsat (matching the manifest note "edifices discernible at 10-30 m"). The GVP record
is a **summit census point**, not a delineated edifice polygon, so — exactly like
dams-as-points / mines-as-points (see `goodd_global_georeferenced_dams`) — we treat the point
as a **presence detection** using the tunable detection encoding (spec §4).

## Label / class mapping

Multi-class detection by **`Primary_Volcano_Type`** ("Adds volcano-type classes" per the
manifest). Raw GVP type strings carry plural/parenthetical/uncertainty variants
("Shield" vs "Shield(s)", "Shield(pyroclastic)", "Stratovolcano?"); these are normalized
(strip any parenthetical group and `?`) and collapsed to canonical types. `Unknown`/`None`
types (82 records) are dropped. Result: **19 volcano-type classes**, ids assigned 1..19 by
descending global frequency; **id 0 = background**, **255 = nodata/ignore** (buffer rings).

Per-center class counts (positive tile centers; each tile also marks any neighbor volcano
falling inside it with its own type):

| id | type | tiles | id | type | tiles |
|----|------|-------|----|------|-------|
| 1 | Stratovolcano | 1000 (capped from 1216) | 11 | Maar | 22 |
| 2 | Shield | 345 | 12 | Tuff cone | 11 |
| 3 | Volcanic field | 313 | 13 | Tuya | 11 |
| 4 | Caldera | 151 | 14 | Lava cone | 9 |
| 5 | Pyroclastic cone | 146 | 15 | Explosion crater | 7 |
| 6 | Lava dome | 120 | 16 | Crater rows | 5 |
| 7 | Fissure vent | 75 | 17 | Volcanic remnant | 3 |
| 8 | Complex | 73 | 18 | Tuff ring | 2 |
| 9 | Cone | 28 | 19 | Pyroclastic shield | 1 |
| 10 | Compound | 27 | 0 | background (neg. tiles) | 700 |

Sparse classes (single-/few-sample) are kept per spec §5 (downstream assembly filters
too-small classes); no class exceeds the 254 uint8 cap.

## Encoding

Tunable detection (spec §4), following the GOODD dam template:
- **32×32 (320 m) UTM tile**, 10 m/px, single-band uint8.
- **1 px positive** at the summit carrying its volcano-type class id.
- **10 px nodata (255) buffer ring** around each positive — absorbs summit-point imprecision
  and the fact that the summit is a weak proxy for the whole edifice.
- **background (0)** fills the rest; other GVP volcanoes inside a tile are marked with their
  own type class.
- **700 background-only negative tiles** placed ≥2 km from any volcano (5–40 km offsets),
  so the class has spatially-meaningful negatives (spec §4).

## Time range & change handling

Volcanoes are **persistent landforms** → static labels → each sample gets a **1-year
Sentinel-era window** pseudo-randomly spread across **2016–2024** (`change_time=null`).

**No dated-eruption change label** (judgment call, spec §5): GVP eruption dates
(`Last_Eruption_Year`) are **year-resolved at best** and often historical/BCE. A change
label requires the event date known to ~1–2 months, which GVP does not provide, so eruptions
are **not** encoded as change events.

## Judgment calls / caveats

1. **Summit point vs edifice / type observability.** The summit is a weak label for the whole
   edifice, and `Primary_Volcano_Type` is a full-edifice morphological property that a 320 m
   summit tile only partly reveals. Type labels are therefore best-effort; treated as
   detection presence + best-effort type. Recorded, not blocking.
2. **No eruption change label** (see above) — eruption timing not resolvable to ~1–2 months.
3. **Pleistocene included** alongside Holocene (manifest says "Holocene/Pleistocene"); older
   Pleistocene edifices are more eroded but still large landforms.
4. **Type normalization** collapses plural/parenthetical/uncertain variants; `Unknown`/`None`
   dropped (82 records).
5. **Stratovolcano capped at 1000/class** (spec §5); 216 of 1,216 dropped. All other classes
   fully retained.

## Verification (spec §9)

- 3,049 `.tif` + 3,049 `.json`. Sampled tiles: single-band uint8, UTM (e.g. EPSG:32638/32719),
  10 m res, nodata 255, 32×32. Positive tiles show {0, class-id, 255}; negatives show {0}.
- All distinct non-nodata pixel values across the corpus = {0..19}, exactly matching
  `metadata.json` class ids.
- Every `.json` has a 1-year `time_range` and `change_time=null`.
- Georeferencing via validated `io.utm_projection_for_lonlat` / `lonlat_to_utm_pixel`.

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.smithsonian_global_volcanism_program
```
Idempotent: skips already-written `locations/{id}.tif`. Re-downloads the two WFS GeoJSON
layers into `raw/` only if missing.
