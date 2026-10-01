# GEM Global Active Faults Database

- **Slug**: `gem_global_active_faults_database`
- **Status**: completed
- **Task type**: classification (per-pixel, positive-only line mask)
- **Num samples**: 3,988 tiles (64×64, UTM, 10 m)
- **Classes**: `0 normal`, `1 reverse`, `2 strike-slip`, `3 oblique` (non-fault = nodata 255)

## Source

GEM Global Active Faults Database (GAF-DB) — Styron, R. & Pagani, M. (2020), "The GEM
Global Active Faults Database", *Earthquake Spectra* 36(1_suppl):160-180
(doi:10.1177/8755293020944182). A global, homogenized, expert-compiled database of
**active fault surface traces** (Quaternary-active faults) with kinematics and slip-rate
attributes. License **CC-BY-SA-4.0**.

- **Access**: public GitHub, no credentials. Downloaded
  `geojson/gem_active_faults_harmonized.geojson` (13,696 WGS84 LineStrings; the
  harmonized-attribute release) via raw.githubusercontent.com.
- **Key field**: `slip_type` (fault kinematics). Others (slip rates, dip, accuracy,
  epistemic/exposure quality) are mostly sparse/NULL.
- **Land mask**: Natural Earth 50 m physical land polygons (public domain, via cartopy),
  copied to `raw/` for provenance.

## Suitability judgment (the key decision)

Fault traces are geological **lineaments**, and per spec §4 (lines) a line dataset must be
rejected "if the feature is not observable at 10-30 m". Many faults are **not** surface
expressed at 10-30 m (blind/buried faults, subtle sub-pixel scarps), and a large fraction
of GEM's classes are **offshore/submarine** (oceanic spreading ridges, subduction-trench
traces, oceanic transforms). **Decision: ACCEPT with strict observability filtering and
prominent caveats** (spec option b), because a substantial subset of continental active
faults are mapped precisely *because* they have geomorphic surface expression — fault
scarps, offset ridges/drainages, range-front escarpments, fault-line valleys, sag ponds —
which is exactly the linear/geomorphic signal S2 / Landsat / S1 pretraining can learn.

### Observability filters applied
1. **Land mask** (strongest guard): every tile centre must fall on Natural Earth 50 m
   land. Removes offshore spreading ridges, subduction-trench traces and oceanic
   transforms directly. Of 874,909 grid cells crossed by mapped-slip-type faults, 667,945
   are on land.
2. **Slip-type drops**: `Spreading_Ridge` (1,853) and `Subduction_Thrust` (1,181) —
   plate-boundary features, overwhelmingly submarine, not mapped surface fault traces;
   `Blind Thrust` (1, blind by definition); `Anticline`/`Syncline` (302, folds not faults,
   outside the slip-type scheme); and `NULL` slip_type (319). Anything not in the class map
   below is dropped.

## Class mapping (GEM `slip_type` → class id)

| id | class | GEM slip_type values | segments (global, mapped) |
|----|-------|----------------------|---------------------------|
| 0 | normal | Normal | 2,716 |
| 1 | reverse | Reverse | 2,558 |
| 2 | strike-slip | Dextral, Sinistral, Strike-Slip, Dextral_Transform, Sinistral_Transform | 3,558 |
| 3 | oblique | all combined kinematics (Dextral-Reverse, Sinistral-Normal, Reverse-Strike-Slip, …) | 1,208 |

**The manifest "thrust" class is NOT represented.** Its only GEM members are
`Subduction_Thrust` (offshore, dropped by land mask + slip-type filter) and `Blind Thrust`
(blind, dropped). Compressional/thrust-sense faulting is captured under **reverse** (thrust
is kinematically a low-angle reverse fault). This is a deliberate, documented choice — per
spec §5 we keep every class we *can* and note the rest.

## Processing recipe

- **Rasterization** (spec §4 lines): each fault LineString is reprojected to the tile's
  local UTM 10 m pixel space and buffered by **2 px (~40-50 m wide zone)**, then burned
  (`all_touched`) with its class id; non-fault pixels = **255 (nodata)** — a positive-only
  mask (spec §5), no fabricated background/negatives.
- **Tiling / bounded sampling** (spec §5 global product): fault lines are densified
  (~320 m) and assigned to **every ~640 m latitude-aware grid cell they cross** (not just
  a bbox centre — faults are long, so bbox-centre assignment produced ~58% empty tiles;
  crossing-cell assignment + an STRtree per-tile line gather cut empties to 6/3,994).
  Each occupied on-land cell → one 64×64 local-UTM 10 m tile centred on the cell.
  Candidate cells are **tiles-per-class balanced** (rarest class first) to ≤1,000
  tiles/class, bounding the otherwise-global product. A tile counts toward every slip-type
  it contains; tiles with < 4 fault px are dropped.
- **Time** (spec §5, static labels): faults are persistent static features →
  `change_time = null`, a static **1-year window** per tile spread deterministically over
  **2016-2019** for imagery diversity.

## Sample counts

3,988 tiles. Per-class tile counts (a multi-fault tile counts for each class present):
`normal 1001, reverse 1001, strike-slip 997, oblique 1002`. Anchor years spread across
2016-2019. Well under the 25k per-dataset cap.

Geographic spread is global (lon -165.6..177.9, lat -54.5..68.0) with concentrations in the
world's major active-fault zones: Central Asia/Tibet (~712), Western US (~319),
Anatolia/Mediterranean (~243), Japan (~60), plus Andes, Tierra del Fuego (Magallanes-Fagnano
transform), etc.

## Verification (spec §9)

- 3,988 `.tif` ↔ 3,988 `.json`, all paired; single-band **uint8**, **64×64**, local UTM
  (varied zones e.g. 32616/32719/32750), 10 m, nodata 255; pixel values ∈ {0,1,2,3,255}.
- All `change_time = null`; `time_range` = 1 year; `metadata.json` class ids {0,1,2,3}
  cover every value in the tifs.
- **Spatial sanity**: tile centres verified on-land and located in known active-fault
  regions (e.g. a class-0 tile at (-116.75, 36.03) sits in the Death Valley fault zone,
  Basin & Range — a textbook active normal fault with strong scarp expression). A
  pixel-level Sentinel-2 overlay was not performed because exact alignment is not expected
  (see caveat) and the land-mask + geolocation checks already confirm placement.
- Re-running is idempotent (all 3,988 skipped on re-run).

## Caveats (important)

- **Positional accuracy is variable and often coarse.** Where the GEM `accuracy` field is
  recorded it holds source mapping scales as coarse as **1:100k-1:1M** (hundreds of metres
  to kilometres of positional uncertainty), and it is NULL for most traces. A rasterized
  line may therefore **not sit exactly on the imaged geomorphic feature**. The 2 px buffer
  partially absorbs this; downstream assembly filtering + the positive-only scheme mitigate
  residual noise, but labels are **noisier than field-mapped reference data**.
- **Partial observability.** Even after filtering, some retained continental faults have
  weak/no 10 m surface expression. This dataset supplies a fault/lineament *class signal*,
  not per-pixel-precise fault detection.
- **"thrust" class absent** by design (offshore/blind only) — see class mapping.

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.gem_global_active_faults_database
```
Idempotent (skips already-written tiles). `--probe` scans/reports without writing.
Outputs: `datasets/gem_global_active_faults_database/{metadata.json, locations/*.tif+*.json}`
on weka; raw source + land mask in `raw/gem_global_active_faults_database/`.
