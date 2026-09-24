# UNOSAT Conflict Damage Assessments

- **Slug:** `unosat_conflict_damage_assessments`
- **Registry status:** `completed`
- **Task type:** classification (per-pixel damage severity)
- **Samples written:** 1185 GeoTIFF tiles (64×64, UTM, 10 m)
- **Source:** UNITAR / UNOSAT via the Humanitarian Data Exchange (HDX),
  https://data.humdata.org/organization/unosat — **open access, no credentials**.
- **Annotation method:** expert visual photo-interpretation of VHR satellite imagery.

## What the source is

UNOSAT publishes per-structure conflict-damage assessments across active conflict zones.
Each structure is a point/polygon with a **damage severity class** (Destroyed / Severe /
Moderate / Possible Damage) and one or more analysis **sensor dates**. Data are distributed
per event as SHP and/or File-Geodatabase zips on HDX.

### Packages used (curated, post-2016 conflict, per-structure geodata)

| Region | HDX package | Coverage | Analysis date |
|---|---|---|---|
| Gaza | `unosat-gaza-strip-comprehensive-damage-assessment-11-october-2025` | Whole Gaza Strip | 2025-10 |
| Ukraine | `mariupol-updated-building-damage-assessment-overview-map-livoberezhnyi-and-zhovtnevyi-dist` | Mariupol | 2022 |
| Ukraine | `sumy-rapid-damage-assessment-overview-map` | Sumy + Kharkiv | 2022 |
| Ukraine | `kremenchuk-damage-assessment-overview` | Kremenchuk | 2022 |
| Syria | `damage-assessement-of-hama-hama-governorate-syria` | **All Syria CDA-2016 cities** (Damascus, Daraa, Deir-ez-Zor, Hama, Homs, Idlib, Raqqa, Aleppo) — one zip bundles them | 2016 |

The Gaza package is a cumulative time series with 14 sensor columns (~1–2 months apart,
Nov 2023 → Oct 2025); the Ukraine/Syria packages are single- or few-date assessments.
The Sumy and Hama zips each bundle several cities/layers, so this small package set covers
Gaza + four Ukrainian cities + eight Syrian cities. **Iraq products were excluded** because
they are all pre-2016 (2014–2015), outside the Sentinel era.

## Class mapping

Manifest 4-class scheme, most-severe first. UNOSAT numeric domain `1..4` and equivalent text
labels are mapped to ids; the last **valid** code in the per-structure column series is used
(later columns are frequently `0`/placeholder). Codes 5 (No Visible Damage), 6 (Not Affected),
11 (Impact Crater), etc. are **not** building-damage classes and are dropped.

| id | name | UNOSAT code / text |
|---|---|---|
| 0 | destroyed | 1 / "Destroyed" |
| 1 | severely damaged | 2 / "Severe Damage" |
| 2 | moderately damaged | 3 / "Moderate Damage" |
| 3 | possibly damaged | 4 / "Possible Damage" |

- nodata / ignore value: **255**.

## Why classification, not a change label (change-timing decision)

UNOSAT comprehensive/cumulative assessments compare a post-event image to a **baseline that is
often 1–3 years earlier**, so *when within that span* a given structure was damaged is not
resolvable to ~1–2 months. A dated change label would therefore be misaligned with the paired
imagery (spec §5 change-timing rule → would be a rejection).

However, destroyed / heavily-damaged structures are a **persistent post-change state**: rubble
stays visible for years in these zones (no near-term reconstruction). Per spec §5 this is recast
as **presence/state classification**:
- `change_time = null`
- `time_range` = a **1-year window anchored forward** on the assessment year
  (`[Jul 1 Y, Jul 1 Y + 360 d]`), so paired imagery post-dates the damage and shows the
  persistent state.

Only assessments with a latest sensor date in **2016 or later** are kept (per-feature filter).

## Resolution handling (aggregation to damaged zones)

Individual buildings are ~1 pixel at 10 m, so per the manifest note ("aggregate to
heavily-damaged zones for 10–30 m") we do **not** emit 1×1 point labels. Instead:
1. All damaged structures are binned onto the local-UTM 10 m grid (UTM zone chosen per
   feature lon/lat: zones 32635/32636/32637 appear).
2. 64×64 tiles are cut over grid cells containing a **cluster** of damage
   (`MIN_DAMAGE_PER_TILE = 3` structures).
3. Each labeled pixel carries the **most severe** damage class of the structures in it;
   non-damage pixels are nodata (255).

This is a **positive-only** dataset (spec §5): no synthetic negatives are fabricated;
assembly supplies negatives from other datasets.

## Sampling / balancing

- `select_tiles_per_class(per_class=1000, total_cap=25000)` — tiles-per-class balanced,
  rarest class first.
- Structures read (post-2016): **285,217** — gaza 196,134 / ukraine 10,405 / syria 78,678.
  By class: destroyed 147,678 / severe 48,107 / moderate 67,132 / possible 22,300.
- Candidate tiles (≥3 structures): 2,239 → **selected 1,185 tiles**.
- Tiles per class: destroyed 1,077 / severe 1,048 / moderate 1,000 / possible 980.
- Tiles per region: gaza 784 / ukraine 222 / syria 179. (Gaza dominates the raw pool; the
  per-class balance keeps all severity classes near their 1000 target and the region mix
  spans all three conflict theatres.)

## Output layout

- `datasets/unosat_conflict_damage_assessments/metadata.json` — dataset metadata + class map.
- `datasets/unosat_conflict_damage_assessments/locations/{id}.tif` — 64×64 uint8 damage mask,
  single band, local UTM @ 10 m, nodata 255.
- `datasets/unosat_conflict_damage_assessments/locations/{id}.json` — crs, pixel_bounds,
  time_range (≤1 yr), `change_time=null`, source_id (`region:epsg/tcol_trow`), classes_present.

## Verification (spec §9)

- 1185 `.tif` / 1185 `.json`; every tif single-band `(1,64,64)` uint8, resolution `(10,-10)`,
  UTM CRS (32635/36/37); pixel values ⊆ {0,1,2,3,255}.
- All `time_range`s are 360 days; all `change_time` are null.
- `metadata.json` class ids {0,1,2,3} cover every non-nodata pixel value present.
- **Spatial sanity:** all 1185 tiles' pixel-center lon/lat round-trip into the correct
  conflict-zone bounding boxes (gaza 784/784, ukraine 222/222, syria 179/179), confirming the
  CRS / pixel-bounds / north-up sign conventions are exact. (A full Sentinel-2 image overlay
  was not rendered; georeferencing was validated by exact round-trip instead.)

## Caveats

- Moderate/Possible severity is a VHR-scale distinction and is only weakly resolvable at 10 m;
  the destroyed/severe classes (rubble, roof loss) are the most reliable at S2/Landsat scale.
- Per-structure geometries are collapsed to a single representative pixel; the tile mask is a
  damaged-area footprint, not exact building outlines.
- The Gaza time series (`Damage_Status_N` per ~1–2-month sensor date) *could* support proper
  dated change labels in a future revision; here we deliberately use the simpler, uniform
  persistent-state recast that also works for the single-date Ukraine/Syria products.

## Reproduce

```bash
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.unosat_conflict_damage_assessments
# --probe to read/report without writing; idempotent (skips existing tiles).
```
