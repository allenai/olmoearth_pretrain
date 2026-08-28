# Earth Impact Database

- **Slug:** `earth_impact_database`
- **Status:** completed
- **Task type:** classification (single-class presence segmentation)
- **Num samples:** 149 label tiles (one per retained impact structure)
- **Family:** geomorphology
- **Source:** Earth Impact Database (EID), Planetary and Space Science Centre (PASSC),
  University of New Brunswick, Canada. http://www.passc.net/EarthImpactDatabase/
- **License:** "free scholarly use" (not-for-profit scientific resource). Attribution:
  *Earth Impact Database, Planetary and Space Science Centre, University of New Brunswick,
  Canada (managed by J. Spray).* In scope for this research use.

## Source & access

The EID is the definitive catalog of confirmed terrestrial impact structures (190
confirmed as of the 2018 web release). There is no bulk download; the catalog is served
as HTML. Access method:

1. Fetch the "sorted by Name" index
   (`New website_05-2018/Namesort.html`) and enumerate the per-structure page filenames
   (198 links).
2. Fetch each per-structure HTML page (cached under `raw/{slug}/pages/`) and parse its
   small data table: name, location, latitude, longitude (DMS), diameter (km), age (Ma).

197 of 198 pages parse cleanly; 1 page (`Riocuarto.html`) returns HTTP 404 and is skipped
(a small elongated crater group, well below the diameter cutoff regardless). Coordinates
are DMS to the arc-minute (a few to arc-second, e.g. Dhala); a DMS parser handles degree/
minute/second with N/S/E/W sign. Parsed catalog saved to `raw/{slug}/catalog.json`.

Georeferencing verified: tile centers round-trip exactly to catalog coordinates for
iconic structures (Chicxulub 21.333,-89.500; Manicouagan 51.383,-68.700; Vredefort
-27.000,27.500; Sudbury 46.600,-81.183).

## Class mapping

Single presence class (positive-only; no background/negative class — the assembly step
supplies negatives from other datasets, spec §5):

| id | name | definition |
|----|------|------------|
| 0  | impact_structure | interior of a confirmed EID impact structure, diameter ≥ 3 km |

Geological attributes (age, target rock, bolide type, exposed/drilled flags) are not
inferable from optical/SAR imagery at 10 m and are deliberately **not** used as classes.

## Key decisions

- **Diameter cutoff = 3 km** (149 of 197 parseable structures retained; range 3.0–160 km).
  The cutoff is derived from the encoding + coordinate precision: EID coordinates are
  arc-minute (worst-case ±0.5′ latitude ≈ 0.93 km). For a 64 px (640 m) label tile
  centered on the catalog point, the farthest tile pixel is 0.93 km (coord error) +
  0.45 km (tile half-diagonal, 320 m·√2) = 1.38 km from the true structure center.
  Requiring this ≤ structure radius gives diameter ≥ 2.77 km → rounded to a clean 3 km, so
  the whole footprint tile is **guaranteed to lie inside** the structure despite
  coordinate imprecision. Structures < 3 km are dropped (near the resolution limit and/or
  the tile cannot be guaranteed to fall inside).
- **Encoding = circular footprint GeoTIFF (not a 1×1 point).** Impact structures are
  roughly circular with a real footprint, so each structure is rasterized as a circle of
  radius = diameter/2 into a 64×64 UTM tile at 10 m centered on the point; interior = class
  0, outside-circle-within-tile = 255 (nodata). Because every retained structure is ≥ 3 km
  (≥ 150 px radius), the circle **fills the entire 640 m tile** — each output is a coherent
  640 m patch of confirmed impact-structure surface (far more labeled positive area than a
  single point). This is the spec §4 polygon/footprint recipe, positive-only.
- **Time = static.** Impact structures are persistent landforms (ages Ma–Ga); the
  formation event is not an observable Sentinel-era change. So **not a change dataset**:
  `change_time = null`, static 1-year window on 2020 (representative Sentinel era).
- **Caveat:** some retained structures are deeply eroded or partly buried (weak surface
  expression). They are kept as valid presence labels per spec §5; the diameter cutoff and
  guaranteed-inside geometry keep the labels well-placed. Analogous to the accepted
  `collapse_caldera_database_ccdb` presence dataset (which used a 1×1 point; here the
  larger, cleanly-filtered structures justify a footprint tile).

## Outputs

- `datasets/earth_impact_database/locations/{000000..000148}.tif` — single-band uint8,
  local UTM, 10 m, 64×64, all pixels class 0 (footprint fills tile), nodata 255.
- `datasets/earth_impact_database/locations/{id}.json` — per-sample CRS, pixel bounds,
  2020 1-year `time_range`, `change_time=null`, `source_id` (structure name), `classes_present=[0]`.
- `datasets/earth_impact_database/metadata.json` — dataset metadata (class, cutoff, diameter stats).

## Verification

- 149 `.tif` each matched by a `.json`; all single-band uint8, UTM EPSG:326xx/327xx at
  10 m, 64×64; global unique pixel value = {0}; all `time_range` are exactly 1 year with
  `change_time=null`; metadata class ids cover all tif values.
- Georeferencing round-trips exactly to catalog coordinates for known craters (above).

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.earth_impact_database
```
Idempotent: cached raw HTML pages and existing `locations/{id}.tif` are skipped on re-run.
