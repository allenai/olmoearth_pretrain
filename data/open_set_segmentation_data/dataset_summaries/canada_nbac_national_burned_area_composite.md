# Canada NBAC (National Burned Area Composite)

- **Slug:** `canada_nbac_national_burned_area_composite`
- **Status:** completed
- **Task type:** classification (binary burned-area segmentation, `label_type: polygons`)
- **Num samples:** 25,000 label tiles (hard per-dataset cap)
- **Source:** Natural Resources Canada / Canadian Forest Service — National Burned Area
  Composite (NBAC). License: Open Government Licence – Canada (OGL-Canada).

## What the source is

NBAC is the authoritative annual "best-available" fire-perimeter polygon layer for all of
Canada. For each fire NBAC picks the best available mapping among agency perimeters,
satellite hotspot delineation, and Landsat/Sentinel-2 burned-area imagery. Distributed as
per-year shapefile ZIPs (no credentials) at
`https://cwfis.cfs.nrcan.gc.ca/downloads/nbac/NBAC_{YEAR}_{VERSION}.zip`
(landing page `https://cwfis.cfs.nrcan.gc.ca/datamart/download/nbac`). Version used:
`20260513`. We pulled only YEAR 2016–2025 (Sentinel era). Source CRS is Canada Lambert
Conformal Conic (NAD83).

Per-fire attributes: `YEAR`, `NFIREID`, `GID`, `POLY_HA`/`ADJ_HA` (burned area),
`FIRECAUS` (cause), `PRESCRIBED`, and several dates — `HS_SDATE`/`HS_EDATE` (satellite
hotspot fire start/end, day-resolved), `AG_SDATE`/`AG_EDATE` (agency fire start/end,
day-resolved), `CAPDATE` (burned-area mapping-image capture date).

## Class / label mapping

Binary segmentation, uint8, nodata 255 (unused):

- `0 = background` — outside the fire perimeter (genuine non-fire context; the NBAC
  perimeter authoritatively delimits burned extent, so no synthetic far negatives added).
- `1 = fire` — burned area inside an NBAC fire perimeter.

`FIRECAUS`, `POLY_HA`, and `PRESCRIBED` are per-fire attributes not observable per-pixel
from 10–30 m imagery (a burn scar looks the same regardless of cause), so they are kept
out of the class scheme (recorded as provenance context only). Prescribed burns (101 of
15,272 features, `PRESCRIBED='true'`) are retained — they are real burn scars.

## Time-range and change handling (spec §5)

A fire is a dated **change** event. `change_time` is set to the fire start date with
priority `HS_SDATE` (satellite hotspot start) > `AG_SDATE` (agency start) > `CAPDATE`
(same-year mapping capture), and is retained as the reference used to build the windows.
Instead of a single centered window, each sample emits two independent six-month windows:
a `pre_time_range` (the ≤183 days immediately **before** `change_time`) and a
`post_time_range` (the ≤183 days immediately **after** it), with `time_range` set to null.
The windows are adjacent and split exactly at `change_time` (built via
`io.pre_post_time_ranges(change_time, ...)`), so pretraining pairs a "before" image stack
with an "after" stack and probes on their difference. This meets the hard timing-precision
rule (event known to ≤ ~1–2 months): HS/AG start dates are exact day-resolved fire dates;
the small CAPDATE fallback (152 of 15,201 kept fires, ≈1%) is a same-fire-year capture of
the scar, so the pre+post span still brackets the fire. Anchor-source breakdown of kept
fires: HS_SDATE 6,685, AG_SDATE 8,364, CAPDATE 152.

Filtering: only YEAR ≥ 2016 used (NBAC's 1972–2015 perimeters excluded). A fire is dropped
only if it has no parseable date, or its only dates fall outside `[YEAR-1, YEAR+1]`
(data-quality outliers) — 71 of 15,272 fires dropped, leaving 15,201.

## Tiling

Perimeters reprojected per-fire to a local UTM projection at 10 m/pixel (UTM zone chosen
from the perimeter centroid's lon/lat). A fire fitting in a 64×64 tile (640 m) → one
centered 64×64 tile; larger fires are gridded into non-overlapping 64×64 windows, keeping
windows intersecting the perimeter and sampling up to 40 per fire. Inside polygon → 1,
outside → 0. Selection is round-robin across fires (every fire contributes ≥1 tile before
big fires add more), capped at 25,000 tiles.

## Sample counts

- 15,201 fires kept; 165,875 candidate tiles generated; 25,000 selected (cap).
- Tile composition: 24,047 tiles contain both background and fire; 953 are fire-only
  (fully inside large perimeters).
- Samples per fire year: 2016:1576, 2017:2631, 2018:2843, 2019:1444, 2020:1000,
  2021:2978, 2022:2414, 2023:3950, 2024:3117, 2025:3047.

## Verification (spec §9)

- 25,000 `.tif` each with a matching `.json`. Sampled tiles: single-band uint8, 64×64,
  local UTM CRS at 10 m, pixel values ⊆ {0, 1}, nodata 255, `time_range` null with
  adjacent `pre_time_range` / `post_time_range` (each ≤183 days) and `change_time` set.
- `metadata.json` classes {0 background, 1 fire} cover all values appearing in tiles
  (union over 200 tiles = {0, 1}).
- Geographic sanity: round-trip of tile CRS/bounds → WGS84 lon/lat for 400 random samples
  all land inside Canada's bounding box (boreal fire regions across UTM zones 8–20).
  (A bug was caught and fixed during development: the source-CRS `Projection` resolution
  must be `(1, 1)`, not `(1, -1)` — the `-1` negated northing and threw locations far
  south; corrected before the final run.)
- Idempotent: existing `locations/{id}.tif` are skipped on re-run.
- Note: a Sentinel-2 pixel overlay was not run (needs imagery infra); correctness rests on
  exact polygon rasterization plus the verified reprojection/round-trip to Canadian fire
  regions.

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.canada_nbac_national_burned_area_composite --workers 64
```

Raw ZIPs + extracted shapefiles: `raw/canada_nbac_national_burned_area_composite/`
(≈530 MB for 2016–2025). Outputs:
`datasets/canada_nbac_national_burned_area_composite/{metadata.json, locations/}`.
