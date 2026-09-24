# GRIP4 Global Roads Inventory

- **Slug:** `grip4_global_roads_inventory`
- **Status:** completed
- **Task type:** classification (per-pixel road-type line masks)
- **Samples:** 2,949 tiles (64×64, local-UTM, 10 m/pixel)
- **Label type:** lines (spec §4 "lines" recipe)

## Source

GRIP4 (Global Roads Inventory Project, version 4) — Meijer, Huijbregts, Schotten &
Schipper (2018), "Global patterns of current and future road infrastructure",
*Environmental Research Letters* 13:064006 (doi 10.1088/1748-9326/aabd42). Compiled by
PBL Netherlands Environmental Assessment Agency by harmonizing ~60 national / regional /
global road sources (including OpenStreetMap) into a single global vector database of
~21.7 M km of roads (~21 M LineString segments), finalized in 2018.

- Download page: https://www.globio.info/download-grip-dataset
- Per-region vector shapefiles: `https://dataportaal.pbl.nl/downloads/GRIP4/GRIP4_Region{r}_vector_shp.zip`
- CRS: EPSG:4326 (WGS84 lon/lat degrees). License: CC0 (regional data ODbL where sourced from OSM).
- No credentials required (public HTTP download).

Key attribute `GP_RTP` (road type): 1 Highways, 2 Primary, 3 Secondary, 4 Tertiary,
5 Local, 0 Unspecified. Other fields used: `GP_REX` (existence; 4 = under construction).
`GP_RSY` (source year) is ≤ 2015 for all segments.

## Access / bounded sampling (spec §5)

GRIP4 is a **large global derived product**, so global coverage was **not** attempted.
A representative subset of GRIP4 **regions** was downloaded and processed, spanning
multiple continents, hemispheres, and development levels (dense↔sparse, paved↔unpaved):

| Region | Area | shp zip size | segments kept |
|---|---|---|---|
| 1 | North America | 909 MB | 5,360,526 |
| 3 | Africa | 242 MB | 1,532,578 |
| 5 | Middle East & Central Asia | 151 MB | 1,354,055 |
| 6 | South & East Asia | 711 MB | 5,244,510 |
| 7 | Oceania | 59 MB | 396,271 |

Total: 13,887,940 segments read. **Region 2 (Central & South America)** and **Region 4
(Europe)** were omitted to keep the download/processing bounded; the retained regions
already contain all 5 road types and span dense/sparse networks on both hemispheres.

Raw files: `raw/grip4_global_roads_inventory/` (region shapefiles + `SOURCE.txt`).

## Class mapping (id = GP_RTP − 1)

| id | name | GRIP4 GP_RTP | tiles containing class |
|---|---|---|---|
| 0 | highways | 1 | 1,076 |
| 1 | primary | 2 | 1,034 |
| 2 | secondary | 3 | 936 |
| 3 | tertiary | 4 | 926 |
| 4 | local | 5 | 993 |

Non-road pixels = **nodata (255)**. This is a **positive-only** mask (spec §5): no
background class or synthetic negatives are fabricated; a tile counts toward every road
type it contains, and the assembly step supplies negatives from other datasets. Segments
with `GP_RTP == 0` (unspecified) and `GP_REX == 4` (under construction) were dropped.
Full-source class distribution across the read regions: highways 303,107 · primary
457,125 · secondary 829,818 · tertiary 2,136,494 · local 10,161,396.

## Processing

- **Recipe:** rasterize road centerlines (spec §4 "lines"). Each segment is reprojected
  WGS84 → local UTM pixel space and buffered by ~1 px (`DILATE_RADIUS_PX=1.0`) → a
  ~20–30 m (2–3 px) wide mask, `all_touched=True`. Wider road types are drawn **last** so
  they win pixel conflicts (local→…→highways).
- **Tiling:** segments are partitioned onto a ~640 m **latitude-aware** geographic grid
  (`DLAT ≈ 0.00575°`, `dlon = DLAT/cos(lat)`). Each occupied cell → one 64×64 (640 m)
  local-UTM 10 m tile centered on the cell center; all segments assigned to the cell are
  rasterized (clipped to the tile). Tiles with < `MIN_ROAD_PIXELS`=3 road px are dropped.
- **Sampling (tiles-per-class balanced, spec §5):** per-cell class bitmasks are computed
  over all segments; a bounded candidate set (≤ 4,000 cells/class, seeded) is fed to
  `sampling.balance_tiles_by_class(per_class=1000, total_cap=25000)`, which fills rarest
  class first (highways). 3,297 cells selected → 2,949 written (348 dropped as too-small).
- **Time range (static labels, spec §5):** roads are persistent (source years ≤ 2015, so
  every mapped road exists in the Sentinel era). Each tile gets a static 1-year window
  (`change_time=null`) spread deterministically over 2016 / 2017 / 2018 (the manifest
  range) for imagery diversity: 981 / 978 / 990 tiles respectively.

## GeoTIFF spec

Single-band **uint8**, 64×64, local UTM at 10 m/pixel, north-up, nodata **255**. Verified
across sampled tiles: correct band count, dtype, size, per-zone UTM CRS, 10 m resolution,
and raster values ∈ {0,1,2,3,4,255}. Every `.tif` has a matching `.json` with a ≤1-year
`time_range` and `change_time=null`.

## Verification (spec §9)

- Format: 2,949 tif + 2,949 json; all single-band uint8 64×64 UTM 10 m, nodata 255,
  values within {0–4,255}.
- Geographic spread: tile centers span lon −158…178, lat −46…67 across all 5 sampled
  regions (S/E Asia 1,794 · N. America 469 · ME/C.Asia 290 · Africa 261 · Oceania 116;
  remainder near crude region boundaries).
- Spatial overlay: for a highway tile in New Zealand (lon 170.45, lat −45.90) a Sentinel-2
  RGB (2017, 0.9% cloud, from Planetary Computer) was fetched; the road mask visibly
  traces the road/track in the imagery, and mean S2 brightness on road pixels (493.5)
  exceeds off-road (404.8), consistent with roads.
- Idempotent: re-running skips all existing tiles (`{'skip': 2949, 'empty': 348}`).

## Caveats

- GRIP4 is a harmonized **derived product** (incl. crowdsourced OSM), so omitted /
  misclassified roads and positional error of a few pixels are possible.
- Local (and to a lesser extent tertiary) roads are often < 10 m wide, near the S2/Landsat
  resolution limit; the ~1 px dilation makes them visible but width is not preserved
  (all classes rasterized to the same ~20–30 m mask), so the label distinguishes road
  **type**, not road width.
- Bounded to 5 of 7 GRIP4 regions (see above); not global.

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.grip4_global_roads_inventory
```
(idempotent; `--probe` scans/reports without writing; `--workers N` sets pool size.)
