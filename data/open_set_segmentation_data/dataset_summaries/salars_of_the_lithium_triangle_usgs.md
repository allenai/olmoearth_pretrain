# Salars of the Lithium Triangle (USGS)

- **Slug**: `salars_of_the_lithium_triangle_usgs`
- **Status**: completed
- **Task type**: classification (per-pixel; unified segmentation + point-detection)
- **Samples written**: 1,043 label tiles (`locations/{id}.tif` + `.json`)

## Source

Mihalasky, M.J., Briggs, D.A., Baker, M.S., Jaskula, B.W., Cheriyan, K., and
DeLoach-Overton, S.W., 2020, *Lithium Occurrences and Processing Facilities of Argentina,
and Salars of the Lithium Triangle, Central South America*: U.S. Geological Survey data
release, https://doi.org/10.5066/P9RLUH4F. ScienceBase item `5e90cd8f82ce172707edfc74`.
License: **public domain** (USGS).

Access method: the attached file geodatabase `Li_Triangle_ARG_MRP_NMIC.gdb` is delivered
as a 9.5 MB 7-zip (`Li_Triangle_ARG_MRP_NMIC.gdb.7z`) from the ScienceBase
`/catalog/file/get/` endpoint (no credentials). Downloaded with a browser User-Agent and
extracted with `py7zr`. Label-only extraction; no imagery pulled.

Feature classes:

| layer | geom | n | used as |
|---|---|---|---|
| `Salars_Li_Triangle_MRP_NMIC` | MultiPolygon | 186 | class 1 `salar` |
| `Arg_Occurrences_MRP_NMIC` (Deptype=Salar) | Point | 106 | class 2 `li_brine_occurrence` |
| `Arg_Occurrences_MRP_NMIC` (Deptype=Pegmatite) | Point | 18 | class 3 `li_pegmatite_occurrence` |
| `Arg_Facilities_MRP_NMIC` | Point | 10 | class 4 `processing_facility` |
| `Salar_Centroids_Li_Triangle_MRP_NMIC` | Point | 186 | **not used** (redundant with polygons) |
| `MRP_NMIC_Refs` | table | 145 | **not used** (bibliography) |

Native CRS is `ESRI:104015` (≈ WGS84 G1762 geographic); geopandas reprojects to EPSG:4326.
Salar coverage spans Argentina (82), Chile (59), Bolivia (37), border areas, and Peru (1).

## Unified class scheme (spec §5 multi-target)

This is a mixed polygon + point source, so the targets are combined into **one** dataset
with a single unified class map rather than split into separate datasets:

| id | name | source |
|---|---|---|
| 0 | background | non-target context within a tile |
| 1 | salar | salt-flat / laguna polygon footprint |
| 2 | li_brine_occurrence | Li brine occurrence point (Deptype=Salar) |
| 3 | li_pegmatite_occurrence | Li pegmatite occurrence point |
| 4 | processing_facility | Li processing/extraction facility point |
| 255 | nodata/ignore | detection buffer ring around each point |

## Processing recipe

All tiles are single-band **uint8**, **64×64** (640 m), local UTM at **10 m/px**,
north-up, written with rslearn `GeotiffRasterFormat` (exact georeferencing).

- **Salar tiles** (polygons, §4): each salar polygon is reprojected to local UTM and
  rasterized (class 1 vs background 0, `all_touched=True`). Salars ≤ 64 px are centered in
  one tile; larger salars (most — footprints run 0.4–12,078 km², e.g. Salar de Uyuni) are
  gridded into non-overlapping 64×64 windows, of which up to **16 intersecting windows per
  salar** are randomly sampled so no single giant salar dominates. 2,753 salar candidate
  windows generated; capped to 1,000 selected (see balancing).
- **Point tiles** (detection encoding, §4): each occurrence/facility point gets a 64×64
  context tile centered on it. Any salar polygons overlapping the tile are rasterized as
  class 1 (real context — most brine occurrences sit on a salt flat), then the point gets
  the tunable detection encoding: a **1×1 positive** of its class at the center ringed by a
  **10 px nodata (255) buffer** (point coords are not pixel-exact). All 134 point tiles are
  kept.

**Balancing** (`sampling.balance_tiles_by_class`, per_class=1000, total_cap=25,000): point
classes (10–106 tiles) are all retained; `salar` is capped at 1,000. No synthetic
negatives are fabricated (§5) — non-object pixels are genuine background/salar or the
nodata ring; downstream assembly supplies negatives from other datasets.

**Time** (§5): salars are persistent landforms and the outlines were digitized from
2018-2019 imagery → a static representative **1-year window `[2018-01-01, 2019-01-01)`**;
`change_time` is null. All labels are post-2016.

## Sample counts

- Total tiles: **1,043** (909 salar-only tiles + 134 point tiles).
- Tiles containing each class: salar 1,000 · li_brine_occurrence 106 ·
  li_pegmatite_occurrence 18 · processing_facility 10 · background 575 · nodata-ring 134.
- Pegmatite (18) and facility (10) are sparse; retained per §5 (downstream removes
  too-small classes).

## Verification (§9)

- 1,043 `.tif` each with a matching `.json`; all single-band uint8, ≤64×64, UTM CRS
  (EPSG:326xx/327xx), 10 m resolution. Pixel values observed: {0,1,2,3,4,255} — all covered
  by the class map (+255 nodata).
- Every `.json` has a 1-year `time_range` and `change_time=null`.
- **Spatial sanity (back-projection against source geometry):** 200/200 sampled salar-tile
  centers fall inside a source salar polygon; occurrence-point positive pixels
  back-project to within ~3–6 m of the source occurrence coordinates (40/40 within 200 m),
  confirming exact georeferencing. A direct Sentinel-2 pixel overlay was attempted but hit
  MGRS-tile-edge coverage gaps for the sampled point; note that the salar outlines are
  themselves manually digitized from 2018-2019 Sentinel-2 imagery, so the labels derive
  from that imagery by construction.

## Caveats

- **Observability**: `salar` (bright high-albedo salt flats) and `processing_facility`
  (evaporation-pond/plant complexes) are clearly observable at 10–30 m. `li_brine_occurrence`
  marks subsurface brine and `li_pegmatite_occurrence` a hard-rock outcrop — the *points*
  are not visually distinct objects at 10–30 m; they are kept as reference-location presence
  labels (with the salt-flat surface as observable context for brine), and downstream
  filtering drops classes that end up too sparse.
- Only the Argentina occurrence/facility feature classes exist in the geodatabase; salar
  *polygons* cover the whole triangle (AR/CL/BO/PE), but occurrence/facility points are
  Argentina-only (per the source's scope).

## Reproduce

```bash
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.salars_of_the_lithium_triangle_usgs
```

Idempotent: existing `locations/{id}.tif` are skipped.
