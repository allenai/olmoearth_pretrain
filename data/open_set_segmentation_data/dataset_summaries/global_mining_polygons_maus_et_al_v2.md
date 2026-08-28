# Global Mining Polygons (Maus et al. v2)

- **Slug**: `global_mining_polygons_maus_et_al_v2`
- **Status**: completed
- **Task type**: classification (positive-only, single foreground class)
- **Label type**: polygons
- **Num samples**: 25,000 label tiles
- **Family / region**: mining / Global (145 countries)
- **License**: CC-BY-SA-4.0

## Source

Maus, V., da Silva, D.M., Gutschlhofer, J., da Rosa, R., Giljum, S., Gass, S.L.B.,
Luckeneder, S., Lieber, M., McCallum, I. (2022): *Global-scale mining polygons (Version 2)*.
PANGAEA, https://doi.org/10.1594/PANGAEA.942325 — supplement to Maus et al., "An update on
global mining land use", *Scientific Data* 9, 433 (2022),
https://doi.org/10.1038/s41597-022-01547-4.

The dataset contains **44,929 hand-digitized mining land-use polygons** covering
101,583 km² across 145 countries. Each polygon delineates the *entire surface footprint* of
a mine — open cuts/pits, tailings dams, waste-rock dumps, water ponds, processing
infrastructure and other mining-related land cover — as one undifferentiated class. Polygons
were digitized by visual interpretation of the **2019 Sentinel-2 cloudless 10 m mosaic**
(aided by Google Satellite / Bing), within a 10 km buffer of 34,820 S&P mining coordinates.
Independent validation: overall accuracy 88.3%, F1 0.87 (mine class).

## Access method

Direct HTTPS download of the main GeoPackage — **no account required** for single files
(only PANGAEA's `allfiles.zip` bundle needs login):

```
https://download.pangaea.de/dataset/942325/files/global_mining_polygons_v2.gpkg   # 23.5 MB
```

Saved to `raw/global_mining_polygons_maus_et_al_v2/global_mining_polygons_v2.gpkg`. Fields:
`ISO3_CODE`, `COUNTRY_NAME`, `AREA` (km²), `geom` (WGS84 EPSG:4326 polygons). The companion
grid rasters (30 arc-sec / 5 / 30 arc-min), per-country CSV, and validation-point GPKG were
**not** used (the polygons are the label signal).

## Class mapping

Single foreground class; **positive-only** (spec §5 — no synthetic negatives are
fabricated):

| id | name | meaning |
|----|------|---------|
| 0  | mining area | inside a Maus et al. mining polygon (any mining ground feature) |
| 255 | *(nodata)* | everything outside a polygon — left as ignore; assembly adds negatives |

The manifest lists 6 fine feature types (pits, tailings dams, waste rock, ponds,
processing). These are **not** per-polygon attributes in the release (only ISO3/country/area
exist), so per-feature-type classification is not expressible; we map to the single
undifferentiated mining-area footprint.

## Processing

- **Rasterization**: each selected polygon → one 64×64 UTM 10 m tile centered on the polygon
  (placement point = centroid, or a guaranteed-interior representative point when the
  centroid falls in a concavity). All Maus polygons intersecting the tile bbox are burned to
  class 0 with `all_touched=True` (so the smallest mines, down to ~3 px, survive); the rest
  of the tile is nodata 255. Geometries intersecting each tile are read on demand from the
  GeoPackage via a pyogrio bbox filter (GPKG R-tree spatial index), so both phases
  parallelize over a 64-worker pool.
- **Sampling**: geographically-stratified round-robin over 1-degree lon/lat cells (as in the
  sibling `global_mining_footprint_tang_werner` script) so dense mining regions do not
  dominate; **one tile per selected polygon, capped at 25,000 tiles** (spec hard cap; the
  full set has 44,929 polygons so ~20k are not sampled).
- **Large polygons**: ~40% of polygons (18,160) exceed a single 640 m tile (up to
  2,546 km²); for these the tile captures a **central all-mining window** rather than the
  whole footprint — still a valid positive patch. Foreground fraction across a 200-tile
  sample: median ~0.50, mean ~0.54, with 12% fully-mine tiles.
- **Time range**: 1-year window anchored on **2019** (2019-01-01 → 2020-01-01). Although the
  manifest lists `time_range` 2016-2019 (the S&P coordinate vintage), the polygons were
  digitized specifically from the 2019 S2 mosaic, so 2019 is the year the labels are known to
  match the imagery; mining land use is persistent, so a static 2019 window is appropriate.
  `change_time` is null (not a change dataset).

## Output

- `datasets/global_mining_polygons_maus_et_al_v2/metadata.json`
- `datasets/global_mining_polygons_maus_et_al_v2/locations/{000000..024999}.tif` — single-band
  uint8, local UTM @ 10 m, 64×64, nodata 255.
- `datasets/global_mining_polygons_maus_et_al_v2/locations/{000000..024999}.json` — per-sample
  crs / pixel_bounds / 1-year (2019) time_range / classes_present.

## Verification (§9)

- 25,000 `.tif` each paired with a `.json`, ids contiguous `000000..024999`.
- Sampled tiles: all single-band uint8, 64×64, UTM at 10 m, pixel values ⊆ {0, 255} only
  (no invalid class ids); 59 distinct UTM zones (global coverage).
- Georeferencing spot-check: tile centers reproject back to lon/lat that fall **inside** the
  source Maus polygons in the expected countries (verified for Tanzania, Canada, Australia).
- Idempotent: re-running skips existing `locations/{id}.tif` (same seed → same stratified
  selection, so ids are stable).

## Caveats

- Positive-only: outside-polygon pixels are ignore (255), not background; negatives come from
  the assembly step.
- Only the undifferentiated mining footprint is available (no per-feature-type classes).
- Large mines are represented by a central window, not their full extent.

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.global_mining_polygons_maus_et_al_v2 --workers 64
```
