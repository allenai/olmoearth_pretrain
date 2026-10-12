# Pan-Arctic Ice-Wedge Polygons

- **Slug**: `pan_arctic_ice_wedge_polygons`
- **Status**: completed
- **Task type**: regression (per-pixel ice-wedge-polygon coverage density, fraction 0-1)
- **Num samples**: 5000
- **Source**: NSF Arctic Data Center / Permafrost Discovery Gateway — Witharana, Liljedahl
  et al., "Ice-wedge polygon detection in satellite imagery from pan-Arctic regions,
  Permafrost Discovery Gateway, 2001-2021", https://doi.org/10.18739/A2KW57K57
- **License**: CC-BY-4.0

## What the source is

A deep-learning (MAPLE / CNN) inventory of >1 billion individual ice-wedge polygons detected
in very-high-resolution (~0.5 m Maxar commercial) satellite imagery across the pan-Arctic
tundra (2001-2021, mostly 2016-2021). Distributed as per-polygon vectors (shapefiles /
geopackages with per-polygon attributes incl. a low-centered vs high-centered microtopography
class) AND as a rasterized **coverage-density** product: each cell's value is the fraction of
the cell area occupied by detected ice-wedge polygons.

The DOI landing page (arcticdata.io) holds only metadata + a pipeline diagram + a PDF; the
actual geospatial data is served from an open HTTP directory at
`http://arcticdata.io/data/10.18739/A2KW57K57/` with subfolders `iwp_geotiff_high/` (raster
density), `iwp_geopackage_high/`, `iwp_shapefile_detections/`, `iwp_shapefile_footprints/`.
The density rasters are a **WorldCRS1984Quad** tile pyramid (EPSG:4326, 256x256 tiles),
`iwp_geotiff_high/WGS1984Quad/{z}/{x}/{y}.tif`, zoom 0-15.

## Key decisions

- **Regression on density, not the manifest's low-/high-centered classes.** Individual
  polygons (~10-20 m) are NOT reliably resolvable as objects at 10-30 m S2/Landsat, and the
  publicly-served *raster* product is a single-band coverage-density layer with **no per-cell
  microtopography (LCP/HCP) split** (that attribute exists only in the per-polygon
  geopackage). Per the manifest note ("Use rasterized density at S2/Landsat scale"), the label
  is a per-pixel regression = ice-wedge-polygon coverage density (fraction 0-1), which
  captures polygon presence/density (observable as patterned-ground texture at S2 scale).
- **Zoom level 14 as the density source.** z=14 is a properly *averaged* overview at
  ~4.8 m/px (lat) / ~1.6 m/px (lon) at 70N, whose values are genuine area fractions. z=15 is
  the ~2.4 m native level; z=13 and coarser are *summed* overviews whose values are inflated
  (>>1) and thus unusable as fractions. Verified empirically: z=14 mean over a tile ≈ the mean
  of its four z=15 children (averaging), whereas z=13 means are ~2-4x higher (summing).
- **Bounded sampling of a huge global product** (§5). We do NOT bulk-download. We fetch z=14
  tiles only over **10 representative high-IWP tundra regions** (~0.4°lon × 0.2°lat each):
  Alaska (prudhoe, utqiagvik, teshekpuk), Arctic Canada (tuktoyaktuk, banks, mackenzie),
  Siberia (lena, yamal, kolyma, indigirka). Total raw download ≈ 2.9 GB / 5779 tiles.
  (Two probed regions — Seward Peninsula, Taimyr — had no staged tiles at those spots and were
  dropped.)
- **Reprojection**: each region's tiles are mosaicked (EPSG:4326) and reprojected to local UTM
  at 10 m with **nodata-aware average** resampling (a continuous fraction field; unmapped gaps
  stay nodata). Only 64x64 windows that are ≥98% within mapped ground are kept. The fraction is
  **clipped to [0, 1]** (source values >1 are duplicate-scene overlap artifacts).
- **Bucket balancing**: the density distribution is heavily zero-inflated, so the 6309
  candidate windows were balanced across fixed density buckets
  `[0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 1.0]` down to 5000 samples, giving an even spread
  of density levels (selected-window mean-density bucket counts:
  385 / 552 / 911 / 1307 / 1184 / 554 / 107).
- **Time range**: multi-year (2001-2021) composite of a persistent geomorphic landform;
  anchored to a representative 1-year Sentinel-era window (2020). `change_time` = null.

## Output

- `datasets/pan_arctic_ice_wedge_polygons/locations/{000000..004999}.tif` — single-band
  float32, local UTM, 10 m/pixel, 64x64 (~640 m), nodata -99999. Values are IWP coverage
  fraction in [0, 1].
- `.json` sidecars carry `crs`, `pixel_bounds`, a 1-year `time_range` (2020), and
  `source_id` (`{region}/z14/px_{i}_{j}`).
- `metadata.json` — regression block `ice_wedge_polygon_coverage_density`, unit "fraction
  (0-1)", value_range [0.0, 1.0].

## Per-region sample counts (selected)

alaska_prudhoe 659, canada_mackenzie 657, russia_kolyma 604, russia_yamal 585, russia_lena
575, canada_banks 521, russia_indigirka 404, canada_tuktoyaktuk 371, alaska_utqiagvik 331,
alaska_teshekpuk 293.

## Verification

- 5000 tif + 5000 json, all paired; single-band float32, UTM 10 m, size 64x64, nodata -99999,
  per-pixel values within [0, 1].
- Geolocation sanity: window centers reproject back to their intended regions (e.g.
  `russia_lena` → 126.16°E, 72.36°N; `alaska_utqiagvik` → -156.79°, 71.28°;
  `canada_tuktoyaktuk` → -132.99°, 69.58°) — all in known pan-Arctic IWP tundra.
- Idempotent: re-running skips existing `{id}.tif` (verified the write path leaves existing
  files untouched).
- A full Sentinel-2 pixel overlay was not performed (the label is a derived continuous density
  field rather than a hard land-cover class); geolocation was validated against the known
  ice-wedge-polygon tundra regions instead.

## Caveats

- The label is a **derived deep-learning product** (not in-situ reference), so density values
  carry the detector's errors; sampling is confined to mapped tundra where detections exist.
- The low-centered vs high-centered microtopography distinction from the manifest is **not**
  represented (unavailable in the raster product); only total polygon coverage density is used.
- Source values >1 (overlapping duplicate scene footprints) were clipped to 1.0.

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.pan_arctic_ice_wedge_polygons --workers 64
```
