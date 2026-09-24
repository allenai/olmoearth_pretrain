# RCMAP (Rangeland Condition Monitoring) — sagebrush fractional cover

- **slug**: `rcmap_rangeland_condition_monitoring`
- **task_type**: regression
- **num_samples**: 5000
- **source**: USGS / MRLC — RCMAP (Rangeland Condition Monitoring Assessment and
  Projection) Fractional Component Time-Series across Western North America.
  Product page https://www.mrlc.gov/data ; data release DOI
  https://doi.org/10.5066/P13QF8HT (V7, 1985-2024).
- **license**: public domain (US Government work).

## What the source is

RCMAP maps the per-pixel **percent cover (0-100)** of ten rangeland components
(annual herbaceous, bare ground, herbaceous, litter, non-sagebrush shrub, perennial
herbaceous, sagebrush, shrub, tree, and shrub height) across western North America at
**30 m**, one map per year (1985-present), derived from **Landsat via a regression model
trained on field plots**. Native rasters are single-band **uint8, EPSG:5070 (CONUS
Albers), nodata = 101** (values 0-100 are valid percent cover; 101 marks non-mapped /
masked pixels — water, non-rangeland, outside the mapping area).

## Component / target choice

The manifest lists nine fractional components as candidate classes. Fractional cover is a
**regression** target, and the spec directs picking one primary component for a
multi-component product. We regress the **sagebrush** component — RCMAP's flagship and
namesake product (the project exists to monitor sagebrush ecosystems). Each label patch is
a single-band continuous cover field, so only one component fits per dataset; sagebrush is
the most defining choice.

## Access / download

Downloaded the current MRLC sagebrush decade bundle
`https://www.mrlc.gov/downloads/sciweb1/shared/mrlc/data-bundles/Sagebrush_2015_2025.zip`
(~14 GB; one `rcmap_sagebrush_{year}.tif` per year 2015-2025) to
`raw/rcmap_rangeland_condition_monitoring/`. The five years used (below) are extracted from
the zip into `raw/.../tifs/`. The MRLC bulk product is packaged only as full-extent
per-decade zips, so the 14 GB bundle is the minimum download granularity for this
component; this is the bounded-tile approach of spec §5 (large regional derived product,
no in-situ reference alternative — download only one component/decade and sample tiles).

## Processing

- **Bounded-tile sampling** across the full RCMAP mapping extent (western North America).
  For each of 5 years spanning the Sentinel era — **2016, 2018, 2020, 2022, 2024** (all
  within the manifest 1985-2024 range) — a decimated (factor 21 ≈ 630 m) nearest read of
  the year raster yields candidate pixel centers; valid (0-100) pixels only. 60k
  candidates/year → 300k pooled.
- **Reprojection**: candidate Albers pixel centers → WGS84 (vectorized pyproj), then each
  selected tile is written in **local UTM at 10 m/pixel, 64×64 (~640 m)**. Source 30 m
  Albers is reprojected per-tile with **bilinear** resampling (continuous cover field;
  GDAL WarpedVRT respects the declared nodata=101 so nodata is not blended into valid
  pixels). Output pixels with value <0 or >100 (or equal to source nodata) are set to
  `-99999`.
- **Bucket balancing** (sagebrush cover is heavily zero-inflated — most of the mapped
  extent is not sagebrush steppe). Tiles are balanced across **fixed percent-cover buckets
  `[0,1,5,10,20,30,101]`** by the center-pixel cover value, ~833 tiles per bucket. This
  gives the label bank an even spread of cover levels instead of mostly-0% tiles. Buckets
  are recorded in `metadata.json`.
- **Time range**: each tile gets a 1-year window `[year-01-01, (year+1)-01-01)` for its
  RCMAP product year (seasonal/annual label per spec §5). No change labels.

## Output

- `datasets/rcmap_rangeland_condition_monitoring/locations/{000000..004999}.tif` —
  single-band **float32**, local UTM, 10 m, 64×64, nodata **-99999**, values = percent
  sagebrush cover 0-100.
- matching `.json` sidecars (crs, pixel_bounds, ≤1-year time_range, source_id).
- `metadata.json` — regression block (`name: sagebrush_cover`, unit `percent cover`,
  dtype float32, value_range, nodata -99999, buckets).

## Stats

- **num_samples**: 5000. Year counts: 2016:1013, 2018:969, 2020:1005, 2022:1011, 2024:1002.
- Center-value bucket counts: {0-1%:835, 1-5%:833, 5-10%:833, 10-20%:833, 20-30%:833,
  30-100%:833}.
- Observed per-pixel cover range across tiles: **[0, 53] %** (sagebrush canopy cover
  rarely exceeds ~50%; this is expected/realistic). Pixel-level sample: mean ≈11%,
  median 8%, p90 27%, p99 36%, ~80% of valid pixels non-zero (bucket balancing pulls the
  sample toward vegetated sagebrush areas).
- Spatial spread: tiles fall in UTM zones 10N-15N (Pacific coast → Great Plains), across
  the western US and into the Canadian/Mexican fringes of the RCMAP extent.

## Verification

- 5 random tiles: single-band float32, 64×64, 10 m, UTM CRS, nodata -99999, values in
  range — all pass.
- All 5000 `.tif` have a matching `.json`; 0 time ranges exceed 366 days.
- Tile-center coordinates land in western North America (spot-checked 200; the few outside
  a tight CONUS bbox are legitimate eastern-Great-Plains / Canada extent points).
- Idempotent: re-running skips existing tiles (and reads them back so metadata stats stay
  correct).
- A full Sentinel-2 image overlay was not performed; georeferencing was validated via
  exact CRS/coordinate placement (standard USGS EPSG:5070→UTM reprojection).

## Caveats / judgment calls

- One component (sagebrush) chosen; the other RCMAP components (bare ground, herbaceous,
  etc.) are not included — each would be a separate single-band regression dataset.
- The live MRLC bundle is the current generation (2015-2025); the manifest references the
  1985-2024 V7 release. Both are the same RCMAP product line at 30 m; only Sentinel-era
  years 2016-2024 were used, all within the manifest range.
- Zero-inflation handled by fixed-bucket balancing rather than the quantile helper (which
  degenerates on zero-inflated data).

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.rcmap_rangeland_condition_monitoring --workers 64
```
(Downloads the ~14 GB Sagebrush_2015_2025 bundle on first run if absent; idempotent.)
