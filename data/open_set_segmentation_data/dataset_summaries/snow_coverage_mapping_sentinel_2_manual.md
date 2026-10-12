# Snow Coverage Mapping (Sentinel-2, manual)

- **Slug:** `snow_coverage_mapping_sentinel_2_manual`
- **Status:** completed
- **Task type:** classification (dense per-pixel)
- **Num samples:** 1954 label patches (64×64, 10 m, single-band uint8)
- **Family:** snow_ice · **Label type:** dense_raster · **Sensors:** Sentinel-2

## Source

Wang, Y.; Su, J.; Zhai, X.; Meng, F.; Liu, C. *Snow Coverage Mapping by Learning from
Sentinel-2 Satellite Multispectral Images via Machine Learning Algorithms.* Remote Sens.
2022, 14(3), 782. DOI [10.3390/rs14030782](https://doi.org/10.3390/rs14030782).

- Paper: https://www.mdpi.com/2072-4292/14/3/782
- Data + code: https://github.com/yiluyucheng/SnowCoverage (branch `main`)
- License: CC-BY-4.0 (open access)

The largest manually-annotated Sentinel-2 snow-segmentation dataset: **40 Sentinel-2 L2A
scenes** across six continents (2019–2021), each a ~1000×1000 px crop that was
pixel-labelled in QGIS (Semi-Automatic Classification Plugin, Minimum-Distance seeding)
and **checked by two human experts**. The paper claims 3 classes (snow / cloud /
background); the manifest's guessed 4-class snow/cloud/water/land scheme is **incorrect**
— the actual masks have three classes.

## Access method

Downloaded **only the label masks** `datasets/masks/*.tif` (40 files, ~19 MB total) via
the GitHub raw endpoint, listed through the repo git-tree API. The ~1.3 GB of
co-registered Sentinel-2 raw imagery (`datasets/raw_images/*.tif`) was **not**
bulk-downloaded — pretraining supplies its own imagery. One raw image (`T59GLM`) was
fetched transiently only to verify the class-value mapping spectrally, then discarded.

Raw masks live at
`raw/snow_coverage_mapping_sentinel_2_manual/masks/` on weka (+ `SOURCE.txt`).

## Class / label mapping

Source masks are single-band **int16**, already in scene-local UTM at 10 m/pixel, north-up
(`nodata = -999`). Class values are **1/2/3** (plus 18 stray `0` px across the whole
corpus, treated as nodata). The value↔class assignment was **verified spectrally** against
the raw Sentinel-2 bands (definitive, since the paper only gives colour legends): snow has
high visible reflectance, very low SWIR (B11) and NDSI≈0.85; cloud is bright across *all*
bands incl. SWIR; background is dark/low-reflectance.

| source value | output id | class | note |
|---|---|---|---|
| 1 | 0 | background | snow/cloud-free land/dark surface (rock, soil, veg, water) |
| 2 | 1 | cloud | bright across all bands incl. SWIR |
| 3 | 2 | snow | high visible reflectance, low SWIR, high NDSI |
| -999 / stray 0 | 255 | nodata/ignore | |

`background` is a genuine, spatially-meaningful class here, so it is kept as a normal class
id (0), not treated as ignore.

## Processing

`label_type = dense_raster`. Each mask is already UTM 10 m north-up, so **no reprojection**:
masks are tiled directly into **64×64** patches, reusing each scene's CRS, with rslearn
integer pixel bounds derived from the raster transform (GEE exports are S2-grid aligned,
origins multiples of 10). Partial edge tiles (scene dims are ~1000–1051, not multiples of
64) are dropped by floor-division tiling.

- A tile is a candidate if it is ≤50% nodata and contains ≥32 px of at least one class.
- Selection: **tiles-per-class balanced** (`sampling.select_tiles_per_class`, ≤1000
  tiles/class, 25k dataset cap), rarest class filled first.
- 9262 candidate tiles → **1954 selected** from 39 scenes.
- Tiles containing each class: background 1023, cloud 1022, snow 1000.

## Time-range handling

Snow / cloud / background are **per-image states** valid only for the exact Sentinel-2
acquisition (snow is highly time-specific). Per spec §5 (specific-image labels),
`time_range` is a **~1-hour window at the scene's acquisition timestamp** parsed from the
product-ID prefix in the filename (e.g. `20200804T223709` → 2020-08-04T22:37:09 UTC), and
`change_time = null`. All scenes are 2019–2021 (post-2016), so no pre-Sentinel filtering
was needed. This is not a change dataset.

## Verification

- All 1954 `.tif` are single-band **uint8, 64×64, UTM @ 10 m** (18 distinct UTM zones);
  every `.tif` has a matching `.json`; value set is {0,1,2} (+255 nodata); class map covers
  all values.
- All `time_range`s are valid ≤1-year (here 1-hour) windows; all `change_time` null.
- **Spatial/spectral overlay** (written tiles vs. raw S2 for scene `T59GLM`):
  background NDSI −0.06 / SWIR 965, cloud NDSI −0.06 / SWIR 3062, snow NDSI 0.80 / SWIR 609
  — matches the source scene statistics, confirming georeferencing and class mapping are
  correct.
- Re-running is idempotent (existing `.tif` skipped; deterministic selection/ids).

## Caveats

- 40 scenes only, so spatial diversity is modest; balancing caps each class near 1000 tiles.
- Cloud/snow boundaries are hard even for experts (the paper's motivation); a small fraction
  of edge pixels may be ambiguous, but masks were double-checked.
- `background` lumps together all non-snow/non-cloud surfaces (including open water), so it
  is a heterogeneous class.

## Reproduce

```bash
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.snow_coverage_mapping_sentinel_2_manual --workers 64
```
