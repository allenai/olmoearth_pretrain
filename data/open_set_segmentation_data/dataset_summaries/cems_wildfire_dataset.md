# CEMS Wildfire Dataset — cems_wildfire_dataset

- **Status**: completed
- **Task type**: classification (dense_raster; burn-severity segmentation)
- **Samples**: 2760 label patches (64x64, UTM 10 m, uint8, nodata=255)
- **Source**: HuggingFace `links-ads/wildfires-cems` (repo GitHub `MatteoM95/CEMS-Wildfire-Dataset`); CC-BY-4.0
- **URL**: https://github.com/MatteoM95/CEMS-Wildfire-Dataset
- **Access**: public HF download of split `*.tar.NNNN.gz.part` files (concatenated per split, then untarred). No credentials.
- **Source sample dirs scanned**: 433 (train+val+test all used)

## What the dataset is

500+ Copernicus EMS rapid-mapping wildfire activations (Jun 2017 - Apr 2023, mostly Europe). Each sample directory carries a post-fire Sentinel-2 L2A GeoTIFF plus georeferenced label rasters: `*_DEL.tif` (binary burned-area delineation, present for all samples) and `*_GRA.tif` (burn-severity grading 0-4, present for a subset). Rasters are EPSG:4326 at ~10 m; GRA non-zero footprint exactly matches DEL (verified), so GRA subsumes DEL where present.

## Processing

- Reprojected each categorical label from EPSG:4326 (~10 m) to local UTM at 10 m with **nearest** resampling (categorical), via `calculate_default_transform` + `rasterio.warp.reproject`.
- Tiled each reprojected label into non-overlapping 64x64 UTM patches; dropped tiles touching the reprojection border (nodata) and tiles with no burned pixels.
- **Unified class scheme** (spec sec.5 multi-target combine): severity grades map directly to ids 0-4; for delineation-only activations (no GRA) burned pixels get id 5 (`burned_ungraded`). Unburned (id 0) is a real observed background class (CEMS delineates the whole AOI), so this dataset is NOT positive-only.
- **Time**: burn scars are change/event labels. `change_time` = post-fire S2 acquisition date (from `*_S2L2A.json`); `time_range` = +/-180 d (360 d) around it. Burn scars persist for months, so a yearly window is well-posed (not flagged).
- **Sampling**: tiles-per-class balanced, rarest-severity-first, <=1000 tiles per class, 25k total cap (`sampling.select_tiles_per_class`).
- Did NOT apply the cloud mask (`*_CM.tif`): the CEMS burn labels are authoritative vector rapid-mapping products independent of clouds in the particular S2 mosaic.

## Classes (id: name — candidate tiles / selected tiles)

- 0: no_visible_damage — 32961 / 1946
- 1: negligible_to_slight — 5380 / 1041
- 2: moderately_damaged — 26450 / 1333
- 3: highly_damaged — 28388 / 1429
- 4: destroyed — 15159 / 1000
- 5: burned_ungraded — 15934 / 1000

Class 1 (negligible_to_slight) is the least common severity grade but has enough candidate tiles to reach the ~1000/class target. Class 0 (background) is co-present in nearly every burned tile so it exceeds the 1000 guideline; this is inherent (cannot drop background without dropping burn signal). All severity classes reach ~1000-1400 selected tiles; downstream assembly drops any class below its minimum.

## Verification

Output tifs are single-band uint8, UTM CRS at 10 m, 64x64, values in {0..5} plus 255 nodata; each tif has a matching JSON with a 360-day `time_range` and a `change_time`. Georeferencing derived from the reprojection transform.

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.cems_wildfire_dataset
```
