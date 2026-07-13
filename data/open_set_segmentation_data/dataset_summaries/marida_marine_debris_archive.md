# MARIDA (Marine Debris Archive) — marida_marine_debris_archive

- **Status**: completed
- **Task type**: classification (dense_raster)
- **Samples**: 3322 label patches (64x64, UTM 10 m, uint8, nodata=255)
- **Source**: Zenodo record 5151941 (Kikaki et al., PLOS ONE 2022); CC-BY-4.0
- **URL**: https://doi.org/10.5281/zenodo.5151941
- **Access**: public Zenodo download (`download_zenodo('5151941', raw_dir)`), no credentials.

## What MARIDA is

Manually photo-interpreted Sentinel-2 pixel annotations distinguishing marine debris from co-occurring sea-surface features. The release provides 1381 patches, each a 256x256 Sentinel-2 crop already georeferenced in local UTM at 10 m/pixel, with a `*_cl.tif` class raster (float32; 0 = unlabeled, 1-15 = classes) and a `*_conf.tif` confidence raster (1=High, 2=Moderate, 3=Low). Annotations are sparse within each patch.

## Processing

- Each 256x256 `*_cl.tif` cropped into non-overlapping 64x64 UTM 10 m tiles (16 per patch); reused the source CRS/geotransform exactly (native UTM 10 m).
- Kept every tile containing >=1 labeled pixel (3322 of 22096 candidate crops).
- Class remap: MARIDA id 1-15 -> output id 0-14; unlabeled (0) -> 255 nodata.
- **Sampling**: tiles-per-class balanced. The full candidate set (3322 tiles) is far below the 25k cap and below the 1000/class target for all classes except Marine Water (1606 tiles). Kept all tiles: dropping Marine-Water-heavy tiles would also remove co-present rare debris classes, so no truncation was applied. Marine Water is the only class above the 1000 guideline.
- **Time range**: 1-day window of the Sentinel-2 acquisition date parsed from the scene name (`S2_dd-mm-yy_TILE`). Sea-surface features are transient, so each label is tied to its single acquisition (well under the 1-year limit).
- All classes are natively annotated on 10 m Sentinel-2, so all 15 are viable at 10 m (this dataset's raison d'être). Small-footprint classes (Ship, Wakes) are kept as annotated.

## Classes (output id: name -> tiles containing class)

- 0: Marine Debris — 687
- 1: Dense Sargassum — 68
- 2: Sparse Sargassum — 182
- 3: Natural Organic Material — 105
- 4: Ship — 226
- 5: Clouds — 386
- 6: Marine Water — 1606
- 7: Sediment-Laden Water — 203
- 8: Foam — 82
- 9: Turbid Water — 373
- 10: Shallow Water — 104
- 11: Waves — 125
- 12: Cloud Shadows — 122
- 13: Wakes — 144
- 14: Mixed Water — 177

## Verification

Output tifs are single-band uint8, UTM CRS at 10 m, 64x64, values in 0-14 plus 255 nodata; each tif has a matching JSON with a 1-day time_range. Georeferencing reuses the source patches' exact UTM transform.

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.marida_marine_debris_archive
```
