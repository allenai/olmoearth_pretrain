# OlmoEarth solar farm

- **Slug**: `olmoearth_solar_farm`
- **Name**: OlmoEarth solar farm
- **Task type**: classification (dense segmentation), 2 classes
- **Status**: completed
- **Num samples**: 1018 tiles (solar_farm class present in 1000, background present in 1000)
- **Source**: local rslearn dataset (`have_locally: true`, not copied)
  `/weka/dfive-default/rslearn-eai/datasets/solar_farm/dataset_v1/20250605`
- **Family / region**: solar / Global
- **Annotation method**: manual annotation (Satlas)
- **License**: ODbL/internal

## What the source is

The existing OlmoEarth eval / Satlas solar-farm segmentation dataset: 3561 manually
annotated windows in group `default` (splits train=3115, val=446; **both used**), spread
over 58 UTM zones. Each window is a variable-size crop (~180–490 px) already in a **local
UTM projection at 10 m/pixel**. Relevant layers per window:

- `label_raster` band `label` — single-band uint8 PNG (`single_image` format): `0` =
  background, `1` = solar_farm (ground-mounted photovoltaic footprint).
- `mask` band `mask` — single-band uint8 PNG: `255` = valid/annotated region (covers
  ~96–100 % of each window), `0` = outside the annotated footprint (window borders).

## Class mapping

| output id | name       | source |
|-----------|------------|--------|
| 0         | background | label_raster == 0 within mask |
| 1         | solar_farm | label_raster == 1 within mask |
| 255       | nodata     | mask == 0 (unannotated border pixels) |

## Processing decisions

- **No resampling.** Source is already local UTM @ 10 m, so `read_label_raster`'s WarpedVRT
  reprojection is unnecessary. Also, the label/mask layers are `single_image` **PNG** (not
  GeoTIFF), which the shared `rslearn_read.read_label_raster` (GeotiffRasterFormat) cannot
  open. Each window's label + mask PNGs are read directly with PIL — the exact array
  rslearn's own decoder returns — and tiled into ≤64×64 patches on the native pixel grid.
  Tile pixel bounds are derived from the window's `bounds` + pixel offsets, so
  georeferencing is exact.
- **Masking.** Output uint8 = label where `mask==255`, else `255` (nodata). This treats
  unannotated window borders as ignore. Solar pixels sit ~entirely within the mask
  (verified: stray out-of-mask solar is edge-only, tens of px).
- **Tile filtering.** A tile is kept only if it has ≥256 valid (masked-in) pixels. A tile
  counts as a `solar_farm` tile only if it has ≥10 solar pixels (avoids noise-level
  positives); it counts as `background` if it has ≥1 background pixel.
- **Tiles-per-class balanced (spec §5).** 143,374 non-empty tiles found (7,687 with solar,
  135,687 background-only). The rare class is prioritized: all solar tiles shuffled and
  capped at 1000; 982 of those already carry background, so 18 background-only tiles were
  added to bring background to 1000. Final = **1018 tiles** (solar_farm=1000,
  background=1000). Well under the 25k cap.
- **Time range.** Each sample uses its source window's own ~180-day acquisition range
  (≤1 year). Solar farms are persistent, so this is a valid annual-style label;
  `change_time` is null.

## Verification

- 1018 `.tif` + 1018 matching `.json`. All single-band uint8, UTM CRS (EPSG:326xx/327xx),
  10 m, dims ≤64. Pixel values ⊆ {0, 1, 255}. All time ranges ~180 days.
- Spatial/temporal sanity: overlaid labels on the source window's Sentinel-2 RGB for a
  full-solar tile and a mixed tile — solar labels align with the characteristic panel-array
  texture / distinct installation regions; background sits off-panel.

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.olmoearth_solar_farm
```

Idempotent: existing `locations/{id}.tif` are skipped on re-run.
