# OlmoEarth landslide (Sen12Landslides)

- **Slug**: `olmoearth_landslide_sen12landslides`
- **Task type**: classification (dense per-pixel, binary landslide-scar segmentation)
- **Status**: completed — 1000 label tiles
- **Label type**: `dense_raster`
- **have_locally**: true (source not copied; `raw/<slug>/SOURCE.txt` points at it)

## Source

Local rslearn project (Sen12Landslides): binary landslide-scar segmentation from
Sentinel-1, Sentinel-2 and SRTM pre/post-event acquisitions, manual annotation, global.

- Path: `/weka/dfive-default/piperw/rslearn_projects/data/landslide/sen12landslides/all_positives`
- Group used: `windows/sen12_landslides` (74847 positive + 74847 negative windows).
- Label layer: `layers/label_raster/label/geotiff.tif` (uint8; `0`=no_landslide,
  `1`=landslide, `2`=no_data buffer ring). A vector `label` layer exists too but the raster
  is authoritative and already rasterized.
- Each source window is **already 64x64 at 10 m in a local UTM CRS** — no reprojection or
  resampling needed; we read, remap, and re-emit.

### Judgment call — group scope
The rslearn project contains several groups: `sen12_landslides`, `glc`, `icimod`,
`fwn_mtli`, `osm_ski`, `osm_ski_resorts_trial`. Only `sen12_landslides` corresponds to this
manifest entry ("from Sen12Landslides"); the others are separate landslide inventories
(Global Landslide Catalog, ICIMOD, ski-resort trials) and were **excluded**. If those should
become their own datasets, they can be processed separately.

### Judgment call — positive windows only
Each location has a paired `positive` window (time range spanning the event; contains the
scar) and a `negative` window (same location, one year earlier; label all `no_landslide`).
We use **only the positive windows**. Positive tiles already contain abundant `no_landslide`
background, so with 2 classes and tiles-per-class balancing they saturate both classes at
1000; adding the negatives would only contribute near-duplicate all-background tiles at the
same locations. Per spec §5 the assembly step supplies negatives from other datasets, so no
real information is lost. (These are not fabricated negatives — the decision is about
avoiding redundant duplicate-location tiles, not about inventing background.)

## Class scheme

| id  | name          | meaning |
|-----|---------------|---------|
| 0   | no_landslide  | source label 0 (observed, no landslide) |
| 1   | landslide     | source label 1 (manually annotated landslide scar) |
| 255 | nodata/ignore | source label 2 = 30 m `no_data` buffer ring around scars |

Output GeoTIFFs: single-band **uint8**, local UTM, **10 m/pixel**, **64x64**, nodata **255**.

## Sampling

- Tiles-per-class balanced (spec §5), `<= 1000` tiles/class, via
  `sampling.select_tiles_per_class`. Every positive tile contains both classes, so
  selection stops at 1000 tiles: `no_landslide` in 1000 tiles, `landslide` in 1000 tiles.
- 74847 positive candidate windows scanned; all contained `landslide` pixels and fell in
  2016-2023, so no candidates were dropped for emptiness or the pre-2016 rule.
- **num_samples = 1000**.

## Time range / change label

Landslide is an event label. For each tile:
- `change_time` = `options.event_date` (the landslide event date).
- `time_range` = a **1-year window centered on `change_time`** (spec §5), so pretraining
  only pairs the tile with imagery whose window spans the event.
All event dates are 2016-2023 (Sentinel era). Pre-2016 windows are defensively filtered
(none were present). Event dates are specific days, so a 1-year centered window is
appropriate (the pre/post-event imagery lies within it).

## Verification

- 1000 `.tif` + 1000 matching `.json`. All tifs: single band, uint8, UTM CRS, 10 m,
  64x64, nodata 255, pixel values ⊆ {0, 1, 255}.
- Every sample JSON has `change_time` set and a `time_range` of exactly 365 days.
- Georeferencing exact: output tile bounds == source window bounds; label content ==
  source with `2 -> 255` remap; tile center lon/lat matches the source window's encoded
  coordinates.
- Re-running is idempotent (existing `{id}.tif` skipped).

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.olmoearth_landslide_sen12landslides
```

## Caveats

- Landslide scars are small; positive pixels are a minority within each 64x64 tile
  (typically tens to low-hundreds of pixels), with a 30 m ignore buffer around them.
  Downstream training should respect the 255 ignore label.
- Only 1000 of 74847 available positive windows are kept (the per-class cap). More could be
  emitted if a larger landslide sample is desired later (still <= 25k cap).
