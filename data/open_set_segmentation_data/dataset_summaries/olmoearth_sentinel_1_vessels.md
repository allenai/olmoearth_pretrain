# OlmoEarth Sentinel-1 vessels

- **Slug**: `olmoearth_sentinel_1_vessels`
- **Registry name**: OlmoEarth Sentinel-1 vessels
- **Task type**: classification (object-detection encoded as per-pixel classes)
- **Status**: completed — **2000 samples** (1000 vessel-positive tiles + 1000 background-negative tiles)
- **Family**: vessels · **Region**: global oceans · **License**: internal

## Source

Local rslearn dataset (`have_locally: true`, **not copied** — `raw/olmoearth_sentinel_1_vessels/SOURCE.txt` points at it):

```
/weka/dfive-default/rslearn-eai/datasets/sentinel1_vessels/dataset_v1/20250602
```

The existing OlmoEarth / Satlas **Sentinel-1 SAR** vessel-detection eval. 1776 windows in
four groups (`train_ascending` 1040, `train_descending` 496, `val_ascending` 160,
`val_descending` 80 — train/val × orbit direction). Each window is a specific-image S1 crop
already in a **local UTM projection at 10 m/pixel** (~810×810 px) with a **~10-minute S1
acquisition `time_range`** (all windows are 2021 → post-2016). Manifest `label_type` is
`bboxes`, but the on-disk annotations are **object-centroid Points**.

Config `layers.label` is a vector layer with `class_property_name = "category"`. Each label
GeoJSON (`layers/label/data.geojson`) has one `Point` feature per vessel with
`properties.category == "vessel"`. `metadata.options.has_objects` reliably flags
vessel-bearing windows: **677 positive** windows (1412 vessels total) and **1099 vessel-free**
windows.

### Coordinate convention (verified)

The label GeoJSON declares a WGS84 `crs` header, but the point coordinates are actually in
the **window projection (pixel) coordinates** — e.g. a feature at `[21357.7, -58481.8]` lies
inside window bounds `[20775, -58793, 21587, -57987]`. This is the same quirk as the
wind-turbine `label` group. So coordinates are used directly as pixel coords in the window's
own UTM projection; **no reprojection** is applied.

## Encoding (detection → per-pixel classes, spec §4)

- Classes: `0 = background` (open water), `1 = vessel`. `nodata = 255`.
- One **32×32** context tile per annotated vessel, centered on the vessel pixel, written in
  the window's own UTM projection (source already local UTM @ 10 m → georeferencing exact).
- Vessel = **1×1 positive** (class 1), ringed by a **10 px nodata (255) buffer** (vessel
  centroids are not pixel-exact and SAR layover shifts bright returns slightly); everything
  else is background (0). All vessels of the source window that fall inside a tile are marked.
- **Negatives**: background-only 32×32 tiles randomly placed inside vessel-free windows
  (`has_objects == false`), giving the background class spatially-meaningful negatives
  (spec §5 detection exception).

Detection params: `tile_size=32, positive_size=1, buffer_size=10`.

## Sampling & time range

- Single vessel class → up to **PER_CLASS=1000** positive tiles + **N_NEGATIVES=1000**
  negatives = 2000 samples (well under the 25k cap). One candidate positive tile is generated
  per annotated vessel (1412 available), shuffled (seed 42), truncated to 1000.
- All four groups/splits used (splits are pretraining-agnostic, spec §5).
- **Time range**: each sample uses its window's own **~10-minute S1 acquisition window**
  (specific-image label; vessels are point-in-time — only the matching S1 acquisition shows
  them). All ≤ 1 year; all 2021.
- `sensors_relevant = ["sentinel1", "sentinel2"]` — SAR is the native sensor; a coincident
  optical S2 pass within the short window could also catch the vessel. Landsat omitted (its
  revisit essentially never coincides with the ~10-min window and 15–30 m can't localize
  small vessels).

## Verification (spec §9)

- 2000 `.tif` + 2000 `.json`, 1:1 paired. Every tile: single-band, **uint8**, UTM CRS at
  **10 m**, **32×32**, `nodata=255`.
- Values across all tifs ⊆ `{0, 1, 255}`, fully covered by the class map. 1000 positives
  carry class 1; negatives are all-background (0). 17 distinct UTM zones (global spread).
- All `time_range`s ≤ 360 days.
- **Spatial sanity**: for 6 positive samples, the Sentinel-1 VV backscatter at the vessel
  center is **2.6–28× brighter** than the tile median — labels sit squarely on bright SAR
  vessel signatures.

## Caveats

- Detection encoding, not dense segmentation: the informative signal is the 1×1 positive +
  buffer within a background field.
- Negative tiles assume `has_objects == false` windows truly contain no vessels (they are the
  eval's dedicated vessel-free windows).

## Reproduce

```bash
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.olmoearth_sentinel_1_vessels
```

Idempotent: re-running skips already-written `locations/{id}.tif`.
