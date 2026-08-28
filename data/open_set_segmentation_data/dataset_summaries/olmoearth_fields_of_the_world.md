# OlmoEarth Fields of the World

- **Slug:** `olmoearth_fields_of_the_world`
- **Task type:** classification (dense per-pixel field-boundary segmentation)
- **Status:** completed — 1164 label patches
- **Source:** local rslearn dataset
  `/weka/dfive-default/rslearn-eai/datasets/fields_of_the_world/rslearn_dataset_utm/`
  (the ingested Fields of the World / FTW benchmark; also public on Source Cooperative).
- **License:** CC-BY (mixed, per national LPIS providers).
- **Annotation method:** national LPIS parcel polygons + manual QC, pre-rasterized to a
  dense label raster in the source rslearn dataset.
- **Region / time:** Global, 24+ countries (25 on-disk groups); per-window ~240-day
  growing-season windows spanning 2016–2023.

## Source layout

70,484 windows under `windows/<country>/<name>/`, each ~154 px tall × ~82–154 px wide,
already in a **local UTM projection at 10 m/pixel**. The label is a dense raster at
`layers/<name>/layers/label/label/geotiff.tif` (uint8). Verified value set over the whole
dataset is exactly `{0, 1, 2, 3}`, with source geotiff `nodata = 3`:

| source value | meaning        | output class id |
|--------------|----------------|-----------------|
| 0            | background     | 0               |
| 1            | field          | 1               |
| 2            | field_boundary | 2               |
| 3            | nodata/unlabeled | 255 (CLASS_NODATA) |

The window `metadata.json` gives `projection`, `bounds`, and a `time_range` (verified to
be exactly 240 days for every sampled window — well under 1 year), which we use directly.

## Processing

The manifest `label_type` is `polygons`, but on disk the labels are already
pre-rasterized into a dense categorical raster, so this is processed as a
**dense-raster classification** task (recipe §4 dense_raster, §5 tiles-per-class balanced):

1. **Sample windows** for geographic diversity: up to `WINDOWS_PER_GROUP = 300` windows
   per country (seeded random; smaller countries fully included). 7,020 windows scanned.
2. **Tile** each window into ≤64×64 patches. Tiles are edge-aligned full 64×64 blocks
   covering the whole window (the last tile in each axis is aligned to the window edge and
   may overlap the previous one). 42,203 candidate tiles collected.
3. **Read natively, no reprojection.** The source is already UTM at 10 m/pixel, so the
   label is read at its native projection/bounds (an identity read via rasterio — no
   interpolation of the categorical labels, equivalent to nearest resampling). Source
   value 3 is remapped to 255.
4. **Tiles-per-class balanced selection** (≤1000 tiles per class): a tile counts toward
   every class present in it; the rarest under-target class is served greedily. Tiles that
   are entirely nodata are dropped.
5. Each tile is written as a single-band uint8 GeoTIFF (10 m UTM, nodata 255) plus sidecar
   JSON carrying the parent window's `time_range`, `source_id`
   (`<country>/<window>#<row_off>_<col_off>`), and `classes_present`.

## Outputs

- `datasets/olmoearth_fields_of_the_world/metadata.json`
- `datasets/olmoearth_fields_of_the_world/locations/{000000..001163}.tif` + `.json`
- `raw/olmoearth_fields_of_the_world/SOURCE.txt` (pointer to the local source; not copied)

**num_samples = 1164.** Per-class tile counts (tiles-per-class semantics; a tile counts
toward each class it contains):

| class          | id | tiles |
|----------------|----|-------|
| background     | 0  | 1001  |
| field          | 1  | 1059  |
| field_boundary | 2  | 1000  |

`field` slightly exceeds 1000 because it co-occurs in almost every field/boundary tile and
rides along while background/boundary are still being filled — expected under
tiles-per-class balancing. All 25 countries are represented in the selected samples
(e.g. cambodia 103, vietnam 87, brazil 73, luxembourg 63, croatia 56, netherlands 56,
plus every other group; smallest: portugal 7, rwanda 14).

## Verification

- All 1164 tifs: single-band uint8, UTM (EPSG:326xx/327xx), 10 m, exactly 64×64, values
  ⊆ {0,1,2,255}, no invalid values. 1:1 tif↔json pairing.
- All sample JSONs carry a 240-day (≤1 yr) `time_range`; `pixel_bounds` match tif shape.
- **Georef/value round-trip** (8 random samples across countries): the written tile equals
  the source window's label slice at the recorded pixel offsets (with 3→255), and the
  tile's top-left geo-coordinate equals the source pixel geo-coordinate — exact match.
- **S2 co-location** is exact by construction: labels inherit the projection/bounds of the
  same source windows that hold the ingested Sentinel-2 imagery.

## Caveats

- `field_boundary` is a thin 1–2 px class at 10 m; it is present in nearly every
  field-containing tile, so it is not truly rare — the 1000-tile cap is easily reached.
- Only a diverse subsample of windows is used (300/country cap); this is intentional
  (target is ≤1000/class), not global coverage. Increase `WINDOWS_PER_GROUP` to draw more.

## Reproduce

```bash
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.olmoearth_fields_of_the_world
```

Idempotent: existing `locations/{id}.tif` are skipped on re-run.
