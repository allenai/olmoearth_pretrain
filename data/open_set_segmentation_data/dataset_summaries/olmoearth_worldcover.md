# OlmoEarth WorldCover

- **Slug:** `olmoearth_worldcover`
- **Status:** completed
- **Task type:** classification (dense multi-class raster)
- **Samples:** 5,503 label GeoTIFFs
- **Source:** local rslearn dataset `/weka/dfive-default/rslearn-eai/datasets/worldcover` (`have_locally: true`; `raw/olmoearth_worldcover/SOURCE.txt` points at it — nothing copied).
- **License:** CC-BY-4.0.

## What the source is

An existing OlmoEarth eval rslearn dataset. It has 165,696 windows (single group
`20260109`), each a ~53×53 pixel UTM tile at 10 m/pixel, north-up, in its local UTM zone.
The `label_raster` layer holds a single-band land-cover label. Only a **central ~10×10
pixel block** (a 100 m reference plot) is labeled in each window; all surrounding pixels
carry the source `no_data` class (pixel value 0). The 10×10 plot is frequently multi-class,
so this is a genuine `dense_raster` label (written as GeoTIFFs, not a point table).

The source `label_raster` legend (13 entries, index = pixel value; index 0 = `no_data`):
`no_data, bare, burnt, crops, fallow/shifting cultivation, grassland, Lichen and moss,
shrub, snow and ice, tree, urban/built-up, water, wetland (herbaceous)`.

**Provenance note.** The manifest labels this "ESA WorldCover 11-class" (a derived
product). The legend actually present here (with `burnt` and `fallow/shifting cultivation`,
and 100 m plots) is the crowd-sourced **Geo-Wiki reference legend used to validate ESA
WorldCover** — i.e. reference plots rather than the raw map. The manifest's 11-class list
also mentions `mangrove`, which does not appear in this dataset's legend. Either way, the
on-disk `label_raster` layer is treated as ground truth and its own legend is used.

## Class mapping (output ids)

Source pixel value `v` → output id `v-1` for `v ∈ [1,12]`; source `no_data` (0) → **255**
(nodata/ignore). uint8, `nodata_value = 255`.

| id | name | id | name |
|----|------|----|------|
| 0 | bare | 6 | shrub |
| 1 | burnt | 7 | snow and ice |
| 2 | crops | 8 | tree |
| 3 | fallow/shifting cultivation | 9 | urban/built-up |
| 4 | grassland | 10 | water |
| 5 | lichen and moss | 11 | wetland (herbaceous) |

## Processing

- Scanned all 165,696 windows in parallel (`multiprocessing.Pool(64)`), reading each
  `label_raster/label/geotiff.tif`. 159,062 windows had a non-empty labeled plot
  (~6.6k were fully `no_data` and dropped).
- For each, cropped the labeled (non-nodata) bounding box (the central ~10×10 block),
  remapped values, and derived exact UTM pixel bounds from the source window's pixel bounds
  (`bounds` + crop offsets), so georeferencing is inherited exactly from the source window.
- **Tiles-per-class balanced** selection (`sampling.balance_tiles_by_class`, `per_class=1000`,
  rarest-class-first) → 5,503 tiles (well under the 25k cap). A tile counts toward every
  class present in it.
- Output: one single-band uint8 GeoTIFF per plot (10×10, local UTM, 10 m, nodata 255) plus
  a per-sample JSON with crs/pixel_bounds/time_range/classes_present/source_id.

### Tiles available vs. selected, per class

| id | class | available | selected |
|----|-------|-----------|----------|
| 0 | bare | 28,949 | 1,000 |
| 1 | burnt | 252 | 252 (all) |
| 2 | crops | 26,823 | 1,034 |
| 3 | fallow/shifting cultivation | 1,971 | 1,015 |
| 4 | grassland | 93,833 | 1,874 |
| 5 | lichen and moss | 1,070 | 1,001 |
| 6 | shrub | 72,677 | 2,027 |
| 7 | snow and ice | 580 | 580 (all) |
| 8 | tree | 80,856 | 1,999 |
| 9 | urban/built-up | 9,788 | 1,063 |
| 10 | water | 10,569 | 1,009 |
| 11 | wetland (herbaceous) | 8,012 | 1,083 |

`burnt` (252) and `snow and ice` (580) are sparse — all available tiles kept; downstream
assembly drops any class below its minimum-count threshold (do not treat sparse classes as
a defect). Selected counts exceed 1,000 for some classes because multi-class tiles selected
to satisfy a rare class also add to the common classes they contain.

## Time range & change handling

No change labels. Each sample uses its **source window's own ~1-year time range** (all are
2016-01-01 → 2016-12-31, ≤1 year). The manifest lists 2020–2021 for the WorldCover product,
but the eval windows are anchored on 2016 imagery; land cover is near-static so the small
label-vs-image year offset is immaterial. All post-2016 (Sentinel era) — no pre-2016
filtering needed. All source splits (train/val/test) are used.

## Verification

- Opened output tifs: single-band uint8, local UTM (e.g. EPSG:32601), 10 m/pixel, 10×10,
  nodata 255; pixel values across a 500-tif sample = {0..11, 255}, all valid ids.
- Every `.tif` has a matching `.json` with a ≤1-year `time_range` and `classes_present`;
  `metadata.json` class ids (0–11) cover all values in the tifs.
- Spatial sanity: a `crops` plot overlaid on the source S2 mosaic gave NDVI 0.41 / NDWI
  -0.54 (vegetation) — sensible. (One `water` plot's source S2 base mosaic layer was empty
  for that window — a source-imagery gap, not a label issue; pretraining supplies its own
  imagery and the label georeferencing is inherited exactly from the source window.)
- Re-running is idempotent: the scan is deterministic (seeded, stable-ordered selection)
  and the write step skips samples whose `.tif`+`.json` already exist.

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.olmoearth_worldcover --workers 64
```

## Caveats

- Reference/derived-product legend differs from the manifest's stated ESA WorldCover
  11-class list (extra `burnt`, `fallow/shifting cultivation`; no `mangrove`). Used the
  on-disk legend.
- Label vs. imagery year offset (2016 windows for a 2020/2021-named product); acceptable
  for near-static land cover.
- Sparse classes (`burnt`, `snow and ice`) kept in full; downstream min-count filtering
  applies.
