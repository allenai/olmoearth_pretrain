# Munich480 / MTLCC — dataset summary

- **Slug:** `munich480_mtlcc`
- **Task type:** classification (per-pixel crop type), `dense_raster`
- **Status:** completed — **5,302** label tiles, 17 classes
- **Region / time:** ~102 × 42 km area north of Munich, Bavaria, Germany; 2016 & 2017 growing seasons.

## Source

MTLCC — Rußwurm & Körner (2018), *"Multi-Temporal Land Cover Classification with Sequential
Recurrent Encoders"*, ISPRS IJGI 7(4):129. Repo: <https://github.com/TUM-LMF/MTLCC>. Data
on Zenodo record **5712933** (CC-BY-4.0). Crop labels are Bavarian farmer declarations
(IACS / STMELF).

### Access method (label-only, no bulk download)

The canonical MTLCC training release ships as 24/48-px **TFRecord tensor tiles** in the
42 GB `data_IJGI18.zip`; those tiles need a separate `geotransforms.csv` to recover
geolocation. Rather than pull 42 GB to extract a thin label layer, we take the
**georeferenced ground-truth crop-parcel shapefiles** bundled in the 1.4 GB `showcase.zip`
on the same Zenodo record. We extract **only** `fields16.{shp,shx,dbf,prj}`,
`fields17.{shp,shx,dbf,prj}`, and `classes.csv` (~120 MB total) via **HTTP range requests**
into the remote zip's central directory (`download.HttpRangeFile` + `zipfile`) — no
full-archive download, and imagery is not fetched (pretraining supplies its own).

- `fields16.shp` — 90,181 crop parcels, 2016.
- `fields17.shp` — 89,115 crop parcels, 2017.
- Attributes: `if` (field id), `labelid` (source crop id, 1..26 non-sequential), `label`
  (crop name), `doystart`/`doyend` (unused, all NaN in the release).
- CRS: **EPSG:32632 (WGS84 / UTM zone 32N)**, coordinates in metres — **fully
  georeferenced natively**, no CRS-recovery needed (the georeferencing concern flagged for
  some MTLCC `.npy`/TFRecord releases does not apply to these shapefiles).

## Georeferencing check

The parcels carry a real UTM 32N `.prj`. Written tiles land at ~48.5° N, 10.8–11.7° E
(directly north of Munich, 48.14° N / 11.58° E), confirming correct placement.

## Label / class mapping

17 crop classes, taken in `classes.csv` order and remapped to 0-based ids (`class_id` =
row index; matches the MTLCC `labid → dimid` lookup):

| id | name | source labelid |
|----|------|----------------|
| 0 | sugar beet | 1 |
| 1 | summer oat | 2 |
| 2 | meadow | 3 |
| 3 | rape | 5 |
| 4 | hop | 8 |
| 5 | winter spelt | 9 |
| 6 | winter triticale | 12 |
| 7 | beans | 13 |
| 8 | peas | 15 |
| 9 | potatoe | 16 |
| 10 | soybeans | 17 |
| 11 | asparagus | 19 |
| 12 | winter wheat | 22 |
| 13 | winter barley | 23 |
| 14 | winter rye | 24 |
| 15 | summer barley | 25 |
| 16 | maize | 26 |

All parcel `labelid`s in both years fall within this 17-class set (no extra/other classes to
drop). Well under the 254-class uint8 cap.

## Processing recipe (`dense_raster`)

This is a genuinely dense multi-class crop map (each 640 m window holds many adjacent
fields), so instead of one-tile-per-parcel we:

1. Rasterize **all** parcels of a year onto a single UTM-32N 10 m label array covering the
   region bounds (`rasterio.features.rasterize`, `all_touched=False`, fill = **255**).
   Only declared fields carry ground truth; unlabeled land (forest, urban, water, roads) is
   **255 = ignore** — there is no background class and no synthetic negatives (assembly-time
   negatives per spec §5). ~47.5 % of the region is labeled.
2. Cut the array into non-overlapping **64 × 64 (640 m)** tiles; keep tiles with ≥1 labeled
   pixel (9,771 tiles in 2016; 9,773 in 2017; 19,544 candidates).
3. **Tiles-per-class balanced** selection (`sampling.balance_tiles_by_class`, rarest class
   first) with `per_class=1000` and the 25k per-dataset cap → **5,302** tiles selected.

Both years are included; a tile at the same location in 2016 and 2017 is two independent
samples (different crop / different 1-year window). `source_id` = `fields{16,17}/tile_<col>_<row>`.

## Output tiles

- Single-band **uint8** GeoTIFFs, EPSG:32632, 10 m, 64 × 64, nodata/ignore = **255**.
- Verified: all pixel values across the dataset ∈ {0..16, 255}; every `.tif` has a matching
  `.json`; all `time_range`s are ≤ 1 year.

### Time range

1-year window anchored on each tile's labeled year: 2016 tiles → `[2016-01-01, 2017-01-01)`,
2017 tiles → `[2017-01-01, 2018-01-01)`. Both post-2016 (2016 explicitly allowed). Static
seasonal crop labels → `change_time = null`.

### Class counts (tiles containing each class; a tile counts toward every class present)

sugar beet 1000, summer oat 1152, meadow 3733, rape 1959, hop 1022, winter spelt 1047,
winter triticale 1221, beans 1075, peas 994, potatoe 1697, soybeans 1050, asparagus 790,
winter wheat 4572, winter barley 3435, winter rye 1017, summer barley 1463, maize 4889.

Two classes fall short of the 1000 target because too few distinct 640 m tiles contain them
(asparagus 790, peas 994) — kept anyway per spec §5 (sparse classes are not dropped;
downstream assembly handles minimum-count filtering).

## Caveats

- Labels are the parcels used in the MTLCC benchmark (farmer-declared, per-parcel constant
  crop); intra-parcel variation is not represented.
- Only declared agricultural fields are labeled; all other land cover is ignore (255).
- Tiles do not overlap (stride = tile size), so total labeled area is sub-sampled by the
  25k-cap balancer, prioritizing rare crops.

## Reproduce

```bash
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.munich480_mtlcc
```

Idempotent: skips already-written `locations/{id}.tif`. Raw label shapefiles are cached
under `raw/munich480_mtlcc/` (selective HTTP-range extraction from `showcase.zip`).
