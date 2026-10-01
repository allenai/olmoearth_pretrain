# Five-Billion-Pixels / GID — COMPLETED (classification, dense_raster)

- **Slug**: `five_billion_pixels_gid`
- **Name**: Five-Billion-Pixels / GID
- **Source**: Xin-Yi Tong, Gui-Song Xia, Xiao Xiang Zhu, *Enabling country-scale land
  cover mapping with meter-resolution satellite imagery*, ISPRS J. Photogramm. Remote
  Sens. 196 (2023) 178–196. Project page:
  https://x-ytong.github.io/project/Five-Billion-Pixels.html
- **Family / region**: land_use / China
- **Label type**: dense_raster (per-pixel land cover), 24 classes + unlabeled
- **License**: free for research use (public, no credentials)
- **Task type**: classification
- **Samples produced**: **10,327** tiles (≤64×64, single-band uint8, local UTM @ 10 m)
- **Status**: **completed**

## What the dataset is

Five-Billion-Pixels (FBP) extends the Gaofen Image Dataset (GID / GID-15). It is 150
Gaofen-2 (GF-2) multispectral scenes (~4 m, ~6900×7300 px, ~5 billion labeled pixels)
distributed across China, each with a manually photointerpreted per-pixel land-cover
annotation in a 24-class system (plus class 0 = "unlabeled" for miscellaneous/unclear
areas). Categories follow Chinese Land-Use Classification (GB/T 21010-2017), adapted to
what is recognizable at 4 m.

## Access method

Public Google Drive, sibling folders under the project page. Only the label + geolocation
folders are used (the 16-bit and 8-bit GF-2 imagery folders are **not** downloaded — the
open-set pipeline supplies its own imagery):

- `Annotation__index/` → `{scene}_24label.png` (single-band uint8 class-index masks)
- `Coordinate_files/` → `{scene}.rpb` (per-scene RPC00B rational-polynomial coefficients)

Folders are **listed** at runtime with `gdown.download_folder(skip_download=True)` (no
hardcoded ids), then each file is fetched via the
`drive.usercontent.google.com/download?...&confirm=t` endpoint (added as
`download.download_gdrive_file` / `download.list_gdrive_folder`). This endpoint serves
public files reliably; gdown's `uc?id=` path returns "Cannot retrieve the public link …
have had many accesses" (a transient anonymous-quota throttle) after a burst and should be
avoided. 150 scenes have a matching PNG + RPC.

## Georeferencing (the crux)

**The distributed label masks are plain PNGs with no CRS/geotransform.** The authors
instead released per-scene `.rpb` files carrying the GF-2 RPC00B coefficients ("The
coordinate information … is now available"). Geolocation is recovered by:

1. Parsing the `.rpb` into a `rasterio.rpc.RPC`.
2. Warping the label grid to a **local UTM** projection at **10 m** with GDAL's RPC
   transformer (`rasterio.warp.reproject(..., rpcs=RPC, RPC_HEIGHT=HEIGHT_OFF,
   resampling=nearest)`), evaluated at the scene's mean height `HEIGHT_OFF` — **no external
   DEM**. **nearest** is used because labels are categorical (never bilinear).

Validated: image centres map to each RPC's nominal centre; scene footprints (~28 km) land
correctly in China; and the 150 recovered scene centroids span the whole country
(lon 82.7–127.1°E, lat 20.0–52.7°N), matching the paper's Fig. 1 distribution map.
**Caveat:** RPC-without-DEM geolocation for near-nadir GF-2 is accurate to ~tens of metres
(worse over rugged terrain). This is acceptable for approximate label↔imagery co-location
at 10 m and is the sanctioned metadata-recovery path; a per-tile Sentinel-2 overlay was not
performed because that sub-100 m uncertainty makes pixel-exact overlay non-diagnostic.

## Resolution handling (native 4 m → 10 m)

Each scene is warped straight from its native 4 m label grid to the 10 m UTM grid (≈2.5×
downsample, nearest), then cut into **non-overlapping ≤64×64 tiles**. The output grid is
snapped to integer 10 m rslearn pixels (`col*10`, `row*-10`) so `pixel_bounds` are exact.
Tiles with **< 50 % labeled** pixels are dropped (scene-edge/rotation gaps and unlabeled
areas warp to nodata). 150 scenes → 144,307 candidate tiles.

## Class mapping

Source index `0` (unlabeled) → **nodata 255**. Source indices `1..24` → output ids
`0..23` (order preserved from the dataset readme):

| id | name | id | name | id | name |
|----|------|----|------|----|------|
| 0 | industrial area | 8 | natural meadow | 16 | bareland |
| 1 | paddy field | 9 | artificial meadow | 17 | rural residential |
| 2 | irrigated field | 10 | river | 18 | stadium |
| 3 | dry cropland | 11 | urban residential | 19 | square |
| 4 | garden land | 12 | lake | 20 | road |
| 5 | arbor forest | 13 | pond | 21 | overpass |
| 6 | shrub forest | 14 | fish pond | 22 | railway station |
| 7 | park | 15 | snow | 23 | airport |

24 classes (well under the 254 uint8 cap; no classes dropped). Some fine urban classes
(overpass, railway station, square, stadium) sit near the 10 m resolution limit but are
kept per spec — downstream assembly discards classes that end up too rare.

## Time range & change handling

The distribution ships **no per-scene acquisition date** (only RPCs; no image metadata
XML). FBP GF-2 scenes are ~2015–2020 and land cover is quasi-static, so every tile is
assigned a single **representative Sentinel-era 1-year window (2018-01-01 … 2019-01-01)**,
documented as an approximation. `change_time = null` (not a change dataset).

## Sampling / balancing

Tiles-per-class balanced (`sampling.select_tiles_per_class`, rarest-class-first),
≤1000 tiles/class, capped at 25,000 total; a tile counts toward every class it contains.
Selected **10,327** tiles. All source train/val/test scenes are fair game (used all 150).

### Per-class tile counts (a tile counts toward every class present)

```
 0 industrial area     2209     8 natural meadow     1131    16 bareland          1089
 1 paddy field         1028     9 artificial meadow  1018    17 rural residential 2131
 2 irrigated field     4170    10 river              1602    18 stadium            263
 3 dry cropland        1009    11 urban residential  2491    19 square             481
 4 garden land         1114    12 lake               1000    20 road               4402
 5 arbor forest        1035    13 pond               1074    21 overpass           1023
 6 shrub forest        1047    14 fish pond          1072    22 railway station     668
 7 park                 413    15 snow                295    23 airport             271
```

Common classes exceed 1000 because they co-occur in tiles selected to satisfy rarer
classes. Rare classes (park, snow, stadium, square, railway station, airport) are all
retained.

## Outputs

- `datasets/five_billion_pixels_gid/metadata.json` — dataset metadata + 24-class map.
- `datasets/five_billion_pixels_gid/locations/{id}.tif` — single-band uint8 label patches,
  local UTM @ 10 m, ≤64×64, nodata 255.
- `datasets/five_billion_pixels_gid/locations/{id}.json` — per-sample CRS/pixel_bounds,
  1-year time range, `change_time=null`, `source_id` (`{scene}/r{row}_c{col}`),
  `classes_present`.
- `raw/five_billion_pixels_gid/` — downloaded index PNGs (134 MB) + `.rpb` RPCs + reprojected
  UTM label caches; `SOURCE.txt`.

## Verification (spec §9)

- 10,327 `.tif` each with a matching `.json`; all single-band uint8, EPSG:326xx/327xx
  (UTM), 10 m, ≤64×64, nodata 255.
- All pixel values ⊆ {0..23} ∪ {255}; metadata class ids cover every value seen.
- All time ranges ≤ 1 year; all tile centroids fall inside China (lon 82.7–127.1,
  lat 20.0–52.7).
- Idempotent: re-running skips existing tiles (finishes in seconds using the reproj cache),
  producing the identical 10,327-tile selection.

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.five_billion_pixels_gid
```
