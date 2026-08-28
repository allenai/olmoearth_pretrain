# LandCover.ai

- **Slug:** `landcover_ai`
- **Status:** completed
- **Task type:** classification (dense land-cover segmentation)
- **Samples:** 656 tiles (64×64, 10 m)
- **Family/region:** land_cover / Poland
- **License:** CC-BY-NC-SA-4.0

## Source

LandCover.ai (Land Cover from Aerial Imagery), Boguszewski et al., CVPR EarthVision 2021.
Distributed as a single public HTTP zip, no account/credential required:
`https://landcover.ai.linuxpolska.com/download/landcover.ai.v1.zip` (~1.5 GB). The archive
holds 41 three-channel RGB orthophotos of urban/rural Poland (33 at 0.25 m, 8 at 0.5 m;
~216 km²) under `images/` and their matching single-channel land-cover masks under
`masks/` (uint8, identical georeferencing), plus `split.py` and train/val/test lists.

## Access method

Only the **masks** are needed (pretraining supplies its own imagery). Rather than pulling
the full 1.5 GB, the script opens the remote zip with `fsspec` + `zipfile` and extracts
only the small `masks/*.tif` members via HTTP range reads (41 files, ~30 MB total) to
`raw/landcover_ai/masks/`. The RGB orthophotos are never downloaded. `raw/.../SOURCE.txt`
records the provenance. Extraction is idempotent (skips masks already on disk).

## Georeferencing / CRS

Masks are stored in a **WGS84-based Transverse Mercator** (central meridian 19°,
scale 0.9993, false easting 500000, false northing −5300000 — the Poland CS92 / PUWG-1992
grid written against a WGS84 datum; the website calls it EPSG:2180). Coordinates are in
metres. Each mask carries a valid affine transform, so tile georeferencing is inherited
exactly from the source (verified: sampled tile centers land at 15–18°E / 51–53°N, i.e.
Poland). Tiles come out in local UTM zones 33N (EPSG:32633) and 34N (EPSG:32634).

## Class mapping

Source mask values are kept unchanged (already 0-based), 5 classes → uint8, nodata 255:

| id | name       | source value | notes |
|----|------------|--------------|-------|
| 0  | background | 0 | residual/other surfaces (none of the four labeled types) |
| 1  | building   | 1 | under-resolved at 10 m |
| 2  | woodland   | 2 | resolves well |
| 3  | water      | 3 | resolves well |
| 4  | road       | 4 | under-resolved at 10 m |

Selected-tile class counts (a tile counts toward every class present in it):
`background 649, building 280, woodland 643, water 366, road 505`.

## VHR → 10 m handling (spec §4)

Each whole orthophoto (0.25/0.5 m) is reprojected to a local UTM grid at 10 m using
**mode** resampling (categorical majority; never bilinear), then cut into non-overlapping
64×64 (640 m) tiles. A reprojected validity mask marks out-of-footprint fill as **nodata
255**, so reprojection padding is never confused with real `background` (0). Partial edge
tiles are padded to 64×64 with nodata; tiles that are entirely nodata are dropped.

**Fine-class judgment:** at 10 m the two narrow classes are under-resolved — individual
buildings (~10–20 m) and roads (~5–10 m wide) only survive where they dominate a 10 m
pixel (dense urban blocks; wide roads/junctions; roads are long so they still touch many
tiles). Their counts are lower than woodland/water/background, but both classes were
**retained** per spec §5 (downstream assembly drops any class that ends up too small); the
under-resolution is documented rather than dropped.

## Time range

Per-file aerial acquisition dates are not published. The manifest gives a 2016–2018 window
(Sentinel era). Per spec §5 (static/seasonal land-cover), every tile gets a representative
static **1-year window: 2017-01-01 → 2018-01-01**. `change_time` is null (no change task).

## Sampling

One record per non-empty 64×64 tile. Tiles-per-class balanced (`sampling.
select_tiles_per_class`, ≤1000 tiles/class, rarest-first, ≤25,000 total). Total (656) is
far below every cap, so all non-empty tiles are kept. All 41 orthophotos (all source
splits) are used.

## Verification (spec §9)

- Opened multiple output `.tif`s: single band, uint8, UTM (32633/32634) at 10 m, 64×64,
  nodata 255, values ∈ {0,1,2,3,4,255}. ✓
- Every `.tif` has a matching `.json` with a 1-year `time_range`; `metadata.json` classes
  (0–4) cover all values in the tifs. ✓
- Tile centers verified inside Poland (15–18°E, 51–53°N); georeferencing inherited directly
  from source GeoTIFF transforms (no coordinate fabrication). ✓
- Idempotent: re-running skips existing `{sample_id}.tif` and already-extracted masks.

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.landcover_ai
```

## Caveats

- `building` and `road` are under-resolved at 10 m (see above); retained for downstream
  filtering to decide.
- Source CRS is a WGS84-datum variant of Poland CS92; treated as a valid projected CRS and
  reprojected via pyproj/rasterio to UTM — sub-pixel datum differences are negligible at
  10 m.
- Acquisition dates unknown per file; a single representative 2017 window is used for all
  tiles.
```
