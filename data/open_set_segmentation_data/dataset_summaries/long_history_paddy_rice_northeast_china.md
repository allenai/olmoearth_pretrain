# Long-history Paddy Rice, Northeast China

- **Slug**: `long_history_paddy_rice_northeast_china`
- **Task type**: classification (binary paddy-rice presence)
- **Status**: completed — 2000 samples (1000 paddy rice, 1000 non-paddy)
- **License**: CC-BY-4.0

## Source

figshare / ESSD: *Long history paddy rice mapping across Northeast China with deep
learning and annual result enhancement method.* The project has **two** figshare records:

- `10.6084/m9.figshare.28283606` — the **manifest DOI**. This is only a small
  **training-set sample**: 50 Landsat image/mask pairs (256×256, 30 m, EPSG:32653). On
  inspection only **5 of 50 masks** contain any paddy-rice pixels (9,045 rice px total);
  45 masks are all-background. Far too few rice samples for a balanced dataset.
- `10.6084/m9.figshare.27604839` — the companion **annual maps** (the "rasters" the
  manifest refers to): one 30 m paddy-rice presence GeoTIFF per year, 1985–2023
  (2012 missing), ~56 MB each, EPSG:32653 (UTM 53N).

We therefore used the **companion annual maps** (27604839), which are the proper
georeferenced dense rasters, rather than the sparse training sample.

Raster encoding (per year): `0 = non-paddy (observed land)`, `1 = paddy rice`,
`3 = nodata (outside study area)`.

## Access

Public figshare download, no credentials. We downloaded a single representative year
(`2020.tif`, file id 50181699) to
`raw/long_history_paddy_rice_northeast_china/2020.tif` via `download.download_http`.

## Labeled year / time range

The maps span 1985–2023; we sampled one representative Sentinel-era year, **2020**
(within the manifest 2016–2023 range). Each sample gets a **1-year** time range
`[2020-01-01, 2021-01-01)`. `change_time` is null — this is annual presence
classification, not a dated event. The per-tile Landsat acquisition year cannot be
recovered from an annual composite, so the single 2020 window is applied to all tiles.

## Method (dense_raster, bounded tiles-per-class balanced)

Regional derived-product map → bounded-tile sampling:

1. **Scan** the 2020 raster (56945×60922, 128×128 LZW tiled) in horizontal bands
   (`multiprocessing.Pool(64)`), dividing each into `21×21` native blocks
   (21 px × 30 m ≈ 630 m ≈ one 64 px @ 10 m output tile).
2. For each block requiring ≥90% observed (valid) pixels, classify as:
   - **paddy rice** if ≥50% of observed pixels are rice (strong majority / high
     confidence), or
   - **non-paddy** if the block has **zero** rice pixels (pure observed land).
   Reservoir-sample within each band to bound memory.
3. Randomly select up to **1000 tiles per class** (seed 42).
4. **Write**: reproject each selected block from EPSG:32653 30 m to **local UTM at 10 m**
   (nearest resampling — categorical), centered on the block's WGS84 center, producing a
   **64×64** single-band uint8 patch. Native ids kept (0/1); non-{0,1} → **255 nodata**.

Output tiles span UTM zones 50N–53N (per-tile local UTM), covering Northeast China
(lon 115.6–134.7 E, lat 38.8–53.5 N).

## Classes

| id | name | count |
|----|------|-------|
| 0 | non-paddy | 1000 |
| 1 | paddy rice | 1000 |

255 = nodata/ignore.

## Verification

- All 2000 `.tif`: single-band, uint8, 64×64, 10 m, local UTM, nodata 255; values in
  {0,1,255}. 2000 matching `.json`, each with a 1-year `time_range`, `change_time`=null.
- Tile values: 1980 tiles contain non-paddy(0), 1001 contain rice(1) (one non-paddy tile
  picked up a single border rice pixel via nearest resampling — negligible), 5 tiles have
  a few 255 pixels at edges.
- **Spatial sanity**: tile centers fall in Northeast China; **rice tiles cluster around
  (126.7 E, 44.7 N)** — the Sanjiang/Songnen Plain paddy belt — as expected. Full
  Sentinel-2 overlay not run; georeferencing verified exact from the source raster's
  own CRS/transform (EPSG:32653, 30 m) and preserved through reprojection.

## Caveats

- Coarse native resolution (30 m Landsat-derived) upsampled to 10 m; labels are blocky at
  10 m but footprints are correct and adequate for pairing with 10 m S2/S1 imagery.
- Derived-product map (not in-situ reference); mitigated by sampling only
  high-confidence homogeneous blocks (≥50% rice or pure non-rice, ≥90% observed).
- Single labeled year (2020) used for all tiles; other years (1985–2023) are available
  from the same figshare record if temporally diverse sampling is later desired.

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.long_history_paddy_rice_northeast_china
```
Idempotent (skips already-written `{id}.tif`). Optional `--workers N` (default 64).
