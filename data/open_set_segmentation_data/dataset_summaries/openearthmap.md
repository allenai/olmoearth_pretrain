# OpenEarthMap — COMPLETED

- **Slug**: `openearthmap`
- **Name**: OpenEarthMap
- **Source**: Zenodo record 7223446 (`OpenEarthMap.zip`, 9.1 GB) / Xia et al., WACV 2023
- **Project page**: https://open-earth-map.org/ · Paper: https://arxiv.org/abs/2210.10732
- **Family / region**: land_cover / Global (97 regions, 44 countries, 6 continents)
- **Label type**: dense_raster, VHR 0.25–0.5 m, manual photointerpretation
- **License**: CC-BY-NC-SA-4.0 (label data; per Zenodo, label license follows each source image)
- **Task type**: classification (8-class land cover)
- **Status**: **completed** — **1704** label tiles

## What OpenEarthMap is

A benchmark for global high-resolution land cover mapping: 5000 aerial/satellite images
with manually annotated 8-class land cover masks (2.2 M segments) at 0.25–0.5 m GSD. The
public Zenodo archive (`OpenEarthMap_wo_xBD/`) omits the xBD-sourced RGB imagery for
licensing but **keeps all 3500 non-xBD label masks** (1024×1024 uint8), laid out as
`OpenEarthMap_wo_xBD/{region}/labels/{region}_N.tif`. We only need the labels — pretraining
supplies its own S2/S1/Landsat imagery.

## Georeferencing gate (§8.2) — PASSED

Unlike LoveDA (coordinate-free PNG → rejected), every OpenEarthMap label `.tif` carries a
real CRS + geotransform. The CRS varies by source region — local UTM zones, EPSG:3857
(Web Mercator), EPSG:4326 (geographic), and national/custom grids (e.g. EPSG:31256, a
custom WKT Transverse-Mercator) — all with genuine real-world coordinates. Verified across
kagera, accra, chisinau, coxsbazar, houston, jeremie, pomorskie, rotterdam, santa_rosa,
shanghai, vienna, and end-to-end after processing: reprojected tile centroids land exactly
on their named regions (vegas → Las Vegas −115.16/36.21; baybay → Philippines 124.83/10.67;
western → Ghana −2.19/6.08; coxsbazar → Bangladesh 92.19/21.09).

## Access method

Public, no credentials. `download.download_zenodo("7223446", raw_dir, filenames=["OpenEarthMap.zip"])`.
Raw zip kept at `raw/openearthmap/OpenEarthMap.zip`; label masks are read directly from the
zip (imagery members never decoded). Scan cache at `raw/openearthmap/scan_cache.pkl`.

## Class mapping

Source uint8 value → output id (source 0 = "unknown" → nodata 255):

| out id | class            | src val |
|--------|------------------|---------|
| 0 | bareland            | 1 |
| 1 | rangeland           | 2 |
| 2 | developed space     | 3 |
| 3 | road                | 4 |
| 4 | tree                | 5 |
| 5 | water               | 6 |
| 6 | agriculture land    | 7 |
| 7 | building            | 8 |

nodata / ignore = 255.

## VHR-at-10 m handling (§4)

Each 0.25–0.5 m mask (256–512 m footprint) is reprojected from its native CRS to a local
UTM grid at **10 m with MODE resampling** (categorical majority; never bilinear), producing
**one ~17–55 px tile per source mask** (max observed dimension 55, all ≤ 64 — no sub-tiling
needed). Fill outside the source footprint = 0 → nodata.

**Class-set suitability decision: all 8 classes kept, none dropped or merged.** OpenEarthMap's
scheme is already a coarse land-cover taxonomy (much coarser than FLAIR/LoveDA's fine
classes), so it survives 10 m resampling well. The two finest classes — **road (3)** and
**building (7)** — are only *partially* resolvable at 10 m: mode resampling preserves them
where they form contiguous majorities (dense urban blocks, wide highways) but folds isolated
buildings and narrow rural roads into the surrounding class. They are **retained** (both are
well-populated: road 1544 tiles, building 1582 tiles) rather than dropped; downstream
assembly filters any class that ends up too sparse.

## Time-range handling

The release provides **no per-tile acquisition date**; imagery spans ~2016–2023 from mixed
VHR sources. Land cover is treated as a persistent/static label (task spec §5 static-label
rule) and assigned a representative 1-year Sentinel-era window, **2020-01-01 → 2021-01-01**.
No change labels. Caveat: a minority of source regions derive from disaster events some of
which predate 2016 (e.g. xBD-region label masks are present even though their RGB is omitted);
because land cover class (building/road/tree/water) is persistent, these still pair sensibly
with post-2016 S2. Not all labels are pre-2016, so the §8.2 pre-2016 rule does not apply.

## Sampling

dense_raster, **tiles-per-class balanced**, ≤1000 tiles/class, rarest-class-first, capped at
25 000 total (well under). A tile counts toward every class it contains, so common
co-occurring classes overshoot 1000. All source splits (train/val/test) used.

Selected **1704** tiles of 3500 masks. Per-class tile counts:

| class | tiles |
|-------|-------|
| bareland         | 544 (rare — all available; limiting class that stops selection) |
| rangeland        | 1670 |
| developed space  | 1585 |
| road             | 1544 |
| tree             | 1671 |
| water            | 1140 |
| agriculture land | 1000 (hit per-class cap) |
| building         | 1582 |

Selection stops once the only below-cap class (bareland) is exhausted. bareland is the
one sparse class — noted; downstream filtering handles too-small classes.

## Verification (§9)

- 1704 `.tif` + 1704 `.json`; all single-band uint8, UTM CRS @ 10 m, max dim 55 (≤64),
  values ∈ {0..7, 255}, nodata=255.
- Every `.tif` has a matching `.json` with a 1-year `time_range`, `change_time=null`,
  `classes_present`, and `source_id` (`{region}/{tile}`).
- metadata.json class ids 0–7 cover all values in the tifs.
- Spatial sanity: reprojected centroids match named regions (see georeferencing section).

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.openearthmap
```

Idempotent (skips existing `locations/{id}.tif`; caches the reprojection scan). Downloads
the Zenodo zip if absent.
