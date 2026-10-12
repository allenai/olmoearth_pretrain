# Global Lakes and Wetlands Database (GLWD) v2

- **Slug:** `global_lakes_and_wetlands_database_glwd_v2`
- **Status:** completed
- **Task type:** classification (dense_raster)
- **Num samples:** 24,640 label tiles (64×64 @ 10 m, single-band uint8)
- **Classes:** 33 (ids 0–32); nodata = 255
- **License:** CC-BY-4.0

## Source

Lehner, B., Anand, M., Fluet-Chouinard, E., et al. (2025): *Mapping the world's inland
surface waters: an upgrade to the Global Lakes and Wetlands Database (GLWD v2)*. Earth Syst.
Sci. Data 17, 2277–2329. doi:10.5194/essd-17-2277-2025. Data on figshare
(doi:10.6084/m9.figshare.28519994, CC-BY-4.0), hosted by HydroSHEDS / WWF
(https://www.hydrosheds.org/products/glwd).

GLWD v2 is a global (excl. Antarctica, 84°N–56°S) **15 arc-second (~500 m at the equator)**
raster in **EPSG:4326** mapping **33 lake/river/wetland types** (plus dryland), derived by
fusing many input products for the ~1990–2020 period. It ships as 6 zipped files (Geodatabase
+ GeoTIFF for three products: `area_by_class_ha`, `area_by_class_pct`, `combined_classes`).

## Access method

No credentials required — public figshare download. We pulled only
`GLWD_v2_0_combined_classes_tif.zip` (~925 MB, figshare file id 54001814) and, from inside it,
extracted the **dominant-class raster `GLWD_v2_0_main_class.tif`** (uint8: 0 = inland pixel
without wetland, 255 = nodata, 1..33 = dominant wetland class within the pixel) plus the legend
CSV. The other five distributions (per-class fraction layers, geodatabases) were not needed.
Raw files + `SOURCE.txt` under
`raw/global_lakes_and_wetlands_database_glwd_v2/`.

## Class mapping

Output class id = **GLWD_ID − 1** (source values 1..33 → ids 0..32). Source value 0 (dryland /
non-wetland) and 255 (nodata) → **nodata 255** (no fabricated background, per spec §2/§5).

| id | GLWD_ID | class | id | GLWD_ID | class |
|----|---------|-------|----|---------|-------|
| 0 | 1 | Freshwater lake | 17 | 18 | Palustrine, seasonally saturated, forested |
| 1 | 2 | Saline lake | 18 | 19 | Palustrine, seasonally saturated, non-forested |
| 2 | 3 | Reservoir | 19 | 20 | Ephemeral, forested |
| 3 | 4 | Large river | 20 | 21 | Ephemeral, non-forested |
| 4 | 5 | Large estuarine river | 21 | 22 | Arctic/boreal peatland, forested |
| 5 | 6 | Other permanent waterbody | 22 | 23 | Arctic/boreal peatland, non-forested |
| 6 | 7 | Small streams | 23 | 24 | Temperate peatland, forested |
| 7 | 8 | Lacustrine, forested | 24 | 25 | Temperate peatland, non-forested |
| 8 | 9 | Lacustrine, non-forested | 25 | 26 | Tropical/subtropical peatland, forested |
| 9 | 10 | Riverine, regularly flooded, forested | 26 | 27 | Tropical/subtropical peatland, non-forested |
| 10 | 11 | Riverine, regularly flooded, non-forested | 27 | 28 | Mangrove |
| 11 | 12 | Riverine, seasonally flooded, forested | 28 | 29 | Saltmarsh |
| 12 | 13 | Riverine, seasonally flooded, non-forested | 29 | 30 | Large river delta |
| 13 | 14 | Riverine, seasonally saturated, forested | 30 | 31 | Other coastal wetland |
| 14 | 15 | Riverine, seasonally saturated, non-forested | 31 | 32 | Salt pan, saline/brackish wetland |
| 15 | 16 | Palustrine, regularly flooded, forested | 32 | 33 | Rice paddies |
| 16 | 17 | Palustrine, regularly flooded, non-forested |    |    |    |

## Sampling (bounded-tile, spec §5)

This is a **large global derived-product raster**, so we do **not** attempt global coverage.
The full `main_class` raster (86400×33600 uint8, ~2.9 GB) is streamed in latitude strips and
scanned on its native 15 arc-sec grid for **spatially-homogeneous 3×3 native blocks
(~1.5 km)** in which all 9 cells share a single dominant wetland class (§4 guidance: prefer
homogeneous/high-confidence windows for coarse derived products). Homogeneity over ~1.5 km
gives confidence that the reprojected 640 m (64×64 @ 10 m) output tile is genuinely that
single class despite the coarse source. Candidate block centers are **subsampled per class
with a fixed seed** (probability ∝ 1/total, giving global geographic spread — sampled tiles
land on every continent: verified spot-checks in the US Great Lakes, Caspian/Turkmenistan,
West Siberia, Canadian Shield/Arctic, India, Australia, Mozambique, Greece, Germany,
Argentina, Belarus), balanced **tiles-per-class** via `balance_by_class` (25000 // 33 = **757
per class**), then each is reprojected from EPSG:4326 to a local UTM projection at 10 m with
**NEAREST** resampling (categorical labels).

**Class counts:** all 33 classes = **757** except **Palustrine, regularly flooded, forested
(id 15)** = **416** — only ~416 homogeneous 1.5 km blocks exist globally for that rare class;
kept in full per spec §5 (rare classes are retained; downstream assembly filters too-small
ones).

## Time range and change handling

GLWD v2 is a **static** compilation of the recent (~1990–2020) inland-water/wetland state, not
a dated event. Per §5 static-label rule: `change_time = null`, and each sample's `time_range`
is a representative **1-year window on 2020** (2020-01-01 → 2021-01-01), which sits in the
Sentinel era and near the end of the compilation period. (The manifest's [2016] tag is nominal.)

## Tile size

64×64 @ 10 m (= 640 m), single-band uint8, local UTM, north-up. Because the source is ~500 m,
each 640 m output tile spans ≈1 native cell, so most tiles are (near-)uniform in class after
nearest resampling; some tiles at class boundaries contain 2–3 classes.

## Verification (spec §9)

- Opened multiple output tifs: all single-band, uint8, UTM CRS, 10 m res, 64×64, nodata 255. ✓
- 24,640 tifs each have a matching `.json`; every `time_range` is exactly 1 year, `change_time`
  null. ✓
- Values across a 500-tile random sample span ids 0–32 + 255; `metadata.json` class ids cover
  all observed values. ✓
- **Georeferencing round-trip:** for 12 random samples, tile-center → lon/lat → source GLWD
  `main_class` value equals (tile dominant label + 1) in **12/12** cases. ✓
- **Sentinel-2 overlay:** for a Freshwater-lake tile (lon −82.81, lat 42.52; S2 scene
  T17TLH 2020-07-03, 0.4% cloud) NDWI mean = 0.50, 100% of labeled pixels NDWI>0; for a
  Saline-lake tile (lon 51.73, lat 39.54; T39SWD 2020-09-24) NDWI mean = 0.32, 100% NDWI>0 —
  water labels sit on water. ✓
- Script is idempotent (skips existing `{sample_id}.tif`). ✓

## Caveats

- **Coarse label:** the 500 m label marks the **dominant** wetland type of each cell, not a
  >50% areal majority; each ~640 m output tile is essentially one native cell.
- We deliberately use the plain `main_class` raster, **not** the `main_class_50pct` layer
  (which restricts to pixels where total wetland extent > 50%). The 50% layer eliminates
  linear/sparse classes entirely (e.g. Small streams → 0 qualifying pixels), and we wanted the
  full 33-class taxonomy; the 3×3 homogeneity filter supplies spatial confidence instead.
- **Thematic overlap** with `gwl_fcs30_global_wetland_map_fine_classes` (30 m, 8 classes) and
  `peatmap` — but GLWD v2 offers a distinct, richer hydro-functional taxonomy (peatlands split
  by climate zone; riverine/palustrine split by flooding regime and forest cover; deltas;
  rice paddies) not present in those products.

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.global_lakes_and_wetlands_database_glwd_v2
```
(Optional `--workers N`, default 64. Downloads the ~925 MB combined-classes zip on first run;
idempotent thereafter.)
