# Chesapeake Land Cover — processing summary

- **Slug**: `chesapeake_land_cover`
- **Manifest name**: Chesapeake Land Cover
- **Status**: **completed**
- **Task type**: classification (per-pixel land cover)
- **Num samples**: **2413** label tiles (64×64, 10 m, local UTM GeoTIFFs)
- **Family / region**: land_cover / US Chesapeake Bay watershed (DE, MD, VA, WV, NY, PA, DC)

## Source used (and a deliberate substitution)

The manifest URL points at the **LILA BC "Chesapeake Land Cover"** release
(`s3://us-west-2.opendata.source.coop/agentmorris/lila-wildlife/lcmcvpr2019/cvpr_chesapeake_landcover`,
the CVPR-2019 dataset). That release is a **6-class** land-cover product derived from
**2013/2014 NAIP** — the tile directories are literally `de_1m_2013 … va_1m_2014`, and the
`_lc.tif` labels are 1 m 6-class. **Those labels are entirely pre-2016**, i.e. outside the
Sentinel-2 era. The spec rejects datasets whose labels are all pre-2016.

However, the manifest's own `time_range` is **[2016, 2022]** and it explicitly notes
**"13/54-class LULC variants"** — both of which describe the *newer* product, the **USGS /
Chesapeake Bay Program "Chesapeake Bay Land Use and Land Cover (LULC) Database, 2022
Edition"** (DOI `10.5066/P981GV1L`, ScienceBase item `633302d8d34e900e86c61f81`). That
database contains one-meter **13-class Land Cover (LC)** and 54-class LULC for **two
epochs, 2013/14 and 2017/18**. I therefore used the **2017/18 13-class LC** epoch, which is
fully post-2016 and matches the manifest's intent, instead of the pre-2016 LILA tiles.

- **Access**: public USGS data release, **no credentials**. Per-state LC 2017/18 zips
  downloaded via `https://www.sciencebase.gov/catalog/file/get/<state_item_id>?name=<file>.zip`
  (the `manager/download/...` URLs return an SPA shell — the `catalog/file/get` form is the
  working one). Data dictionary (13-class legend, Table 6) at `raw/.../sciencebase/data_dictionary.pdf`.
- **Format**: per-state single-band 1 m GeoTIFF, **ESRI:102039** (USA Contiguous Albers
  Equal Area Conic), uint8, nodata 255. Downloaded 7 states (DE, MD, VA, WV, NY, PA, DC).
- **License**: public domain (U.S. Government work / USGS data release).
- **Annotation method**: 1 m LC produced by the Chesapeake Conservancy / UVM Spatial
  Analysis Lab / USGS via eCognition supervised classification of NAIP + lidar with manual
  QA / photointerpretation. This is high-quality reference-grade mapping.

## Class mapping (13-class LC → 12 output ids)

Source raster values 1–12 → output ids 0–11 (verbatim legend + descriptions from the data
dictionary go into `metadata.json`). Source value **254 = "Aberdeen Proving Ground"** (an
unmapped U.S. Army facility in Harford County, MD — no LC assigned) and **255 = NoData** are
both mapped to **nodata 255**. So there are **12 real classes**:

| id | name | id | name |
|----|------|----|------|
| 0 | Water | 6 | Impervious Structures |
| 1 | Emergent Wetlands | 7 | Other Impervious |
| 2 | Tree Canopy | 8 | Impervious Roads |
| 3 | Scrub/Shrub | 9 | Tree Canopy Over Structures |
| 4 | Low Vegetation | 10 | Tree Canopy Over Other Impervious |
| 5 | Barren | 11 | Tree Canopy Over Impervious Roads |

## Processing recipe (dense_raster, VHR-native → 10 m tiles)

1. Each state's 1 m Albers raster is reprojected to a **local UTM zone at 10 m** (chosen from
   the state centroid's lon/lat) using a rasterio **WarpedVRT with `Resampling.mode`**
   (majority — never bilinear, since the label is categorical). The VRT transform is snapped
   to a multiple of 10 m so tiles align exactly to the rslearn Projection pixel grid.
2. Random **64×64** tile positions are probed across each state's VRT (12,000 probes/state).
   A tile is kept only if **≥50 % of its pixels are labeled** (not nodata) and its **true UTM
   zone equals the state-VRT zone** (out-of-zone tiles — mostly the non-watershed western
   tails of VA/WV/PA — are dropped, which also focuses sampling on the Chesapeake/eastern
   watershed and guarantees every written tile is in its correct local UTM).
3. **Tiles-per-class balanced** selection (spec §5): a tile counts toward every class present
   in it; classes are filled **rarest-first** up to **1000 tiles/class**, 25 k total cap.
4. GeoTIFFs (uint8, nodata 255) + per-sample JSON written; idempotent (skips existing tifs).

**Time range**: 1-year window on each state's LC epoch year — **2018** for DE/MD/VA/WV,
**2017** for NY/PA/DC (matches the 2017/18 NAIP source year per the data dictionary). Land
cover is a quasi-static annual label, so a 1-year window is appropriate. No change labels.

## Sample counts

Candidate tiles (≥50 % labeled, in-zone): **19,166** from 73,440 probes across 7 states.
Final selected: **2413** tiles. Per-class tile counts (a tile counts toward every class in it):

| id | class | tiles | id | class | tiles |
|----|-------|------:|----|-------|------:|
| 0 | Water | 1599 | 6 | Impervious Structures | 1889 |
| 1 | Emergent Wetlands | 1136 | 7 | Other Impervious | 1994 |
| 2 | Tree Canopy | 2331 | 8 | Impervious Roads | 1895 |
| 3 | Scrub/Shrub | 1196 | 9 | Tree Canopy Over Structures | 1061 |
| 4 | Low Vegetation | 2319 | 10 | Tree Canopy Over Other Impervious | 1229 |
| 5 | Barren | 1000 | 11 | Tree Canopy Over Impervious Roads | 1315 |

Every class reached ≥1000 tiles (common classes overshoot because they co-occur in tiles
selected for rarer classes). No class was dropped; no 254-class cap issue (only 12 classes).

## Caveats / judgment calls

- **Substituted the source product** (2017/18 USGS 13-class LC) for the manifest's LILA URL
  (2013/14 6-class), because the latter is entirely pre-2016. This is the key decision;
  documented above and in `raw/.../SOURCE.txt`. The class scheme is finer (12 vs 6) and
  post-2016, matching the manifest `time_range`/notes.
- **10 m suitability**: Water, Emergent Wetlands, Tree Canopy, Scrub/Shrub, Low Vegetation,
  Barren, and the broad impervious classes map well at 10 m. The **thin / overlap classes —
  Impervious Roads and the three "Tree Canopy Over …" classes** — rarely survive as the
  majority of a 10×10 m block, so their 10 m tiles are noisier and biased toward wider road
  corridors / denser overhang. They are **kept** per spec (downstream assembly filters
  too-small classes); flagged here as low-confidence at 10 m.
- **Per-state UTM zone** (not strictly per-tile): each state is reprojected to one UTM zone
  (its centroid's); tiles whose true zone differs are dropped rather than distorted, so all
  written tiles are exactly georeferenced in their correct zone (18N for most; WV/western
  areas 17N).
- **Bounded sampling**: this is a ~250,000 km² 1 m product; I did not process it exhaustively.
  I probed random windows across all 7 downloaded states to reach the per-class targets. PA
  and VA rasters are multi-GB but only windowed reads were used.
- **Verification**: 5 random tifs confirmed single-band uint8, UTM @ 10 m, ≤64×64, values in
  0–11 + 255, matching JSON with a 1-year `time_range`; all `classes_present` in range.
  Georef cross-check: tile centers sampled back against the source Albers raster agreed
  29/40 at the exact center pixel, the remainder being expected mode-vs-single-pixel
  differences at class boundaries (e.g. a 1 m road pixel vs the 10 m majority); all tile
  centers fell in sensible watershed locations with matching geography (water on water,
  coastal emergent wetlands on the DE shore).

## Reproduce

```
# 1) download per-state 2017/18 LC zips from ScienceBase item 633302d8d34e900e86c61f81 into
#    raw/chesapeake_land_cover/sciencebase/ and unzip each .tif to <state>_lc/ (see SOURCE.txt)
# 2) run:
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.chesapeake_land_cover
```
