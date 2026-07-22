# Supraglacial Lakes, Northeast Greenland

- **Slug:** `supraglacial_lakes_northeast_greenland`
- **Status:** completed
- **Task type:** classification (single foreground class, positive-only)
- **Samples:** 1000 label tiles (GeoTIFF)
- **Source:** Lutz, Bahrami, Braun (2024), *Supraglacial lake outlines over Northeast
  Greenland from 2016 to 2022 using deep learning methods based on Sentinel-2 imagery*,
  PANGAEA, https://doi.org/10.1594/PANGAEA.973251
- **License:** CC-BY-4.0 (usable)
- **Region:** Northeast Greenland — 79°N Glacier and Zachariæ Isstrøm (78.3–81.0°N, −31.2 to −20.9°E)

## Source

Supraglacial (surface) meltwater-lake polygon outlines over two NE Greenland outlet
glaciers, segmented from Sentinel-2 (native 10 m) with a U-Net during the April–September
melt seasons of 2016–2022. A polar cloud-detection model removed scenes with >10 % cloud.
Outlines are **direct model output, not manually corrected** — they may contain false
positives from topographic/cloud shadows and slushy light-blue ice on peak-melt days
(source caveat).

## Access method

PANGAEA serves this as a file collection of **seven annual zips** (`yyyy.zip`, 2016–2022).
Individual files download from `https://download.pangaea.de/dataset/973251/files/{year}.zip`
with no account (the bulk `allfiles.zip` needs a login; single files do not). Each zip
holds **one shapefile per Sentinel-2 acquisition date**, named
`yyyy-mm-dd_pred_vector.shp` — **437 dated scenes, 233,197 polygons total**. Geometries are
**EPSG:3413** (NSIDC Polar Stereographic North) `Polygon`s with attributes `raster_val`
(== 1.0 for every lake) and `id`. Only the label polygons are downloaded (~89 MB total);
pretraining supplies its own imagery.

## Class mapping

Single foreground class; non-lake is not a class (positive-only per spec §5):

| id  | name                | meaning                                        |
|-----|---------------------|------------------------------------------------|
| 0   | `supraglacial_lake` | a mapped surface meltwater lake outline        |
| 255 | nodata / ignore     | everything else (ice / rock / shadow)          |

No fabricated negatives — the assembly step supplies negatives from other datasets.

## Encoding (polygons → tiles, spec §4)

Each **selected** lake becomes ONE tile in the **local UTM zone** (from the lake's lon/lat,
10 m/pixel), centered on the lake's representative point, sized to the footprint + 8 px
margin and **capped at 64×64**. **All same-date polygons falling inside the tile** (not
just the selected one) are rasterized as class 0 with `all_touched=True` so small lakes
survive at 10 m; the rest is nodata (255). Reprojection EPSG:3413 → UTM is done in pixel
space via `rasterize.geom_to_pixels`. Median lake footprint ≈ 2975 m² (~30 px); largest
≈ 4.9 km² (fits within a 640 m tile). Verified tiles: single-band uint8, EPSG:326xx at
10 m, ≤64×64, values ∈ {0, 255}; lake-pixel fraction mean ≈ 0.18.

## Sampling

Single class → up to **1000 tiles** (spec §5). Selected by **round-robin across the 437
dated scenes** (a fresh random lake per scene each pass) so coverage spreads over space and
time instead of being dominated by 2019 (~27 % of all polygons). Only lakes ≥ 100 m²
(≥ 1 pixel) are eligible as tile centers; smaller slivers (often model artifacts) are still
drawn as class 0 when they fall inside a tile. Per-year tile counts:
2016:77, 2017:137, 2018:159, 2019:224, 2020:67, 2021:195, 2022:141.

## Time range & change handling

Every polygon is a **dated S2 acquisition** (2016–2022, all in the Sentinel era; none
filtered). Supraglacial lakes are seasonal/transient. Following the orchestrator directive
and spec §5's seasonal-label rule, `time_range` = the **calendar year of acquisition**
(which contains that year's April–September melt season); the exact acquisition date is
preserved in `source_id` (e.g. `2016/2016-07-11_pred_vector/803`). `change_time` is null
(this is a presence/state label, not a dated change event).

**Caveat:** a lake outline is only strictly valid around its acquisition date, since lakes
drain within weeks. Pretraining's ~360-day input window will include the melt season, but
imagery elsewhere in the window may not show the lake.

## Verification (spec §9)

- 1000 `.tif` + 1000 matching `.json`; all single-band uint8, UTM 10 m, ≤64×64,
  values ∈ {0, 255}, `time_range` ≤ 1 yr and ≥ 2016, `change_time` null.
- Tile centers all fall in the NE Greenland glacier region; lakes are coherent contiguous
  blobs (median 1 blob/tile, ~29.5 px mean).
- **Spatial/temporal S2 overlay** (sample 000002, date 2016-07-11): the S2 scene from that
  exact date was found on Planetary Computer; lake-labeled pixels have B08(NIR) mean **951
  vs 6380** for non-lake, and NDWI **0.82 vs 0.12** — labels sit squarely on dark, high-NDWI
  meltwater. Georeferencing and time assignment confirmed correct.
- Re-running is idempotent (skips existing `{id}.tif`).

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.supraglacial_lakes_northeast_greenland
```
