# OpenSentinelMap

- **Slug:** `opensentinelmap`
- **Manifest name:** OpenSentinelMap
- **Task type:** classification (dense_raster)
- **Status:** completed
- **Num samples:** 5650 label patches (64×64, 10 m, local UTM)
- **Source:** OpenSentinelMap — Johnson, Treible, Crispell, *OpenSentinelMap: A Large-Scale
  Land Use Dataset using OpenStreetMap and Sentinel-2 Imagery*, CVPRW (EarthVision) 2022.
  Vision Systems Inc. Site: https://visionsystemsinc.github.io/open-sentinel-map/
- **License:** open (dataset site states CC BY 4.0).

## What the source is

137,045 global spatial cells (~1.9 km ≈ 3.7 km² each), each with a per-pixel OSM-derived
land-use label plus multi-year (2017–2020) Sentinel-2 imagery. **We use only the labels**
— the imagery is ~445 GB and unnecessary here (OlmoEarth pretraining supplies its own S2).
Labels ship as a single 425 MB tarball of PNG masks; imagery (119 GB/year) was **not**
downloaded.

On-disk (after download + untar, all under `raw/opensentinelmap/`):
- `osm_categories.json` — label channels + class values + OSM tag definitions + precedence.
- `spatial_cell_info.csv` — per cell: `cell_id`, `MGRS_tile`, WGS84 `min/max lat/lon`, split.
- `osm_label_images_v10/{MGRS_TILE}/{cell_id}.png` — 192×192×3 uint8 label mask per cell.

Each PNG is 192 px × 10 m/px = 1920 m in the cell's MGRS UTM zone. Its three channels are
three OSM label "channels":
- **ch0 OSM_land_use** — 0 wooded, 1 agricultural, 2 residential, 3 industrial,
  4 commercial, 5 recreation, 6 airport, 7 quarry, 8 military, 9 desert_sand,
  10 mountain_rock, 11 other_natural
- **ch1 OSM_water_and_roads** — 12 water, 13 road
- **ch2 buildings** — 14 building

with 254 ("none", explicitly no label) and 255 ("unlabeled", outside OSM coverage) as
non-classes in every channel.

## Access method

Free Azure Blob (US-gov cloud), no credentials:
```
BASE=https://vsipublic.blob.core.usgovcloudapi.net/vsi-open-sentinel-map
curl -o osm_categories.json   $BASE/osm_categories.json
curl -o spatial_cell_info.csv $BASE/spatial_cell_info.csv
curl -o osm_label_images.tgz  $BASE/osm_label_images.tgz     # 425 MB
tar -xzf osm_label_images.tgz          # -> osm_label_images_v10/
```
(An AWS S3 mirror `s3://vsi-open-sentinel-map/` exists with `--request-payer`, ~$40 for the
full imagery+labels; not used.)

## Georeferencing (spec §8.2)

The label PNGs carry **no CRS**. They are recovered from `spatial_cell_info.csv`: each
cell's UTM zone is the MGRS tile's zone (e.g. `43QEU` → EPSG:32643), and the cell is an
axis-aligned **1920 m square** in that zone. Verified: at 66°N the cell's WGS84 bbox
envelope is 1983 m, exactly a 1920 m box rotated by the meridian-convergence angle
(≈1.93°) — confirming the cells are UTM-grid aligned, not lon/lat aligned. Each cell's UTM
box is reconstructed by transforming its WGS84 **center** to the MGRS UTM zone and laying a
192 px box (±960 m) around it, snapped to the nearest 10 m pixel. **No resampling** — the
label is already native UTM 10 m. Cells are placed at arbitrary UTM offsets (not on a shared
grid), so each is georeferenced independently; the ≤5 m snap error is negligible for 10 m
OSM labels.

**Spatial sanity check:** overlaid the `water` (12) label on 2019 Sentinel-2 NDWI (via
Planetary Computer) for 13 water-heavy tiles across 33–59°N. One near-all-water Seattle tile
gave 99.4% pixel agreement; median water IoU 0.68 with many tiles at 0.77–0.97. Orientation
(north-up, PNG row 0 = north) confirmed. Low-IoU cases are OSM label imperfection (seasonal /
turbid / vegetated water, coarse polygons), not misregistration — a systematic georef bug
would give near-zero IoU everywhere.

## Label / class mapping

The 3 channels are flattened to **one single-band 15-class uint8 map** by **OSM precedence
compositing**: at each pixel the label with the highest OSM `precedence` across the 3
channels wins; pixels with no class in any channel (all 254/255) → nodata 255. Precedence
order (high→low): building(100) > road(97) > water(96)/industrial(96) > commercial(95) >
recreation(99*) > airport(93) > quarry(92) > military(90) > residential(50) >
desert_sand(10) > agricultural(5) > other_natural(4) > wooded(2) > mountain_rock(1).
(*recreation prec 99 in source; building/road still dominate where they overlap it.)

Output class ids = the OSM channel pixel values (0–14). See `metadata.json` for full
id↔name↔description. nodata/ignore = 255.

## Sampling & tiling (spec §4–§5)

- 192 = 3×64, so each cell splits cleanly into **nine 64×64 patches** (no reprojection).
- A patch is kept only if **≥5%** of its pixels carry a class (OSM coverage is sparse;
  a higher threshold would discard thin road/building patches, which are legitimate but
  low-fraction). 611,363 candidate patches from 137,045 cells.
- **Tiles-per-class balanced** (rare classes first), ≤1000 tiles/class, 25k cap
  (`sampling.select_tiles_per_class`). All 15 classes have >1000 candidate tiles (rarest:
  military 5,958), so every class reaches its 1000 target. Selected **5650** patches
  (common classes appear in more tiles via co-occurrence).
- Candidates are sorted deterministically before the seeded selection → fully reproducible /
  idempotent (re-running skips existing `{id}.tif`).

Per-class tile counts (a tile counts toward every class it contains):

| id | class | tiles | id | class | tiles |
|----|-------|-------|----|-------|-------|
| 0 | wooded | 2319 | 8 | military | 1079 |
| 1 | agricultural | 1723 | 9 | desert_sand | 1022 |
| 2 | residential | 1703 | 10 | mountain_rock | 1057 |
| 3 | industrial | 1000 | 11 | other_natural | 1176 |
| 4 | commercial | 1100 | 12 | water | 2681 |
| 5 | recreation | 1077 | 13 | road | 4283 |
| 6 | airport | 1019 | 14 | building | 2582 |
| 7 | quarry | 1011 | | | |

## Time range

OSM land-use is ~static and the label is a single OSM snapshot (categories `version: 10`),
so all samples use a **1-year window on 2019** (`2019-01-01…2020-01-01`), the midpoint of the
2017–2020 imagery span. No change labels.

## Caveats

- **OSM label quality:** OSM land-use polygons are incomplete and vary in accuracy by region;
  many cell pixels are unlabeled (255). Thin classes (road, building) are dilated in the
  source to be visible at 10 m. Downstream assembly filters too-small classes.
- Labels-only: Sentinel-2 imagery from the source was not used.
- `spatial_cell_info.csv` lists 152,632 cells but only 137,045 have label PNGs; cells
  without a PNG are skipped.

## Reproduce

```
# 1. stage raw (see Access method) into
#    /weka/.../open_set_segmentation/raw/opensentinelmap/
#    {osm_categories.json, spatial_cell_info.csv, osm_label_images_v10/}
# 2. run:
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.opensentinelmap --workers 64
```
Outputs: `datasets/opensentinelmap/{metadata.json, locations/{id}.tif+.json,
registry_entry.json}` on weka.
