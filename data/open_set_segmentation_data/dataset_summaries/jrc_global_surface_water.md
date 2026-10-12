# JRC Global Surface Water

- **Slug:** `jrc_global_surface_water`
- **Manifest name:** JRC Global Surface Water
- **Task type:** classification (per-pixel, 3 classes)
- **Status:** completed — **1504** label patches (64×64 GeoTIFFs)
- **Family / region:** water / Global (bounded-tile sample)

## Source

EC JRC **Global Surface Water Explorer** (Pekel et al. 2016, *Nature*,
doi:10.1038/nature20584; https://global-surface-water.appspot.com/). A 30 m
derived-product mapping of global surface-water dynamics from the Landsat archive
(1984–2021, v1.4). Distributed as 10×10-degree GeoTIFF tiles in EPSG:4326 on the public
Google Cloud Storage bucket `global-surface-water` (no credentials required).

We use the **Seasonality** product. Each pixel value is the number of months surface
water was present in the reference year:

| Seasonality value | mapped class id | class name |
|---|---|---|
| 0 | 0 | no water |
| 1–11 | 1 | seasonal water |
| 12 | 2 | permanent water |

This maps exactly to the manifest's three classes (permanent water / seasonal water /
no water).

Tile URL pattern (verified working, HTTP 200):
```
https://storage.googleapis.com/global-surface-water/downloads2021/seasonality/seasonality_<lonLabel>_<latLabel>v1_4_2021.tif
```
where `<lonLabel>_<latLabel>` is the tile's NW-corner label (e.g. `20E_0N`, `70W_0N`,
`130E_20S`). Each tile is 40000×40000 px (~15 MB).

## Access method

Public GCS HTTP download via the shared `download.download_http`. 8 tiles, ~193 MB total,
written to `raw/jrc_global_surface_water/`. No account, license portal, or auth needed
(open Copernicus/JRC data, free with attribution).

## Sampling (bounded, per spec §5 "large global derived-product raster")

This is a global product, so we do **bounded-tile** sampling from representative
**interior continental** tiles across diverse biomes and both hemispheres:

| Tile (NW corner) | Region |
|---|---|
| `20E_0N` | Congo Basin interior — rivers, wetlands |
| `70W_0N` | Central Amazon — Solimões floodplain, rivers |
| `60E_70N` | W Siberia — Ob wetlands, thermokarst lakes |
| `110W_60N` | Canadian prairies/shield — lakes |
| `80E_30N` | Ganges/Himalaya foreland — seasonal floodplain |
| `130E_20S` | Australia interior — Lake Eyre basin, ephemeral lakes |
| `10W_20N` | Niger inland delta / Sahel — seasonal water |
| `20E_50N` | E Europe — lakes, rivers, reservoirs |

**Interior tiles are chosen deliberately:** JRC GSW masks the ocean to value 0
(== "no water"), so coastal tiles would mislabel open ocean as dry land. Restricting to
interior tiles makes value 0 correspond to genuine terrestrial dry land.

Within each tile we scan **non-overlapping** ~64px-footprint blocks (`BLOCK=22` native
30 m px ≈ 610 m). A block is a candidate if it is either pure land (0 % water) or has a
strong water signal (**≥ 10 % water pixels**, high-confidence); blocks with weak/ambiguous
water (0 < frac_water < 10 %) are skipped. Each candidate records the classes present
(≥ 5 % of the block). Selection uses `sampling.select_tiles_per_class`
(**tiles-per-class balanced, rarest class first**, ≤ 1000 tiles/class, 25k cap): seasonal
water (rarest) is filled first, then permanent, then no-water.

Candidate pool: 24.2 M blocks (per-class candidates: no-water 23.4 M, permanent 1.7 M,
seasonal 0.88 M).

**Selected: 1504 windows.** Tiles-per-class counts (a tile counts toward every class it
contains):

| class | tiles containing it |
|---|---|
| no water (0) | 1210 |
| seasonal water (1) | 1114 |
| permanent water (2) | 1000 |

### Judgment call — no pure-land-only tiles

Because tiles-per-class fills rare classes first and *every* water-containing window also
contains surrounding land, the abundant "no water" class (0) is fully satisfied by the
land present inside water windows before any pure-land window is reached. Consequently
**all 1504 selected tiles contain water**, and "no water" appears as the land/background
within water scenes rather than as standalone dry-land tiles. This is intentional and
appropriate: (a) mixed water/land tiles are the most informative segmentation labels
(they carry the water boundary), and (b) per spec §5 the pretraining-assembly step
supplies additional negatives by sampling other datasets, so dedicating samples to
low-information empty-land tiles is unnecessary.

## Label patches

- Single-band **uint8**, local UTM, **10 m/px**, **64×64**, north-up. `nodata = 255`.
- Values present across all tiles: `{0, 1, 2, 255}` (255 only from out-of-source fill).
- Native 30 m EPSG:4326 windows reprojected to local UTM at 10 m with **nearest**
  resampling (categorical labels).
- **Time range:** 1-year window anchored on the Seasonality reference year **2021**
  (`[2021-01-01, 2022-01-01)`), within the manifest range 2016–2021. Permanent water is
  temporally stable, so the exact anchor year is not critical. `change_time = null` (state
  map, not an event).

## Verification (spec §9)

- 1504 `.tif` + 1504 matching `.json`; all single-band uint8, UTM CRS, 10 m, 64×64,
  nodata 255; only valid class ids {0,1,2} + nodata.
- All sample JSONs carry a 1-year `time_range`; `metadata.json` class ids cover all values
  in the tifs.
- **Spatial/temporal sanity (Sentinel-2 NDWI overlay via Planetary Computer, 2021):**
  reprojecting cloud-free S2 (B03/B08) onto the label grid,
  - a pure permanent-water tile read 99 % NDWI>0 (water);
  - 3 of 4 mixed tiles showed crisp separation — labeled water pixels NDWI>0 fraction
    0.83–1.00 vs labeled land 0.01–0.04.
  - **Caveat noted:** one W-Siberia thermokarst tile had labeled permanent-water pixels
    reading land-like in a single Aug-2021 scene. This reflects GSW's multi-decade /
    all-12-months permanent-water definition vs a single optical date, plus low NDWI over
    shallow/vegetated thermokarst water — an individual-location discrepancy, not a
    georeferencing bug (the pipeline aligns crisply elsewhere). A minority of small,
    turbid, or vegetated water bodies may show this.

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.jrc_global_surface_water
```
Idempotent: tiles are skip-if-present, and each `locations/{id}.tif` is skipped if it
already exists. Script:
`olmoearth_pretrain/open_set_segmentation_data/datasets/jrc_global_surface_water.py`.

## Notes / caveats

- The manifest references a "validated reference point set". Those validation/reference
  points are **not published as a downloadable tile/table on the GSW portal** (the portal
  offers Occurrence, Change, Seasonality, Recurrence, Transitions, Max-Extent rasters; the
  yearly/monthly histories and validation points live in Google Earth Engine only). We
  therefore used the expert-validated Seasonality **raster** directly, which the manifest
  lists as the primary `dense_raster` label type. Seeding from the validation points would
  be a possible future enhancement if a downloadable copy becomes available.
- License: open (free with attribution; Copernicus / JRC open data).
