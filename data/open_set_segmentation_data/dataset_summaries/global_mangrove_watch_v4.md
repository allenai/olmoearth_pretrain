# Global Mangrove Watch v4

- **Slug:** `global_mangrove_watch_v4`
- **Status:** completed
- **Task type:** classification (dense_raster → 2-class per-pixel segmentation)
- **Samples:** 2000 (1000 mangrove windows + 1000 non-mangrove windows), from 1122 distinct
  one-degree source tiles spread across the global mangrove belt.

## Source

Global Mangrove Watch (GMW), UNEP-WCMC / JAXA / Aberystwyth University / soloEO.
Product: **"Global Mangrove Watch: Annual Mangrove Extent" v4.0.19**, the **2020 10 m
Sentinel-2 baseline** — Zenodo record `12756047` (DOI 10.5281/zenodo.12756047), CC-BY-4.0.
Over 30,000 machine-learning models trained on 5M+ photointerpreted reference points
classify mangrove vs non-mangrove from Copernicus Sentinel-2 imagery at 10 m (an upgrade
from the 25 m ALOS/Landsat baseline used in v3).

Distributed as **1647 one-degree GeoTIFF tiles** inside `gmw_mng_2020_v4019_gtiff.zip`
(~180 MB). Each tile is 10000×10000 uint8, EPSG:4326 at 0.0001° (~10 m), value **1 =
mangrove**, **0 = non-mangrove** (file nodata is set to 0). Tiles exist only where
mangroves occur (coastal tropics/subtropics), so the tile set itself delimits
representative mangrove regions.

## Access / download

Bounded download of a global derived-product (spec §5): only the single **2020 baseline**
GeoTIFF zip was pulled (`download.download_zenodo("12756047", ..., filenames=[
"gmw_mng_2020_v4019_gtiff.zip"])`), **not** the full 1990–2024 annual series. Raw file kept
at `raw/global_mangrove_watch_v4/`. No credentials required (public Zenodo, CC-BY-4.0).

## Class mapping

The distributed extent raster is binary, mapped to a 2-class scheme:

| id | name | source value | description |
|----|------|--------------|-------------|
| 0 | mangrove | 1 | Mangrove forest present in 2020 (GMW ML classification) |
| 1 | non-mangrove | 0 | All other cover in the analysis area (water, tidal flat, land, built-up, bare) |

`nodata = 255` (unobserved). uint8.

**Manifest gain/loss NOT encoded.** The manifest lists classes `[mangrove, non-mangrove,
gain, loss]`. gain/loss are the GMW **change** products (baseline-to-year comparisons
resolved only to annual/multi-year epochs). Per the change-timing rule (spec §5: a change
event must be datable to ~1–2 months to be usable), they cannot be placed confidently in a
pairing window and are intentionally omitted. We keep only the near-static **extent**
product, matching the task instruction.

## Sampling / method

- Native 0.0001° EPSG:4326 windows (BLOCK = 64 native px ≈ 640 m) reprojected to a local
  UTM projection at **10 m** with **nearest** resampling (categorical). Output tiles are
  **64×64** single-band uint8.
- **Tiles-per-class balanced**, spread across the global tile set with a **per-tile cap of
  10 per class** for geographic diversity, then `balance_by_class` to **≤1000 windows/class**:
  - **mangrove windows** — native block with mangrove fraction ≥ 10% (prefer
    homogeneous/high-confidence windows per spec §4); these carry **both** classes
    (mangrove core + surrounding non-mangrove boundary).
  - **non-mangrove windows** — no mangrove but within 3 blocks of a mangrove block (genuine
    coastal context, not open ocean/inland); carry only class 1.
- 28,736 candidate windows found across 1510 tiles (13,636 mangrove / 15,100 non-mangrove
  candidates) → 1000 + 1000 selected.
- Class occurrence across the 2000 output tiles (via `classes_present`): **mangrove in 1000
  tiles, non-mangrove in 1988 tiles** (12 mangrove windows are fully mangrove).

## Time range

Static 1-year window **2020-01-01 → 2021-01-01**, anchored on the mapped baseline year
(the 10 m Sentinel-2 layer). `change_time = null`.

## Verification

- 2000 `.tif` + 2000 matching `.json`. Sampled tiles: single band, uint8, 64×64, local UTM
  (e.g. 32647/32738/32755), 10 m resolution, nodata 255; pixel values ∈ {0, 1, 255}.
- All time ranges = 2020 (≤1 yr). `metadata.json` classes cover all observed values.
- Geographic sanity: all 1000 mangrove-window centers fall within lat [-38.6, 29.2]
  (entirely inside the global mangrove belt ≈ 40°S–32°N).

## Caveats

- Binary extent only; gain/loss change layers dropped (change-timing rule).
- Windows are drawn on a native 0.0001° grid then reprojected — a ~11 m→10 m nearest
  resampling introduces sub-pixel category shifts at boundaries (expected, categorical).
- Very sparse tiles (e.g. a tile with <2000 mangrove pixels) contribute no mangrove windows
  since no block reaches the 10% fraction; this is intentional (favors coherent patches).

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.global_mangrove_watch_v4
```
Idempotent (skips existing `locations/{id}.tif`). Re-downloads the zip only if absent.
