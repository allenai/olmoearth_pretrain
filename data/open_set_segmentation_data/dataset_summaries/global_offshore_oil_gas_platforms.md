# Global Offshore Oil & Gas Platforms (OOGPs)

- **Slug:** `global_offshore_oil_gas_platforms`
- **Status:** completed
- **Task type:** classification (positive-only object **detection**, encoded as per-pixel classes)
- **Num samples:** 2,000 (1,000 platform-positive tiles + 1,000 open-water negative tiles)

## Source

"The Offshore Oil and Gas Platforms (OOGPs) dataset based on satellite data spanning 2017
to 2023", Zenodo record 18350974 (<https://doi.org/10.5281/zenodo.18350974>),
license **CC-BY-4.0**. A vector inventory of offshore oil/gas platforms across six major
offshore basins — **Gulf of Mexico, Persian Gulf, North Sea, Caspian Sea, Gulf of Guinea,
Gulf of Thailand** — derived from satellite (Sentinel-1 SAR) observations plus validation.

Only the 977 KB label archive `OOGPs_v1.0.0.zip` was downloaded (**no imagery** — the
pretraining pipeline supplies its own S1/S2/Landsat). Of the two layers in the archive we
use `OOGPs_all_v1.0.0.gpkg` (layer `platforms`, **9,334** Point features, EPSG:4326), which
carries the per-year presence field `Year_label`; the alternate `OOGPs_2023_v1.0.0.gpkg`
(5,358 features, 2023 snapshot only) was not used.

Fields: `Latitude`, `Longitude`, `Area`, `Country`, `EEZ`, `Installation_date` (YYYYMM,
present for only ~407 records), `Removal_date` (YYYYMM, ~969 records), `Flaring_status`
(0/1), `Year_label` (comma-separated calendar years 2017–2023 in which the platform was
detected/present). All 9,334 platforms had ≥1 in-range year.

## Class / label mapping

Single positive class; detection encoding adds background + nodata:

| id | name | meaning |
|----|------|---------|
| 0 | background | open water / ocean surface, no platform |
| 1 | offshore_oil_gas_platform | fixed offshore oil/gas platform (production/drilling, wellheads, related fixed structures) |
| 255 | nodata / ignore | detection buffer ring around each positive |

## Detection encoding (spec §4)

- One **32×32** UTM 10 m context tile per selected platform, centered on its point.
- **1 px positive** (class 1) ringed by a **10 px nodata (255) buffer** → 21×21 ignore
  region, the rest of the tile is background (0). Parameters: `tile_size=32`,
  `positive_size=1`, `buffer_size=10`.
- Any **other platform present the same year** falling inside a tile is also encoded as a
  positive (platforms cluster in fields), so in-field neighbors are labeled consistently.
- **Negatives:** 1,000 background-only open-water tiles obtained by offsetting a random
  real platform 3–8 km in a random bearing and confirming (KD-tree) no platform lies within
  ~1.1 km — real offshore open water in the same basins as the positives.

## Time-range and change handling (spec §5)

Platforms are **persistent structures**, not change events. `Year_label` resolves presence
only to a **calendar year**, and month-precision install dates exist for only ~4% of
platforms — both coarser/sparser than the ~1–2 month change-timing bar — so we do **not**
emit dated change labels (`change_time=null`). Each positive uses a **persistent-structure
1-year window**: a tile is emitted only for a calendar year listed in the platform's
`Year_label`, guaranteeing the structure is present across the whole window. This mirrors
the GFW SAR fixed-infrastructure and DeepOWT persistent-structure precedents. All labels
are post-2016 (2017–2023).

## Sampling

To avoid over-representing long-lived platforms (3,135 are present in all 7 years), **each
physical platform contributes at most one positive tile**, at a randomly chosen year from
its `Year_label`. Positives are then **year-stratified** (balanced across 2017–2023) and
capped at 1,000; negatives capped at 1,000. Total 2,000 — well under the 25k cap. Seed 42.

## Overlap note

This source **partially overlaps the GFW SAR fixed-infrastructure dataset**
(`global_fishing_watch_sar_fixed_infrastructure`) — both are Sentinel-1-derived offshore
oil/gas detections. This is acceptable: downstream assembly handles dedup. The two are
processed independently; OOGPs is a distinct release scoped to six named basins with
per-year presence labels spanning 2017–2023.

## Verification (spec §9)

- 2,000 `.tif` + 2,000 matching `.json`. All tiles single-band **uint8**, local **UTM**
  CRS at **10 m**, **32×32** (≤64). Unique values across the whole dataset: **{0, 1, 255}**
  — matches the class map + nodata 255.
- Positives: 1 px class 1, 440 px nodata ring, rest background; negatives: all background.
- Every `.json` has a **1-year** `time_range` and `change_time=null`.
- Georeferencing round-trip: tile-center lon/lat vs. source platform coordinate agree to
  **~4–8 m** (sub-pixel) for sampled positives; locations fall in the expected offshore
  basins (Gulf of Mexico, Persian Gulf, etc.).

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.global_offshore_oil_gas_platforms
```

Idempotent (skips already-written tiles). Downloads the Zenodo archive to
`raw/global_offshore_oil_gas_platforms/`, extracts the gpkg, and writes outputs to
`datasets/global_offshore_oil_gas_platforms/`.
