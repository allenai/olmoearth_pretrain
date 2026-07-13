# USWTDB (US Wind Turbine Database)

- **Slug:** `uswtdb_us_wind_turbine_database`
- **Status:** completed
- **Task type:** classification (object detection encoded as per-pixel classes)
- **Samples:** 2000 (1000 turbine positive tiles + 1000 background negative tiles)
- **Source:** USGS / LBNL / AWEA — U.S. Wind Turbine Database, public domain (U.S. Government work)
- **URL:** https://energy.usgs.gov/uswtdb/

## What the source is

USWTDB is the authoritative national inventory of onshore and offshore wind turbines in the
United States and its territories. Every turbine is position-verified against high-resolution
aerial/satellite imagery and the database is updated quarterly. It is a **complete inventory**
(every known U.S. turbine), so within any tile a non-turbine pixel is a **true negative** — the
same property the USPVDB solar and Stanford well-pad datasets rely on.

## Access method (label-only, no imagery)

Downloaded the turbine **points** as one JSON array from the public USGS EERSC PostgREST API
(no credentials required; a browser User-Agent header is sent):

```
https://energy.usgs.gov/api/uswtdb/v1/turbines
```

75,727 turbine records, each with a unique `case_id`, WGS84 `xlong`/`ylat`, project online
year `p_year` (year-granular), and attributes `t_cap` (nameplate kW), `t_hh` (hub height m),
`t_rd` (rotor diameter m), `t_model`/`t_manu`, `t_offshore` (0/1, 79 offshore), and
location/attribute confidence `t_conf_loc`/`t_conf_atr`. A new shared helper
`download.download_postgrest_json()` pages the PostgREST endpoint into `raw/{slug}/`.
Imagery is **not** downloaded — pretraining supplies its own imagery.

## Label / class mapping

Single foreground class. `label_type = points` with presence-only semantics → the
**object-detection recipe** (spec §4), identical encoding to the local `olmoearth_wind_turbine`
detection dataset:

- **Class 0 = background**, **Class 1 = turbine**, **255 = nodata/ignore buffer**.
- Each selected turbine gets a 64×64 (640 m @ 10 m) context tile in its own local UTM
  projection, centered on the turbine pixel.
- The turbine is a **1×1 positive** (turbine tower/pad is ~1 px at 10 m but a strong,
  detectable signature — tower shadow, gravel pad, access roads), ringed by a **10 px nodata
  (255) buffer** because position-verified coordinates are still not pixel-exact at 10 m; all
  other pixels are background (0).
- Because the inventory is complete, **every other turbine falling inside a tile is also marked
  positive** (STRtree query over all turbines), so background pixels are true negatives. Dense
  wind farms yield several turbines per tile (405/1000 positive tiles have >1 turbine pixel;
  2619 turbine points total across the 1000 positive tiles).

## Time-range and change handling

`p_year` is **year-granular only**, so the installation event is **not** resolvable to ~1–2
months and cannot be used as a change label (spec §5 timing rule). Instead the **persistent
post-construction state** (a turbine stays visible for years) is used as presence
classification with `change_time = null`, and each tile's 1-year window is anchored **after**
commissioning in the Sentinel-2 era so the turbine is present:

```
window_year = clamp(p_year + 1, 2017, 2024)
```

This keeps pre-2016 turbines (still standing post-2016) while honoring the post-2016 rule.
Turbines with null `p_year` (1,125 in the source) and background negatives use a static 2022
window. All windows are ≤ 1 year.

## Negatives (spec §5 detection exception)

1000 background-only tiles sampled inside the U.S. by offsetting random turbines by 15–60 km
and rejecting any candidate within 3 km of a turbine (vectorized haversine). Verified: negative
tile centers are all ≥ 3.04 km from the nearest turbine (mean 18.6 km).

## Tile size / detection parameters

- `tile_size = 64` (640 m), `positive_size = 1`, `buffer_size = 10`.

## Sampling

Single foreground class → up to 1000 positive turbine tiles + 1000 background negative tiles
(well under the 25k cap), matching the turbine/well-pad/solar detection precedent. Onshore and
offshore turbines both eligible. Selection is seeded (`SEED = 42`) and reproducible.

## Verification (spec §9)

- All 2000 `.tif`s: single band, **uint8**, projected UTM CRS at **10 m**, size **64×64**,
  values ∈ {0, 1, 255}; each `.tif` has a matching `.json` with a ≤1-year `time_range` and
  `change_time = null`; `metadata.json` classes cover all values present.
- **Georeferencing check:** reprojecting class-1 pixel centers back to WGS84 lands within one
  10 m pixel of an actual USWTDB turbine for 100% of checks (max 6.8 m, mean 3.8 m), confirming
  labels sit exactly on real turbines. A full Sentinel-2 RGB overlay was not rendered, but the
  encoding/tile pipeline is the same, already-validated one used by `olmoearth_wind_turbine`.
- Re-running the script is idempotent (skips existing `{sample_id}.tif`).

## Caveats

- Turbines are ~1 px objects at 10 m; the label marks presence, not a resolvable footprint.
  The thick 10 px nodata buffer avoids penalizing near-misses.
- Coordinates are among the most accurate available (manual verification), but the buffer still
  accounts for residual sub-pixel offset.

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.uswtdb_us_wind_turbine_database
```
