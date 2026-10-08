# SDPT v2 (Spatial Database of Planted Trees)

- **Slug**: `sdpt_v2_spatial_database_of_planted_trees`
- **Status**: completed
- **Task type**: classification (per-pixel, rasterized polygons)
- **Samples**: 16,884 single-band GeoTIFF label patches across 254 classes
- **Source**: WRI / Global Forest Watch — "Spatial Database of Planted Trees (SDPT
  Version 2.0)" (Richter, J., Goldman, E., Harris, N., Gibbs, D., Rose, M., Peyer, S.,
  Richardson, S., Velappan, H. 2024). License **CC-BY-4.0**.

## What the source is

SDPT v2 is a near-global compilation of planted-forest and agricultural tree-crop
**polygons** for 158 countries (~264 Mha planted forest + ~65 Mha tree crops, ~90% of
world planted-forest area in 2020). Most country maps come from supervised classification
or manual polygon delineation of Landsat / SPOT / RapidEye imagery. Plantations are
coherent land-cover stands clearly observable at 10 m.

## Access method

The GFW Data API `/query` endpoint requires an API key we do not have, but the full
product is published as an **unauthenticated public File Geodatabase** on the GFW S3
bucket:

```
https://gfw2-data.s3.amazonaws.com/plantations/sdpt/sdpt_v2_v11282023_public.gdb.zip
```

5.5 GB zipped, ~25 GB unzipped — within budget and well under the "impractical download"
threshold, so we pulled it once and sampled locally (no global coverage attempted). The
GDB has **one MultiPolygon layer per country** (`{iso3}_plant_v2`, 116 layers,
**26.6M polygons** total) sharing a harmonized attribute table (CRS EPSG:4326).
Raw archive + `SOURCE.txt` under
`raw/sdpt_v2_spatial_database_of_planted_trees/`.

## Class mapping

- **Class field = `sciName`** (the SDPT harmonized scientific taxon), the fine
  species/type scheme the task calls for. Globally there are **1,178 usable distinct
  values** (genus- or species-level, e.g. *Hevea brasiliensis* = rubber,
  *Elaeis guineensis* = oil palm, *Pinus sp.*, *Eucalyptus sp.*, *Prunus dulcis* = almond,
  *Cunninghamia sp.*).
- **254-class uint8 cap honored**: kept the **top 254 by global frequency** (ids 0..253 in
  descending frequency), **dropped 924** rarer taxa. Class id 0 = *Hevea brasiliensis*,
  1 = *Elaeis guineensis*, 2 = *Pinus sp.*, ...
- **Sentinel value `Unknown`** (species unidentified; ~65% of all polygons, ~16.7M) and
  `Unknown mix` / null were **excluded from the class set** — they are not a usable
  species/type class, so those polygons are simply never sampled (documented judgment
  call).
- Coarser fields recorded for reference but **not used** as the label: `simpleName`
  (13 coarse types: Oil palm / Rubber / Fruit / Wood fiber or timber / Other / ...) and
  `simpleType` (Planted forest / Tree crops).

## Rasterization

Each selected polygon is rasterized (`rasterize.py`, `all_touched=True`) into a **≤64×64
local-UTM 10 m** tile sized to the polygon footprint (centered on the polygon centroid,
capped at 64): the polygon's class id is burned **inside**, **255 (nodata/ignore)
outside**. SDPT only labels planted-tree polygons, so unlabeled land is *ignore*, not a
background class (spec §5: no fabricated negatives; the assembly step supplies negatives
from other datasets). 347 candidates produced an all-nodata patch (centroid fell in a
MultiPolygon hole) and were skipped.

## Sampling (bounded, spec §5)

Tiles-per-class balanced via `sampling.balance_by_class` with the 25k per-dataset cap.
With 254 classes the effective per-class limit is `25000 // 254 = 98`. Rare classes are
prioritized. As a large global product we did **not** attempt global coverage: an
attribute-only pass computed global `sciName` frequency, then only up to `CAND_CAP=400`
polygons per class per country layer were read as candidates (fair seeded random subset)
before balancing — enough to reach the target counts. Result: all 254 classes present,
**6–98 tiles per class** (median 91; most common taxa hit the 98 cap; sparse taxa kept
per §5 — downstream filtering removes too-small classes, we do not drop them here).

## Time range

SDPT plantations are persistent land cover → 1-year window per sample (spec §5
static-label rule), anchored on a representative Sentinel-era year parsed from the
polygon's `imageryYear` (year(s) of imagery used to delineate it), **clamped to the
manifest range [2016, 2020]**; unparseable → 2020. No change labels (`change_time=null`).

## Verification

- 16,884 `.tif` each with a matching `.json`; all single-band uint8, UTM at 10 m, ≤64×64,
  nodata 255, pixel values ∈ {class id 0-253, 255}. Scanned 1,535 tifs: no invalid values.
- All sample `time_range`s span exactly 1 year (≤366 days).
- `metadata.json` class ids (0-253) cover all values in the tifs.
- Geo sanity: sample tile centers reproject back into their source country (e.g.
  `arg_plant_v2` → -54.50, -26.22 in Misiones, Argentina; `kor_plant_v2` → 128.62, 36.86
  South Korea; `gtm_plant_v2` → -91.86, 14.95 Guatemala; `ind_plant_v2` → 80.03, 13.61
  Tamil Nadu, India). Georeferencing written exactly via rslearn `GeotiffRasterFormat`.
  (A full Sentinel-2 overlay eyeball was not run; georeferencing is exact and provenance
  verified.)

## Caveats

- Species-level distinctions (e.g. *Pinus taeda* vs *Pinus rigida*) are generally **not**
  separable from 10 m S2/S1 imagery; the fine `sciName` scheme is kept because the broader
  effort retains species/genus labels (pretraining learns from co-location) and the
  assembly/filtering steps handle over-fine or too-small classes. Coarser `simpleName`
  would be more robustly observable at 10 m but has only 13 classes.
- ~65% of the source is `Unknown` species and is excluded (see above).
- Definitional/temporal inconsistencies exist across contributing countries (per WRI
  cautions); `imageryYear` spans ~2008-2021, clamped into [2016, 2020] for pretraining.

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.sdpt_v2_spatial_database_of_planted_trees
```

Idempotent (skips already-written `{sample_id}.tif`). Downloads + unzips the GDB to
`raw/` on first run.
