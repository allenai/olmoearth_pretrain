# GRID3 Settlement Extents

- **Slug**: `grid3_settlement_extents`
- **Status**: completed
- **Task type**: classification (3-class, positive-only foreground)
- **Label type**: polygons
- **Num samples**: 3,000 label tiles (1,000 per class)
- **Family / region**: population / Sub-Saharan Africa
- **License**: CC-BY-SA-4.0

## Source

GRID3 (Geo-Referenced Infrastructure and Demographic Data for Development) *Settlement
Extents* v3.0 / v3.1, produced by CIESIN at Columbia University with Novel-T, WorldPop
(Univ. of Southampton), UNFPA and Flowminder, funded by the Bill & Melinda Gates
Foundation. Dataset portal: https://grid3.org/ ; data on the Humanitarian Data Exchange:
https://data.humdata.org/organization/grid3

Each country's *settlement-extents* layer is a set of **settlement polygons** derived by
aggregating open building-footprint data (Google Open Buildings v3 (2023), Microsoft
(2014–2023), OSM) onto a 3-arc-second (~100 m) grid, filtering settled grid cells with an
XGBoost settlement-probability model (≥0.5), delineating settled contours, and classifying
each resulting polygon by building count / built-up area (codebook field `type`) into three
settlement types:

| `type` value | meaning |
|---|---|
| Built-up Area (BUA) | ≥ 40 ha (400,000 m²) with ≥ 13 buildings/ha — urban, visible street/block grid |
| Small Settlement Area (SSA) | ≥ 50 buildings, not a BUA — semi-urban / peri-urban |
| Hamlet | up to 49 buildings — rural, low-density, often isolated / hard-to-reach |

The product was derived in **2024**; settlement footprints reflect building imagery mostly
from ~2016–2023 and are a persistent land-use signal.

## Access method

Direct HTTPS download of the per-country settlement-extents GeoPackage zips from HDX — **no
account required** (CC-BY-SA-4.0). This is a large regional product (>15M settlements across
50 countries), so per **spec §5** we do **bounded sampling**: download a representative set
of **6 countries** spanning all major Sub-Saharan regions and draw a class-balanced sample.
We do **not** attempt continental coverage.

| ISO3 | Country | Region | Version | zip size |
|---|---|---|---|---|
| NGA | Nigeria | West Africa | v3.1 | 702 MB |
| SEN | Senegal | West Africa / Sahel | v3.0 | 55 MB |
| KEN | Kenya | East Africa | v3.0 | 341 MB |
| TZA | Tanzania | East Africa | v3.0 | 548 MB |
| COD | DR Congo | Central Africa | v3.1 | 472 MB |
| ZMB | Zambia | Southern Africa | v3.0 | 357 MB |

Saved/extracted under `raw/grid3_settlement_extents/`. Only the `*_settlement_extents_*`
GeoPackage (the polygon layer) is used; the companion `*_settlement_grid_*` (~100 m cell
centroids) is not downloaded. Polygon-layer fields: `country`, `iso3`, `building_count`,
`building_area`, `type`, `probability`, `date`, `source`, `mgrs_code`; geometry WGS84
(EPSG:4326). The exact HDX resource URLs are hardcoded in the script (from the HDX CKAN
`package_show` API).

## Class mapping

**Positive-only** (spec §5): settlement types are foreground land cover; non-settlement is
left as nodata/ignore (no synthetic background/negative class is fabricated — this matches
the manifest's 3-class scheme).

| id | name | source `type` |
|----|------|---------------|
| 0  | built-up area | `Built-up Area` |
| 1  | small settlement area | `Small Settlement Area` |
| 2  | hamlet | `Hamlet` |
| 255 | *(nodata)* | all pixels outside any settlement polygon — ignore; assembly adds negatives |

## Processing

- **Rasterization** (polygons): each selected polygon → one **64×64** UTM **10 m** tile
  (640 m) centered on the polygon (centroid, or an interior representative point when the
  centroid falls outside a concave shape). **Every** settlement polygon intersecting the
  tile bbox is burned in with its own `type` id (`all_touched=True` so tiny hamlets survive
  at 10 m); all other pixels are 255. Polygons are read on demand per tile via a pyogrio
  bbox filter (uses the GeoPackage R-tree index).
- **Sampling**: candidate placement points are pooled across the 6 countries (capped to
  3,000 per country/class to bound memory; BUAs kept in full — only ~3,524 across all 6),
  then selected **class-balanced at up to 1,000 tiles per class** (`balance_by_class`).
  BUAs are globally rare, so this lifts them to parity with the abundant SSAs/hamlets. Tiles
  are counted by the **centered** polygon's type.
- **Time range**: settlement extents are persistent land use. Each tile gets a **1-year
  static window on 2021** (`change_time=null`) — a representative year within the manifest's
  2016–2021 span; the product was derived in 2024 from ~2016–2023 building footprints, and
  settlements persist across the Sentinel era, so any Sentinel-era window shows them. This
  is presence/type classification, not a dated change event.

## Sample counts

- **Total**: 3,000 tiles — **class 0 (BUA) 1,000, class 1 (SSA) 1,000, class 2 (hamlet)
  1,000** (counted by centered polygon type).
- Value coverage across all tiles (tiles containing each value): BUA in 1,057, SSA in
  1,197, hamlet in 1,342, nodata(255) in 2,635 — many tiles contain neighboring settlements
  of other types, so a tile can carry multiple classes.
- Per-country tile counts: NGA 888, COD 567, KEN 429, SEN 391, TZA 387, ZMB 338.
- Candidate pool after per-country/class capping: BUA 3,524, SSA 18,000, hamlet 18,000.

## Verification

- All 3,000 tifs: single band, `uint8`, exactly 64×64, local UTM CRS at 10 m
  (`x_res=10, y_res=-10`); pixel values ∈ {0,1,2,255} only; metadata class ids {0,1,2}
  cover all non-nodata values. Every `.tif` has a matching `.json` with a ≤1-year
  `time_range` and no `change_time`.
- **Geographic/semantic sanity**: BUA-centered tiles are 86–99% built-up (large cities fill
  the 640 m window), SSA tiles 16–64%, hamlet tiles ~3% (tiny settlements in mostly-nodata
  patches) — matching the settlement-type definitions. Tile centers round-trip to plausible
  urban/rural locations in the expected countries and correct UTM zones (e.g. eastern DRC →
  EPSG:32635/32636/32735).
- A pixel-level Sentinel-2 image overlay was **not** rendered (would require configuring an
  imagery source); georeferencing is verified via rslearn's exact `GeotiffRasterFormat`
  encode and coordinate round-trips. Georeferencing is expected to be accurate to the source
  building-footprint aggregation (~100 m grid), so hamlet edges may be a pixel or two off at
  10 m — not material for the coarse settlement-type signal.

## Caveats

- **Bounded, not continental**: 6 of 50 available Sub-Saharan countries; regionally
  representative but not exhaustive. To expand, add more country slugs from
  `data.humdata.org/organization/grid3` (all share the same schema).
- **Model-derived labels**: settlement polygons are model/rule outputs from building
  footprints (XGBoost test accuracy ~0.86), not manual reference; boundaries and rare
  class assignments carry the product's error.
- **Positive-only**: non-settlement pixels are nodata (255), not a mapped background class;
  the pretraining assembly step supplies negatives from other datasets.
- **Version mix**: NGA/COD use v3.1, others v3.0 — the `type` field and codebook are
  consistent across both (v3.1 is a minor correction release). NGA v4.0 exists but is a
  3.4 GB single GeoPackage behind a redirect host, so v3.1 was used.
- **Time**: the manifest's `annotation_method` ("model-derived from Maxar imagery + rules")
  describes earlier GRID3 versions; v3.x is derived from Google/Microsoft/OSM building
  footprints. The chosen 2021 window is a persistence-based approximation, not a per-polygon
  acquisition date (the `date` field is uniformly the 2024 derivation year).

## Reproduce

```bash
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.grid3_settlement_extents
```

Idempotent: raw zips are downloaded+extracted once into `raw/grid3_settlement_extents/`;
existing `locations/{id}.tif` are skipped. Outputs to
`/weka/dfive-default/helios/dataset_creation/open_set_segmentation/datasets/grid3_settlement_extents/`
(`metadata.json`, `locations/{000000..002999}.tif` + `.json`).

*Note*: a handful of very large city (BUA) tiles in Nigeria/DRC are slow to rasterize
(thousands of neighboring polygons in the query bbox), so the final ~20 tiles can take
several minutes even with 64 workers; the run still completes.
