# PEATMAP — global peatland extent

- **Slug**: `peatmap`
- **Task**: classification (binary peatland vs background segmentation)
- **Label type**: polygons → rasterized ≤64×64 UTM 10 m tiles
- **Samples**: 2000 (1000 peatland-positive tiles + 1000 background-only negatives)
- **Status**: completed

## Source & access

PEATMAP (Xu, Morris, Liu & Holden 2018, *Catena*, DOI 10.1016/j.catena.2017.09.010;
dataset DOI 10.5518/252), University of Leeds research-data archive record 251:
https://archive.researchdata.leeds.ac.uk/251/ — **CC-BY-4.0, open, no credential**.

PEATMAP is a meta-analysis that harmonizes the best available global / regional / national
peat maps into a single global set of **peatland extent polygons**. It is a
**derived product** (compiled cartography), not in-situ reference data. Delivered as
per-continent zipped ESRI shapefiles in **ESRI:54034** (World Cylindrical Equal Area,
metres). All continent archives were downloaded and unzipped to
`raw/peatmap/{Africa,Asia,Europe,North_America,Oceania,South_America}/`; a `SOURCE.txt`
records provenance. Every layer is used:

| Continent | layers | polygons |
|---|---|---|
| Africa | AF | 20,412 |
| Asia | EA, NEA, SEA, SIB, Histosols(Hokkaido/Mongolia/N.Korea) | 25,171 |
| Europe | British Isles, Finland, Norway, Sweden, Other European | 172,666 |
| North America | Canada, USA, Other | 46,527 |
| Oceania | Oceania (single multi-polygon) | 1 |
| South America | SA | 3,113 |

(Counts are after 2D/empty filtering; several layers are single large multi-polygons that
explode into many constituent polygons for sampling.)

## Class scheme (uint8)

| id | name | meaning |
|---|---|---|
| 0 | background | any 10 m pixel outside a PEATMAP polygon (other land / water) |
| 1 | peatland | inside a PEATMAP peatland polygon (bogs, fens, mires, tropical peat swamp forest) |
| 255 | nodata/ignore | declared for consistency; unused here |

Binary problem, so — following the `rubber` precedent for a global binary map — the target
is **1000 peatland-positive tiles + 1000 background-only negatives** (well under the 25k
cap). This is a **bounded, regionally-diverse** sample of a large global derived product,
not global coverage (§5).

## How labels are produced

- Each label is a **64×64 (640 m) tile at 10 m/px in the local UTM zone** (per-sample UTM
  chosen from the tile-center lon/lat).
- **Positive tiles**: interior points of peatland polygons, sampled **area-weighted**
  within each continent under an **even per-continent quota** (166–167 each → 1000) for
  regional diversity. Peat polygons intersecting the tile are reprojected 54034→UTM and
  rasterized (`all_touched=True`) as class 1; everything else is class 0.
- **Negatives**: for each of 1000 tiles, a random peat point is offset **30–120 km** and
  rejected unless the tile footprint is verified peat-free, then written as an all-zero
  (background) tile so the background class has genuine negatives.
- **Time range**: PEATMAP is a **static** baseline; each sample gets a 1-year window
  **2020-01-01 … 2021-01-01** (a representative Sentinel-era year). `change_time` is null.

## Sample counts

- peatland (class 1 present): **1000** tiles — Africa 166, Asia 167, Europe 167,
  North America 167, Oceania 167, South America 166.
- background-only negatives: **1000** tiles.
- Total: **2000**.

## Verification

- All 2000 `.tif`s: single-band **uint8**, **64×64**, local **UTM @ 10 m**, values ⊆ {0,1};
  every `.tif` has a matching `.json` with a ≤1-year time range.
- Peat-fraction separation is clean: positives mean 0.885 (min 0.116, **none empty**);
  negatives exactly 0.0 in all 1000.
- **Geolocation sanity**: positive tile centroids reprojected to WGS84 land squarely in
  well-known peatlands — Peruvian Amazon (Pastaza-Marañón), Congo Cuvette Centrale, Papua
  New Guinea peat swamps, Fennoscandian/Siberian mires, Canadian/Hudson-Bay peatlands.
  A pixel-level Sentinel-2 overlay was not run (peat is a soil/vegetation property, not
  directly photo-interpretable like open water, and PEATMAP is a compiled map product);
  correctness rests on exact georeferencing + the region-level geolocation check above.
- Re-running is **idempotent** (existing `{id}.tif` are skipped).

## Caveats

- PEATMAP is a **derived meta-analysis map**, not field reference; polygon boundaries carry
  the uncertainty of the underlying national/regional sources (varying vintages, mapping
  methods, minimum mapping units). Treat as approximate peatland extent.
- Source polygons are in an **equal-area** projection (ESRI:54034); tiles are re-cut to
  local UTM at 10 m. A few hundred source polygons are topologically invalid; validity is
  repaired lazily on the small clipped candidate geometry (full-polygon `make_valid` is
  pathologically slow and is deliberately avoided).
- Positive tiles centered inside large peat complexes are frequently ~100% peat (mean peat
  fraction 0.885); background context comes mainly from the negatives and from
  smaller-polygon positives.

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.peatmap
```

Raw source: `/weka/.../open_set_segmentation/raw/peatmap/` (continent shapefiles + SOURCE.txt).
Outputs: `/weka/.../open_set_segmentation/datasets/peatmap/{metadata.json, locations/*.tif+*.json}`.
