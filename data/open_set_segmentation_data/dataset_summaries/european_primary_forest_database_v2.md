# European Primary Forest Database v2

- **Slug**: `european_primary_forest_database_v2`
- **Status**: completed
- **Task type**: classification (positive-only; single foreground family = primary/old-growth forest, sub-classed by EEA forest type)
- **Label type**: points/polygons
- **Num samples**: 5,762 label tiles
- **Family / region**: forest / Europe (~35 countries)
- **License**: CC-BY-4.0

## Source

Sabatini, F.M., Bluhm, H., Kun, Z. *et al.* **European primary forest database v2.0.**
*Scientific Data* 8, 220 (2021). https://doi.org/10.1038/s41597-021-00988-7

The EPFD v2.0 harmonizes **48 regional-to-continental datasets** of primary / old-growth /
virgin forest into one geodatabase: **18,411 polygon patches** (digitized boundaries) plus
**299 point locations** (patches for which only an approximate centre was known), spread
across ~35 European countries (41.2 Mha). Per patch, the DB records — when available —
forest name, location, naturalness level, extent, dominant tree species, disturbance
history, protection status, biogeographic region, and **forest type** (EEA European Forest
Type). Primary-forest status was verified with a Landsat (1985–2018) LandTrendr disturbance
check; ~94% of patches showed no status-altering disturbance in the prior 30 years.

## Access method

Open-access on Figshare (CC-BY-4.0), a single 112 MB zip — **no account required**:

```
Figshare record 13194095  ->  EPFDv2.0_DatabaseOA.zip
https://ndownloader.figshare.com/files/29091789
```

Saved + extracted under `raw/european_primary_forest_database_v2/`. The zip contains an
**ESRI *personal* geodatabase** `EPFD_v2.0.mdb` (214 MB, JET4/Access), the metadata docx,
and the authors' ArcGIS/R/GEE build scripts. Three source datasets (IDs 17 Hungary, 34
UNESCO Beech WHS, 48 Austria) are **not** in the open-access release; they are excluded
upstream and not needed here.

**Reading the `.mdb`:** the GDAL build available lacks the ESRI `PGeo` driver, so the
tables were read with the `mdbtools` `mdb-json` CLI (`sudo apt-get install -y mdbtools`;
it base64-encodes the binary `SHAPE` column) and the **ESRI shape-binary geometry decoded
directly** (Point = type 1, Polygon = type 5; coordinates are WGS84 **EPSG:4326**). The two
harmonized open-access feature classes used are:

- `EU_PrimaryForests_Polygons_OA_v20` — 18,411 polygons
- `EU_PrimaryForests_Points_OA_v20` — 299 points

The parse is materialized **once** into a reproducible GeoPackage
`raw/european_primary_forest_database_v2/parsed/epfd_oa.gpkg` (layers `polygons`,
`points`); subsequent runs read that GPKG and no longer need mdbtools.

## Class mapping

Single foreground family (all patches are primary/old-growth forest), **sub-classed by the
DB's `FOREST_TYPE1` = EEA European Forest Type** (EEA Technical Report 9/2006, derived by
the authors from the map of Potential Vegetation types for Europe). The FOREST_TYPE1 integer
code is used directly as the class id (contiguous 0–13, 255 = nodata):

| id | class | id | class |
|----|-------|----|-------|
| 0 | Unclassified forest type | 7 | Mountainous beech |
| 1 | Boreal | 8 | Thermophilous deciduous |
| 2 | Hemiboreal & nemoral coniferous/mixed | 9 | Broadleaved evergreen |
| 3 | Alpine coniferous | 10 | Mediterranean/Anatolian/Macaronesian coniferous |
| 4 | Acidophilous oak & oak-birch | 11 | Mire & swamp |
| 5 | Mesophytic deciduous | 12 | Floodplain |
| 6 | Lowland-submontane beech | 13 | Non-riverine alder/birch/aspen |

The scheme was validated by cross-tabulating code vs. `DOMINANT_TREE_SPECIES1` (e.g. codes
6/7 → *Fagus sylvatica*; 1/3 → *Picea abies*/*Pinus sylvestris*; 9 → *Quercus suber/ilex*),
confirming code == the paper's 1–13 category numbering with code 0 = no EEA type assigned.
The 14 classes span the broadleaf (4,5,6,7,8,9,12), coniferous (1,3,10) and azonal/mixed
(2,11,13) groups; the broadleaf↔coniferous distinction is observable from S2/S1/Landsat,
finer biogeographic types less so (see caveats).

### Selected-sample counts per class

```
0 Unclassified ...........  610      7 Mountainous beech ......... 1000
1 Boreal ................. 1000      8 Thermophilous deciduous ...  139
2 Hemiboreal/nemoral .....  115      9 Broadleaved evergreen .....   81
3 Alpine coniferous ...... 1000     10 Medit/Anatol/Macaron con. .   39
4 Acidophilous oak .......  180     11 Mire & swamp .............   158
5 Mesophytic deciduous ...  148     12 Floodplain ...............   113
6 Lowland-submont beech .. 1000     13 Non-riverine alder/birch .   179
                                    ------------------------------------
                                    total ....................... 5,762
```
Candidate pool per class (before the 1000-cap): {0:610, 1:3621, 2:115, 3:6445, 4:180,
5:148, 6:2429, 7:4453, 8:139, 9:81, 10:39, 11:158, 12:113, 13:179}. Only the four common
types (1,3,6,7) hit the 1000/class cap; all rarer types are kept in full (rare classes are
retained per spec §5 — downstream assembly filters too-small ones).

## Representation & tiling

Everything is written as single-band **uint8, 10 m/pixel, local-UTM, ≤64×64** GeoTIFF
label patches in `locations/` (one unified class scheme; positive-only ⇒ every non-patch
pixel = **255 nodata**, no synthetic negatives per spec §5):

- **Polygons (dominant, 5,555 of the tiles are 64×64):** one tile per polygon, centered on
  the polygon's interior representative point; the polygon (holes honored, `all_touched=True`
  so tiny patches survive) is rasterized to its FOREST_TYPE1 class id, rest = 255. Patches
  larger than a 640 m tile are captured as a **central all-forest window**.
- **Points (299 approximate patch centres, no footprint):** a small **uniform-class** tile
  sized from `FOREST_EXTENT_MEASURED` (ha → side px = √area/10 m, clamped **[3, 32] px**;
  default **8 px** when unmeasured). Points without a FOREST_TYPE1 → class 0. (Point tiles
  are the 3–32 px patches in the size histogram.)

Tiles-per-class balanced, **≤1000 tiles/class**, 25k hard cap
(`sampling.balance_by_class`, seed 42 ⇒ reproducible selection).

## Time range & change handling

Primary/old-growth forest is a **persistent, static** land cover. Each sample is assigned
the static **1-year window 2020** (`change_time = null`) — the year v2.0 was compiled, well
inside the manifest's 2016–2021 range and the Sentinel era, and after the Landsat
disturbance verification (through 2018). Not a change dataset.

## Verification

- 5,762 `.tif` ↔ 5,762 `.json`; **0** bad tiles (all single-band uint8, EPSG:32xxx UTM,
  10 m res, ≤64×64). All raster values ∈ {0…13} ∪ {255}; every value is a declared class;
  no `time_range` exceeds 1 year.
- **Georeferencing/scale** validated by comparing rasterized forest area to the DB
  `Area_ha` for polygons that fit inside a tile: ratios ~1.1–1.2× (expected `all_touched`
  1-px border inflation on small patches), <1 where the patch exceeds the 640 m tile
  (clipped) — confirming correct WGS84→UTM projection and 10 m pixels.
- **Geometry decode** independently validated: point OBJECTID 1 decodes to (20.6°E, 63.7°N)
  = "Island Bjuren", Sweden (boreal) as recorded; the code↔dominant-species cross-tab is
  internally consistent. A live Sentinel-2 overlay was **not** fetched; correctness rests on
  the above coordinate/area/attribute cross-checks.
- Re-running the script is **idempotent** (second run: 5,762/5,762 skipped).

## Caveats

- **Forest type is coarse / modeled.** `FOREST_TYPE1` was derived from a *potential natural
  vegetation* map, not observed per-patch, so the finer biogeographic sub-classes carry
  label noise relative to what the sensors see; the broadleaf↔coniferous split is the most
  reliable signal. Downstream can coarsen the 14 classes if desired.
- **Point patches are approximations** — the 299 points are "approximate centres" with no
  digitized boundary, so their small uniform tiles assert forest over a modeled footprint
  centered on an imprecise location. They are 299/5,762 (~5%) of samples.
- **Disturbed patches retained.** ~27% of patches had *some* Landsat-detected disturbance
  1985–2018 (statistically ~6% anthropogenic); the DB still lists them as primary-forest
  patches and per-patch anthropogenic flags are not provided, so none were filtered.
- **Large patches are windowed**, not fully tiled — one central all-forest window per patch
  (not multiple sub-windows), consistent with the ≤64×64 cap.

## Reproduce

```bash
# (one-time, to build the parsed GPKG from the .mdb)
sudo apt-get install -y mdbtools
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.european_primary_forest_database_v2 --workers 64
```

Outputs:
`/weka/dfive-default/helios/dataset_creation/open_set_segmentation/datasets/european_primary_forest_database_v2/`
(`metadata.json`, `registry_entry.json`, `locations/{000000…}.tif`+`.json`). Raw source +
parsed GPKG under the sibling `raw/…` path.
