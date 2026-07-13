# USGS USMIN Mine Features

- **Slug:** `usgs_usmin_mine_features`
- **Task type:** classification (unified point-detection + polygon-segmentation)
- **Status:** completed — 10,500 samples
- **Source:** USGS Mineral Resources "Prospect- and Mine-Related Features from U.S.
  Geological Survey 7.5- and 15-Minute Topographic Quadrangle Maps" (USMIN), version 10.0
  (May 2023). Public domain.
- **Access:** national File Geodatabase downloaded from ScienceBase item
  `5a1492c3e4b09fc93dcfd574`:
  `https://www.sciencebase.gov/catalog/file/get/5a1492c3e4b09fc93dcfd574?name=USGS_TopoMineSymbols_ver10_Geodatabase.zip`
  (project page https://mrdata.usgs.gov/usmin/). No credentials required.

## What the source is

Mine-related symbols manually digitized from historical USGS topographic quadrangle maps.
Each feature has a `Ftr_Type` (feature-type symbol), state/county, source topo name/date,
and geometry (point or polygon) in EPSG:4326. The GDB ships six layers, at three source
map scales:

| scale | points | polygons |
|-------|--------|----------|
| 24k (1:24,000)   | 466,747 | 142,658 |
| 48k (1:48,000)   | 3,165   | 212     |
| 625k (1:625,000) | 101,205 | 11,703  |

## Decisions

- **Layers used: 24k + 48k only.** The **625k layers were dropped** — 1:625,000 source maps
  have positional error on the order of hundreds of metres, far too coarse to place a label
  on a 10 m grid. (Total mapped-and-used after dropping 625k: 459,933 points, 139,549
  polygons.)
- **Unified point + polygon dataset** (spec §5 multi-modality rule), one class scheme:
  - **Polygon features → rasterized** into a ≤64×64 UTM 10 m tile (footprint centered on
    the polygon; footprints >640 m keep their central 64×64). Polygons are large (median
    max-extent ≈289 m; 87% >100 m), so they are genuine 10 m segmentation targets.
  - **Point features → detection encoding** (`encode_detection_tile`): 1 px positive + 10 px
    nodata buffer ring in a 32×32 background context tile. All nearby point features (any
    class, found via a global EPSG:3857 KDTree prefilter + exact in-tile pixel check) are
    also marked positive so multiple co-located features aren't mislabeled background.
  - **"Prefer polygons where available":** within each class, polygon records are selected
    before point records.
- **Class scheme** (uint8; id 0 = background, 255 = nodata/ignore):

  | id | name | mapped `Ftr_Type` values |
  |----|------|--------------------------|
  | 0 | background | (negatives / outside polygons) |
  | 1 | prospect_pit | Prospect Pit, Diggings, Glory Hole |
  | 2 | mine_shaft | Mine Shaft, Air Shaft |
  | 3 | adit | Adit |
  | 4 | quarry_open_pit | Quarry(+ Rock/Limestone/Gypsum/Pumice), Open Pit Mine, Open Pit Mine or Quarry |
  | 5 | gravel_borrow_pit | Gravel Pit, Borrow Pit, Sand Pit, Sand and Gravel Pit, Gravel/Borrow Pit - Undifferentiated |
  | 6 | strip_mine | Strip Mine |
  | 7 | tailings_pile | Tailings - Undifferentiated/Placer/Dredge/Mill, Slag Pile |
  | 8 | tailings_pond | Tailings - Pond, Settling Pond, Leach Pond, Evaporation Pond, Salt Evaporator |
  | 9 | mine_dump | Mine Dump, Ore Stockpile/Storage |
  | 10 | disturbed_surface | Disturbed Surface, Disturbed Surface - Pit, Trench |

- **Dropped `Ftr_Type` values** (kept the class set meaningful; each contributes few
  samples and we cap at 1000/class anyway): Clay/Cinder/Shale/Caliche/Scoria/Chert/Marl/
  Bentonite/Shell/Iron/Lignite Pit, Silica Mine, generic Mine, Coal/Uranium/Placer/
  Hydraulic Mine, Mill Site, Tipple.

## Observability caveat (10 m Sentinel-2/1/Landsat)

- **Resolvable:** quarry_open_pit, strip_mine, gravel_borrow_pit, tailings_pond,
  tailings_pile, mine_dump, disturbed_surface — these are largely polygon footprints tens
  to thousands of metres across.
- **Weak (presence-only):** prospect_pit, mine_shaft, adit are point-only and typically
  sub-10 m; their detection tiles function as "a mine feature is present near here" targets,
  not resolvable footprints. Kept per task instruction (detection encoding with feature-type
  classes), flagged here.

## Sampling & time

- Up to **1000 per feature class** (10 classes → 10,000 tiles), + **500 background-only
  negative tiles** (centers offset 3–15 km from any feature, verified feature-free within
  1 km). Total **10,500** — well under the 25k cap.
- Realized per-class geometry mix: quarry/gravel/strip/tailings/dump/disturbed = 1000
  polygons each; prospect_pit = 76 poly + 924 pt; mine_shaft & adit = 1000 pt each
  (no polygons of those types exist).
- Coverage is **national** (v10 covers the whole US; manifest said "Western US" but the
  current release is nationwide — kept all for a richer label bank). Nevada/California/
  Colorado/Arizona dominate the point classes.
- **Time range:** features are persistent and map-digitized (no Sentinel-era acquisition
  date). Per spec §5 (static labels) each sample gets a 1-year window at a representative
  year pseudo-randomly spread across **2016–2022** for temporal diversity. `change_time` is
  null.

## Verification

- 10,500 `.tif` + 10,500 `.json`, fully paired. All tiles single-band uint8, local UTM at
  10 m, ≤64×64, nodata=255; sampled pixel values all within {0–10, 255}. Max JSON time span
  366 days (leap-year window). `metadata.json` classes cover all label values.
- **S2 overlay:** sample 003000 (quarry, Utah, −111.929/40.328) over cloud-free 2021-09-22
  Sentinel-2 shows bright bare ground (red≈1759) and low NDVI (0.13), consistent with an
  open-pit. A Florida quarry sample (003002) had higher NDVI (0.61) — Florida pits are often
  flooded/vegetated and the 42-px tile also captures surrounding vegetation; location is
  correct. Georeferencing (CRS/bounds/time) validated.

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.usgs_usmin_mine_features
```
Idempotent (skips already-written `locations/{id}.tif`).
