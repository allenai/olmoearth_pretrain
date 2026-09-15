# Coastal Aquaculture Ponds (China & SE Asia)

- **Slug:** `coastal_aquaculture_ponds_china_se_asia`
- **Status:** completed
- **Task type:** classification (binary polygon segmentation)
- **Samples written:** 996 label tiles (64×64, 10 m, local UTM)

## Source

Zenodo record [10370830](https://zenodo.org/records/10370830) — *"Fine-detailed coastal
aquaculture pond dataset in China and Southeast Asia in 2020 at a 30 m resolution"*
(CLAP_CSEA_2020; Duan et al., MDPI *Remote Sensing*). Aquaculture-pond footprints were
derived from the long time-series Landsat archive with an **object-oriented hierarchical
classification** method using manual training samples, for the year **2020**.

Distributed as a single `CLAP_CSEA_2020.rar` (~138 MB) containing **per-country ESRI
shapefiles** of pond polygons in Albers equal-area projections (ESRI:102025 for China,
ESRI:102028 for the SE-Asia countries): Brunei, Cambodia, China, Indonesia (×2 tiles),
Malaysia, Myanmar, Philippines, Singapore, Thailand, Timor-Leste, Vietnam.
Total ~636k pond polygons. License: open (Zenodo open access, CC-BY).

### Access / reproduction

Downloaded via `download.download_zenodo("10370830", ...)`; the `.rar` was extracted with
`bsdtar` (no `unrar`/`rarfile` available). Raw at
`raw/coastal_aquaculture_ponds_china_se_asia/CLAP_CSEA_2020/`.

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.coastal_aquaculture_ponds_china_se_asia
```

Idempotent: existing `locations/{id}.tif` are skipped; the raw archive is downloaded and
extracted once.

## Class mapping

The source is a **complete coastal map** (object-based classification of whole scenes), so
non-pond is a real mapped class, not merely an ignore region.

| id | name | meaning |
|----|------|---------|
| 0 | non-pond | Any mapped coastal pixel that is not an aquaculture pond. |
| 1 | aquaculture pond | Coastal fish/shrimp/crab aquaculture pond footprint. |

`255` reserved for nodata only (not used inside tiles). Both classes appear per tile
(binary segmentation): pond (1) is present in all 996 tiles; non-pond (0) in 987 (9 tiles
sit in dense pond clusters that fill the 640 m window). Mean pond-pixel fraction ≈ 0.31.

## Sampling (bounded — spec §5)

~636k polygons is a large set, so we sample. Every pond **centroid** is snapped to a
**640 m grid** (= one 64 px × 10 m output tile) in the country's Albers CRS; the set of
occupied cells (dedups dense clustering: ~636k polygons → **144,019 cells**) is pooled
across all countries and **1000 cells are uniformly sampled** (seed 42). Uniform pooled
sampling makes the selection track real pond density — China (63,911 cells), Vietnam
(27,493), Indonesia (27,441), Thailand (15,867), Philippines (6,587) dominate; small
countries (Brunei 93, Timor 8, Singapore 2) contribute proportionally.

For each sampled cell one **64×64 tile** is built in local UTM at 10 m, centered on the
cell; every pond polygon intersecting the tile (fetched via a bbox spatial-index query on
the source shapefile, reprojected Albers→UTM pixels) is rasterized as class 1 over a
class-0 background. 4 of 1000 cells produced no resolvable pond pixels in the tile and were
skipped → **996 tiles**.

## Time range

Annual 2020 product → each tile gets a **1-year window `[2020-01-01, 2021-01-01)`**.
Persistent land use, so `change_time = null` (presence classification, not a dated event).

## Verification

- 996 `.tif` each with a matching `.json`; all single-band **uint8**, **64×64**, UTM
  (EPSG:326xx/327xx) at 10 m north-up; values ∈ {0, 1}; nodata 255 declared.
- Every `.json` has a ≤1-year `time_range` on 2020 and `classes_present` matching the tif.
- Tile-center coordinates span **99.6–121.5°E, -7.2–38.5°N** — coastal China + SE Asia
  (Pearl River delta, Mekong delta, Bohai Bay, Java, Malay Peninsula), matching the source
  region. (A full Sentinel-2 pixel overlay was not run; correctness rests on the verified
  reprojection path — identical to the seagrass/salt-pond polygon references — and the
  coordinate sanity check above.)

## Caveats

- Ponds mapped natively at **30 m** are rasterized at 10 m (nearest); very small or thin
  ponds may under-register, and pond edges carry ~30 m positional uncertainty.
- Uniform pooled sampling means the tiny countries (Brunei/Timor/Singapore) contribute few
  or no tiles; coverage is concentrated where aquaculture ponds actually cluster.
- The map only covers coastal study areas, so class-0 (non-pond) reflects coastal land
  cover near ponds, not arbitrary background.
