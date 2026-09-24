# GSDP30 (Global Sand Dune Patterns)

- **Slug:** `gsdp30_global_sand_dune_patterns`
- **Status:** completed
- **Task type:** classification (dense_raster)
- **Samples:** 7,132 label patches (64×64, UTM, 10 m)

## Source

Zhang et al. 2024, *ISPRS J. Photogramm. Remote Sens.* **218**:781–799, "Global
perspectives on sand dune patterns: Scale-adaptable classification using Landsat imagery
and deep learning strategies." Data: Zenodo record 13907012
(https://zenodo.org/records/13907012), license **CC-BY-4.0**.

GSDP30 is a global 30 m per-pixel classification of aeolian **sand-dune-pattern (SDP)
morphology**, produced with a SegFormer deep-learning model applied to **2017 Landsat-8**
surface-reflectance composites (built on the earlier GSDS30 sand-dune/sheet mask).
Distributed as **331 GeoTIFF tiles**, each 15,360 × 15,360 px, **EPSG:3857** (Web
Mercator), 30 m, uint8, nodata=255, named by the lon/lat of their upper-left corner. Each
tile covers a ~460 km square over the world's sand seas (ergs).

Access: unauthenticated Zenodo API download (`download.download_zenodo("13907012", ...)`),
one 115 MB zip → 331 tiles (315 MB unzipped) under
`raw/gsdp30_global_sand_dune_patterns/GSDP30/`. No credentials required.

## Class mapping (11 SDP classes, values 0–10)

The Zenodo description lists the 11 classes in an explicit order; the raster carries
exactly the values 0–10. **We map value == description-list index.** This ordering is a
documented judgment call (the product ships no colormap, band tags, or codebook, and the
paper is paywalled), but it is strongly supported by the global pixel frequencies, which
are geomorphologically self-consistent with that order:

| id | class | global pixels | note |
|----|-------|--------------:|------|
| 0 | simple crescentic dunes | 568 M | |
| 1 | compound-complex crescentic dunes | 445 M | |
| 2 | simple linear dunes | 3.06 B | most extensive true dune form |
| 3 | compound-complex linear dunes | 653 M | |
| 4 | dome dunes | 217 M | rarest true dune (as expected) |
| 5 | star dunes | 293 M | rare (multidirectional-wind form) |
| 6 | parabolic dunes | 581 M | |
| 7 | dendritic dunes | 403 M | |
| 8 | network dunes | 3.47 B | extensive in large sand seas |
| 9 | sand sheets | 68.3 B | ~87% of mapped pixels; extensive flat-sand surface, behaves like a background/extensive class |
| 10 | others | 106 M | small residual |
| 255 | nodata | — | outside the mapped sand domain; kept as `CLASS_NODATA` |

`nodata_value = 255`. Class 9 (sand sheets) fills the great majority of every tile and
functions as the extensive-surface/background class; class 10 (others) is a small
residual. Per the spec we keep all 11 classes (no dropping of sparse classes).

## Processing

GSDP30 is a large global derived-product map, so per §4/§5 we do **bounded
tiles-per-class-balanced** sampling:

1. Scan all 331 source tiles in native 30 m **BLOCK = 21 px** blocks (~630 m ≈ a 64 px @
   10 m UTM footprint), 32-worker `multiprocessing.Pool`.
2. A block's `classes_present` = SDP ids occupying **≥ 10 %** (`MIN_FRAC`) of the block.
   Drop blocks with **> 20 %** nodata (`MAX_NODATA_FRAC`) and the outer border ring of
   blocks (straddle guard against adjacent Web-Mercator tiles). To bound memory, keep up
   to 15 candidate blocks per (tile, class) — abundant, since even the rarest classes have
   thousands of candidate blocks.
3. `select_tiles_per_class(per_class=1000, total_cap=25000)` — rarest-class-first, a tile
   counts toward every class it contains → 7,132 patches selected.
4. Each selected block's ~630 m footprint is reprojected from native 30 m EPSG:3857 to a
   **local UTM projection at 10 m** with **nearest** resampling (categorical labels) into
   a **64×64** multi-class uint8 patch; source 255 → 255.

Output: single-band uint8 GeoTIFFs `locations/{id}.tif` + sidecar `.json`, plus
`metadata.json`.

## Time range

The map was produced from **2017** Landsat-8 composites, so each sample gets a **1-year**
window `[2017-01-01, 2018-01-01)`. All labels are post-2016 (Sentinel era) — nothing
filtered on the pre-2016 rule. Dune morphology is quasi-static; no `change_time`.

## Sample counts (tiles per class; a tile counts toward every class present)

```
simple crescentic dunes            1047
compound-complex crescentic dunes  1088
simple linear dunes                1036
compound-complex linear dunes      1095
dome dunes                         1036
star dunes                         1056
parabolic dunes                    1090
dendritic dunes                    1093
network dunes                      1000
sand sheets                        1172
others                              306   (sparse; kept per spec, downstream may filter)
```
Total distinct patches: **7,132** (well under the 25k cap).

## Verification

- 7,132 `.tif` each with a matching `.json`; all single-band uint8, local UTM at 10 m,
  64×64, nodata=255; pixel values ∈ {0..10, 255}; all 11 class ids appear across the
  corpus and are covered by `metadata.json`.
- All sample `time_range`s are the 1-year 2017 window; `change_time` null.
- **Spatial sanity check:** 12 random sample centers were reprojected to lon/lat and all
  fall squarely in known sand seas / drylands — Taklamakan, Karakum, Gobi (Mongolia),
  Sahara (Mauritania), An Nafud & Rub' al Khali margins (Saudi/Yemen), Great Sandy Desert
  (W Australia), Tengger/Mu Us (China). A full Sentinel-2 pixel overlay was **not**
  performed because dune-morphology *subtype* (dome vs star vs network) is not visually
  separable at 10 m S2 anyway; geographic placement in active dune fields is the
  meaningful available check.

## Caveats / judgment calls

- **Value→class mapping** follows the Zenodo description order (value == index). No
  authoritative codebook was available (no colormap/tags; paper paywalled), but the
  frequency structure is geomorphologically consistent with the order. If the true legend
  differs, only the class *names* change — the raster geometry and IDs are correct.
- **Class 9 (sand sheets)** dominates (~87 % of pixels) and behaves as a
  background/extensive-surface class; class 10 (others) is a small residual (306 tiles).
  Both kept per the "don't drop sparse/rare classes" rule.
- Labels are **derived-product / DL predictions** (reported ~85 % overall accuracy), not
  in-situ reference; sampling favors blocks with ≥10 % of a class to avoid single-pixel
  noise, but the labels carry the source model's error.

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.gsdp30_global_sand_dune_patterns
```
Idempotent: re-runs skip already-written `locations/{id}.tif`. Raw pulled from Zenodo
record 13907012 (no credentials). Runtime ≈ 3–4 min on 32/64 workers.
