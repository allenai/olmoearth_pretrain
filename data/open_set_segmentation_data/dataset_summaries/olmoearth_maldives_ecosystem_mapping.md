# OlmoEarth Maldives ecosystem mapping

- **Slug**: `olmoearth_maldives_ecosystem_mapping`
- **Task type**: classification (dense, tiled polygon rasters)
- **Status**: completed — **94 label patches**, 16 classes
- **Family / region**: ecosystem / Maldives
- **License**: internal

## Source

Local rslearn project (`have_locally: true`, not copied):
`/weka/dfive-default/rslearn-eai/datasets/maldives_ecosystem_mapping/dataset_v1/20240924`

The `crops` group holds **91 manually annotated crops** (Kili annotations) of coastal/marine
ecosystem types using the **IUCN Global Ecosystem Typology (GET)**, rasterized over **Maxar
VHR** imagery. Each crop is a ~1 km patch in local UTM (**EPSG:32643, zone 43N**) at the
Maxar native resolution (**~0.35–0.49 m/pixel**, varies per scene) with a ~2-minute
acquisition `time_range`. The paired image layer is `maxar` (R,G,B); Sentinel-2 / PlanetScope
/ SkySat variants also exist but were not needed here.

Legend source: `rslp.maldives_ecosystem_mapping.config.CATEGORIES` (in
`rslearn_projects`). The label raster is single-band uint8 where the pixel value
is the `CATEGORIES` index: **0 = "unknown" (unannotated)** and **1..16 = the 16 IUCN GET
classes**. Rasterization fill is 0, so all non-polygon pixels inside a crop are 0.

## VHR handling (per spec)

The label is VHR-native and each crop is far larger than 64 px at 10 m, so:

1. **Resample to 10 m with NEAREST** (never bilinear — categorical). Read via
   `GeotiffRasterFormat().decode_raster(..., resampling=Resampling.nearest)` into a
   `Projection(EPSG:32643, 10, -10)` (the source CRS is reused since it is already UTM).
   Target pixel bounds = source pixel bounds × (source_res / 10).
2. **Tile** each crop into ≤64×64 patches (step 64; edge tiles are smaller). Crops are up to
   ~110×110 px at 10 m → typically 2×2 tiles.
3. **Remap**: source `0` (unknown) → **nodata 255**; source ids `1..16` → **output ids
   0..15**. Tiles with zero labeled pixels are dropped (no signal). 94 non-empty tiles result.
4. **Time range**: 1-year window **centered on the Maxar image date** (`±180 days`, = 360
   days; ≤ 1 year). Image dates span 2023–2024.

## Class mapping (output id → IUCN GET) and tile counts

A tile counts toward every class present in it. Dataset is small (91 crops) so **all tiles are
kept** — no per-class subsampling needed (all far below the 1000/class cap).

| id | IUCN GET | name | tiles |
|---:|---|---|---:|
| 0 | FM1.3 | Intermittently closed and open lakes and lagoons | 12 |
| 1 | F2.2 | Small permanent freshwater lakes | 6 |
| 2 | MFT1.2 | Intertidal forests and shrublands (mangroves) | 13 |
| 3 | MFT1.3 | Coastal saltmarshes and reedbeds | 2 |
| 4 | MT1.1 | Rocky shorelines | 8 |
| 5 | MT1.3 | Sandy shorelines | 55 |
| 6 | MT2.1 | Coastal shrublands and grasslands | 69 |
| 7 | MT3.1 | Artificial shorelines | 24 |
| 8 | M1.1 | Seagrass meadows | 24 |
| 9 | M1.3 | Photic coral reefs | 10 |
| 10 | M1.6 | Subtidal rocky reefs | 10 |
| 11 | M1.7 | Subtidal sand beds | 53 |
| 12 | TF1.3 | Permanent marshes | 4 |
| 13 | T7.1 | Annual croplands | 16 |
| 14 | T7.3 | Plantations | 12 |
| 15 | T7.4 | Urban and industrial ecosystems | 67 |

## Suitability assessment at 10 m

All 16 classes survive 10 m nearest resampling with a real pixel footprint, so the full IUCN GET
class set was **kept** rather than dropped/coarsened. Rationale and caveats:

- **Well resolved at 10 m** from Sentinel-2/Landsat: the broad shallow-water benthic classes
  (**Seagrass meadows, Photic coral reefs, Subtidal sand beds, Subtidal rocky reefs**) — the
  clear Maldivian atoll waters are a canonical case for S2 benthic habitat mapping — plus the
  terrestrial/areal classes (**Coastal shrublands/grasslands, Urban/industrial, Plantations,
  Annual croplands, Mangroves, ICOLL lagoons, Freshwater lakes**).
- **Low-confidence / marginal at 10 m (kept but noisy)**: **Rocky shorelines**, **Sandy
  shorelines**, **Artificial shorelines** are narrow linear intertidal features often <10 m
  wide — after nearest resampling they become thin 1–2 px strips. **Coastal saltmarshes and
  reedbeds** (2 tiles) and **Subtidal rocky reefs** are rare and/or spectrally similar to
  neighbours. These are flagged in `metadata.json.notes`; a downstream consumer may choose to
  merge the three shoreline classes or ignore the rarest ones.
- The spec called out "subtidal zonation" as a risk. The subtidal classes are retained because
  broad benthic zonation (seagrass vs sand vs coral) is mappable at 10 m in shallow clear water;
  what is *not* recoverable is finer within-zone structure, which the label did not encode anyway.

## Verification

- 94 `.tif` + 94 `.json`. All tifs: single band, uint8, EPSG:32643, 10 m, ≤64×64, nodata 255,
  values ⊂ {0..15, 255}. All `time_range`s = 360 days. `metadata.json` class ids cover all tif
  values.
- **Spatial/semantic sanity** (tile `000000`, Baarah): label overlaid on the co-registered
  Maxar RGB decoded at the identical 10 m CRS/bounds. Labeled pixels sit on real imagery, and
  per-class mean RGB is ecologically sensible — Sandy shorelines bright (~150), Seagrass
  bluish-green and darker (76,84,83), Mangroves/shrublands dark green (~57), confirming
  label↔image registration.

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.olmoearth_maldives_ecosystem_mapping
```

Idempotent: existing `locations/{id}.tif` are skipped; sample ids are assigned by sorted
`(crop, row, col)` so re-runs are stable.

## Outputs

- `raw/olmoearth_maldives_ecosystem_mapping/SOURCE.txt`
- `datasets/olmoearth_maldives_ecosystem_mapping/metadata.json`
- `datasets/olmoearth_maldives_ecosystem_mapping/locations/{000000..000093}.{tif,json}`

(all under `/weka/dfive-default/helios/dataset_creation/open_set_segmentation/`)
