# RiverScope — COMPLETED (classification, dense_raster)

- **Slug**: `riverscope`
- **Name**: RiverScope
- **Source**: RiverScope: High-Resolution River Masking Dataset (UMass CVL),
  Zenodo record `15376394` (https://zenodo.org/records/15376394); AAAI 2025.
  Docs/code: https://github.com/cvl-umass/riverscope.
- **Family / region**: river / global (1,145 scenes; SWORD-referenced reaches worldwide).
- **License**: CC-BY-4.0 (Zenodo record; the manifest lists CC0-1.0 — either way
  redistribution/use is permitted).
- **Label type**: dense_raster → per-pixel **classification**.
- **Task type**: classification. **num_samples**: 1317 tiles.

## Source

Expert-labeled (15 hydrology experts) water-segmentation masks over 1,145
PlanetScope scenes, co-registered with Sentinel-2, SWORD and SWOT. The 8.05 GB
`RiverScope.zip` unpacks to `RiverScope_dataset/` with four modality folders
(PlanetScope, Sentinel-2, SWORD, SWOT) and `train/valid/test.csv` split files
(787 / 123 / 235 = 1,145 rows). We consume only the label rasters in
`PlanetScope/label/{split}/*.tif`.

Each label GeoTIFF is single-band **float32, 500×500, 3 m/pixel**, already in a
local UTM CRS (per-scene, e.g. EPSG:32638), with pixel values:

- `0.0` = background (non-water)
- `1.0` = river water
- `2.0` = non-river / other water

(a large-negative float `-3.4e38` is the source nodata fill, mapped to 255). The
class scheme matches the manifest `[background, river, other water]` exactly.

RiverScope also carries SWORD/SWOT node **widths** (for width estimation), but
those are node attributes, not a dense raster — the raster product we ingest is
the categorical water mask, so this is **classification**, not width regression.

## Access method

Public, no credentials. `RiverScope.zip` downloaded once from the Zenodo record
to `raw/riverscope/RiverScope.zip`. The script selectively extracts only
`PlanetScope/label/**` + the three split csvs (the 8 GB of PlanetScope/Sentinel-2
imagery, SWORD shapefiles and SWOT pixel clouds are not needed for labels).

## Class mapping (3 classes, manifest order)

| id | name | definition |
|----|------|-----------|
| 0 | background | Non-water land surface (source value 0) |
| 1 | river | River water of the labeled reach (source value 1) |
| 2 | other water | Non-river open water — lakes/ponds/tributaries/other water in the tile (source value 2) |
| 255 | nodata/ignore | Source nodata fill + reprojection padding |

Source values map directly (0→0, 1→1, 2→2); any other value → 255.

## Processing (VHR-native 3 m → 10 m, spec §4)

Each 500×500 3 m label is reprojected **once** to its local UTM zone at 10 m with
**nearest** resampling (categorical; never bilinear), giving a ~150×150 px valid
region (padded up to whole 64-px tiles, ~192×192), then cut into **64×64** tiles.
Tiles >50% nodata are dropped; a tile counts toward a class only with ≥32 px of
it. **Tiles-per-class balanced** selection (spec §5): rare classes filled first up
to 1000 tiles/class; a tile contributes to every class it contains. 4,567
candidate tiles → **1,317 selected**. All three source splits are used.

**Tiles containing each class** (a tile can count for several):

- background: 1169
- river: 1000
- other water: 730

Well under the 25k per-dataset cap.

## Time range & change handling

Each label is the water extent at one PlanetScope acquisition; the acquisition
date is the leading `YYYYMMDD` of `planetscope_id`. Water extent is seasonally
variable, so `time_range` is a **1-year window centered on the acquisition date**
(spec §5, seasonal/annual). **No `change_time`** — the river channel is a
persistent feature, not a dated change event. All acquisitions are 2023–2024
(within the Sentinel era).

## Verification

- 1,317 `.tif` + 1,317 matching `.json` (no unpaired ids). Every tile: single-band
  **uint8**, local UTM, **10 m**, **64×64**, values ⊆ {0,1,2,255}, nodata=255.
- Every sample `time_range` ≤ 365 days; `change_time` null; `metadata.json` class
  ids {0,1,2} cover all values in the tifs.
- **Georeferencing round-trip**: for 4 river tiles, pixels labeled river (1) were
  reprojected UTM→WGS84 and the **source** PlanetScope label read `1.0` (river) at
  those coordinates; tile centers land 0.27–0.67 km from the CSV `mid_lon/mid_lat`
  (well within the ~1.5 km scene footprint). Georeferencing is exact.
- Re-running skips already-written tiles (idempotent).

## Caveats

- **Sub-10 m rivers**: RiverScope's value is fine-scale (3 m) river detail; rivers
  narrower than ~10 m can thin or vanish after resampling to 10 m. Tiles still
  retain the wider channels and are a good match for Sentinel-2 (10 m), as the
  manifest notes.
- ~1/3 of source scenes are **all background** (experts marked no water in that
  PlanetScope crop — dry/narrow/off-channel); these legitimately contribute
  background tiles and are kept.
- "other water" (id 2) is the rarest class (730 tiles); river and background reach
  their caps. No class was dropped.

## Reproduce

```
# (one-time) download the 8 GB archive to raw/riverscope/RiverScope.zip:
#   curl -L -o RiverScope.zip "https://zenodo.org/records/15376394/files/RiverScope.zip?download=1"
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.riverscope --workers 64
```
