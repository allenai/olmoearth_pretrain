# MTBS (Monitoring Trends in Burn Severity)

- **Slug**: `mtbs_monitoring_trends_in_burn_severity`
- **Source**: USFS/USGS interagency MTBS program (https://www.mtbs.gov/). Public domain.
- **Label type**: `dense_raster / polygons` → processed as **dense multi-class burn-severity segmentation** (task_type: `classification`).
- **Status**: completed — **1,886** label tiles.

## What the source is

MTBS maps burn severity and perimeters for all large fires in the US (≥ ~1000 ac West,
≥ ~500 ac East) from analyst-reviewed differenced Normalized Burn Ratio (dNBR). Two
products are combined here:

1. **Burned Areas Boundaries** — national perimeter shapefile `mtbs_perims_DD.shp`
   (EPSG:4269). One polygon per fire event with `event_id` (state-prefixed id),
   `ig_date` (ignition date, `YYYY-MM-DD`, **day precision**), `incid_type`
   (Wildfire / Prescribed Fire / Other), acreage, etc. Provides each fire's **date** and
   **extent**. 30,613 features total; 8,770 with `ig_date` year ≥ 2016.
2. **Thematic Burn Severity Mosaics** — annual national 30 m rasters, one per (year, region):
   CONUS (`ESRI:102039`), AK (`EPSG:3338`), HI (Hawaii Albers). Pixel values
   `0`=background(outside fires), `1`=Unburned to Low, `2`=Low, `3`=Moderate, `4`=High,
   `5`=Increased Greenness, `6`=Non-Mapping Area(mask). This is the per-pixel severity
   signal that distinguishes MTBS from a plain binary burn-scar dataset (e.g.
   `cal_fire_frap_fire_perimeters`).

## Access method

- Boundaries: direct download
  `https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/MTBS_Fire/data/composite_data/burned_area_extent_shapefile/mtbs_perimeter_data.zip`
  (380 MB, no credentials).
- Mosaics: ScienceBase parent item `5e91dee782ce172707f02cdd` has per-year child items;
  each exposes `mtbs_{REGION}_{YEAR}.zip` (CONUS/AK/HI). Downloaded via the ScienceBase
  file API (`?format=json&fields=files` → per-file `downloadUri`). Each CONUS zip is only
  ~3-11 MB (severity is spatially sparse and PackBits-compressed), so all years fit in a
  few hundred MB — no national bulk raster needed.
- Raw files under `raw/mtbs_monitoring_trends_in_burn_severity/`:
  `mtbs_perimeter_data.zip`, `perim_extract/`, `mosaics/*.zip`, `mosaic_tifs/*.tif`,
  `mosaic_uris.json`, `SOURCE.txt`.

## Class mapping

MTBS mosaic value → our compact uint8 class id (severity classes only; §5 positive-only
foreground — no fabricated background):

| id | name | MTBS value |
|----|------|-----------|
| 0 | unburned_to_low | 1 |
| 1 | low | 2 |
| 2 | moderate | 3 |
| 3 | high | 4 |
| 4 | increased_greenness | 5 |
| 255 | nodata/ignore | 0 (background), 6 (non-mapping), and pixels outside the fire's own perimeter |

Each tile's severity is read from the fire's year+region mosaic, reprojected to a local
UTM grid at 10 m (nearest resampling — categorical), and **masked to that fire's own
perimeter polygon** so the severity is attributable to that fire's ignition date. Pixels
outside the perimeter → 255.

## Time-range / change handling (§5)

A fire is a dated **CHANGE** event. `change_time` = `ig_date` (day precision, well within
the ≤1-2 month requirement); `time_range` is a **360-day window centered on** `change_time`
(±180 d). Verified: sampled tiles have `time_range` span = 360 d and center offset = 0 d
from `change_time`. Pretraining will only use a sample when the sampled input window spans
the fire date, so the where-severity mask aligns with before/after imagery.

Only `ig_date` years **2016-2024** are used: pre-2016 perimeters are filtered out
(Sentinel era), and 2025-2026 (26 fires, no mosaic yet) are dropped.

## Tiling / sampling

- Tile size 64×64 (640 m), hard cap 64. A fire fitting a tile → one centered tile; larger
  fires are gridded into non-overlapping 64×64 windows, keeping windows intersecting the
  perimeter, sampling up to `MAX_TILES_PER_FIRE=20` per fire.
- **Tiles-per-class balanced** selection (`select_tiles_per_class`, rarest class first) up
  to **1000 tiles/class**, capped at the 25,000 per-dataset limit (§5). With only 5 classes
  the 25k cap is never binding; the 1000/class rule yields 1,886 unique tiles.
- Both **Wildfire and Prescribed Fire** events are kept (both are dated burn events with
  real severity); `incid_type` is recorded in `source_id`.

## Counts

- Fires used: **7,689** (CONUS 7,402, AK 272, HI 15) with a covering mosaic.
- Candidate severity tiles: 129,697 → **1,886 selected**.
- Tiles containing each class: unburned_to_low 1,789; low 1,872; moderate 1,172;
  high 1,000; increased_greenness 1,081. (A tile counts toward every class present in it.)

## Verification (§9)

- All 1,886 `.tif`: single band, `uint8`, local UTM at 10 m, ≤64×64, nodata 255; pixel
  values only in {0,1,2,3,4,255}. Every `.tif` has a matching `.json`.
- `metadata.json` class ids cover all values in the tifs.
- `change_time` set on every sample; `time_range` = 360 d centered on it (offset 0 d).
- Geolocation spot-check: tile centers match each fire's state and the lat/lon embedded in
  its `event_id` (e.g. CO CalWood Fire → (-105.33, 40.13); AK Telaquana River →
  (-154.26, 61.00); FL prescribed burn → (-84.78, 30.06)).

## Caveats

- **Two mosaic files could not be downloaded** during processing due to a transient
  Cloudflare/ScienceBase 404 on their file blobs: `mtbs_CONUS_2017.zip` and
  `mtbs_AK_2019.zip`. Their fires are skipped (2017 dropped from ~983 to 37 fires; AK 2019
  from ~50 to 0). This did **not** compromise class coverage — every severity class reaches
  its 1000-tile target from the other 23 mosaics (2016 + 2018-2024). To incorporate them
  later, re-download those two ScienceBase blobs (item ids `5e921ab182ce172707f02d03` and
  `62b462eed34e8f4977cbcea6`) into `raw/.../mosaics/` and re-run the script.
- Prescribed fires are included alongside wildfires; if a wildfire-only variant is ever
  needed, filter `source_id` on `:Wildfire:`.
- Severity mosaics are 30 m natively (upsampled to 10 m with nearest); fine severity mosaic
  edges are coarser than the 10 m tile grid.
- Only CONUS/AK/HI mosaics exist (no Puerto Rico mosaic); no PR fires appear in the
  post-2016 subset anyway.

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.mtbs_monitoring_trends_in_burn_severity
```
Idempotent: existing `locations/{id}.tif` are skipped; a full re-run regenerates the
selection deterministically (seed 42).
