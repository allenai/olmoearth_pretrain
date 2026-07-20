# FLOGA

- **Slug**: `floga`
- **Status**: **completed** — task_type = **classification** (dense per-pixel, binary), **1057 samples**
- **Family / region**: fire / Greece
- **Source**: FLOGA — Sdraka et al. 2024, *IEEE JSTARS*, doi:10.1109/JSTARS.2024.3381737.
  Repo: https://github.com/Orion-AI-Lab/FLOGA . Labels from the companion label-only
  release **`Orion-AI-Lab/FLOGA-annotations`** (https://github.com/Orion-AI-Lab/FLOGA-annotations).
- **License**: open for research (repo: MIT + CC-BY-4.0).
- **Access**: public GitHub, no credentials.

## What the dataset is

FLOGA is an ML-ready **Sentinel-2 + MODIS** dataset of **326 Greek wildfire events**
(2017–2021) with high-resolution **burnt-area ground truth produced by the Hellenic Fire
Service** (expert manual annotation). For each event it offers pre/post-fire imagery, cloud
/ sea / CLC masks, and a burnt-area label (values 0 = non-burnt, 1 = burnt, 2 = burnt in
*other* same-year events).

Pretraining supplies its own imagery, so **only the labels are needed**. The full ML-ready
v2 GeoTIFF product on HuggingFace (`orion-ai-lab/FLOGA-GeoTIFFs`, 13 tars ≈ **130 GB**) is
mostly Sentinel-2/MODIS imagery; downloading it to extract a thin label layer is
impractical (spec §2/§8 impractical-download guidance). Instead we use the **v2 annotation
polygons** (`polygons/v2/fb_{2017..2021}_final_images_4326.shp`, a few MB total): per-event
**burnt-area polygons** in EPSG:4326 carrying the **wildfire ignition date** (`Start date`),
`End date`, and the exact pre/post Sentinel-2/MODIS/Sentinel-1 images used.

After deduplicating by `ID` (the multiple rows per event are alternate Sentinel-1 pairs
that share one geometry), there are **344 unique events** (2017:66, 2018:26, 2019:61,
2020:85, 2021:106); ignition dates span **2017-06-29 → 2021-09-25** (all post-2016; none
filtered on the pre-2016 rule).

## Class scheme (dense per-pixel classification)

| id | name     | definition |
|----|----------|------------|
| 0  | unburned | land within the event footprint outside the burnt-area polygon (observed, non-burnt) |
| 1  | burned   | inside the Hellenic Fire Service burnt-area polygon for the event |
| 255| nodata   | inside a **different same-year event's** burnt-area polygon (FLOGA's "value 2 = burnt in other events") → ignored so a neighbouring fire's scar is never mislabeled `unburned` |

The manifest's two classes ("burned", "unburned") map directly; ids follow the fire-dataset
convention (cf. `cabuar_california_burned_areas`: 0 unburned, 1 burned).

## Processing (label_type = dense_raster, from polygons)

- Each event polygon is reprojected to its **local UTM zone at 10 m** (EPSG:32634 / 32635
  for Greece), and its bounding box (padded by **1 tile**) is tiled into **64×64** patches
  (640 m).
- Each patch is rasterized (`rasterio.features.rasterize`) with **other intersecting
  same-year events painted first as 255**, then this event as **1**, `fill = 0`.
- **Sampling**: tiles-per-class balanced (spec §5), ≤ **1000 tiles/class**, rarer class
  filled first, under the 25k cap. From **20,325 candidate tiles**, **1057** were selected
  (**1017 contain burned**, **1000 contain unburned**; mixed tiles count toward both), from
  **233 distinct events**. Selected-sample change-years: 2017:138, 2018:73, 2019:88,
  2020:123, 2021:635 (2021 dominates — the catastrophic August-2021 Evia/Attica megafires).

## Time-range / change handling

Burnt area is a **change/event** label with a **day-precise ignition date**, so this is
processed as a change label (spec §5), not a static presence class:
- `change_time` = the event's `Start date` (ignition), known to the day (≪ the ~1–2-month
  timing-precision requirement); retained as the reference used to build the windows.
- Instead of a single centered window, each sample emits two independent six-month windows:
  a `pre_time_range` (the ≤183 days immediately **before** `change_time`) and a
  `post_time_range` (the ≤183 days immediately **after** it), with `time_range` set to null.
  The windows are adjacent and split exactly at `change_time` (built via
  `io.pre_post_time_ranges(change_time, ...)`), so pretraining pairs a "before" image stack
  with an "after" stack and probes on their difference. The post-fire Sentinel-2 acquisition
  recorded in the shapefile (`S2_e`) lands a few weeks after ignition, comfortably inside the
  post window, so the after-stack spans the burn and the where-mask stays aligned.

## Output

- `datasets/floga/metadata.json` (2 classes, nodata 255).
- `datasets/floga/locations/{id}.tif` — single-band uint8, local UTM, 10 m, 64×64.
- `datasets/floga/locations/{id}.json` — crs / pixel_bounds / time_range / change_time /
  source_id (`{eventID}_r{row}_c{col}`) / classes_present.

## Verification (spec §9)

- All 1057 `.tif` are single-band **uint8, 64×64, UTM (EPSG:32634/32635) at 10 m**; pixel
  values ∈ {0, 1, 255}; every `.tif` has a matching `.json`.
- `time_range` is null with adjacent `pre_time_range` / `post_time_range` (each ≤183 days)
  split at `change_time`; `change_time` set on every sample; no pre-2016 windows.
- Spatial sanity: 100 % of tile centers fall inside the Greece bbox (lon 20.23–28.09,
  lat 35.08–41.40); sampled locations/dates match known Greek wildfires (e.g. lon 23.27
  lat 38.84 on 2021-08-03 = the North-Evia fire). Labels are rasterized directly from
  georeferenced expert polygons (not derived from imagery), so georeferencing is exact by
  construction; a full Sentinel-2 visual overlay was not rendered in this headless run.
- Re-running the script is idempotent (existing `{id}.tif` are skipped).

## Caveats

- `unburned` (0) is the surrounding non-burnt land within each event footprint; because we
  rasterize per-event polygon bboxes (not the FLOGA sea/cloud masks, which live only in the
  130 GB imagery), a small number of coastal tiles could include sea labeled `unburned`.
  Impact is minor (tiles are centered on land burn footprints) and handled downstream.
- Cross-year overlapping burns are not masked to 255 (only same-year, matching FLOGA's own
  value-2 semantics); a location that burned in a prior year appears as `unburned` for a
  later event, which is correct for that event's date.

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.floga --workers 64
```
Downloads the v2 annotation shapefiles into
`raw/floga/` (label-only; imagery not downloaded) and writes all outputs.
