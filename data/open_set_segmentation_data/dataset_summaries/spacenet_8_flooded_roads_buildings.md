# SpaceNet 8 (flooded roads & buildings)

- **Slug**: `spacenet_8_flooded_roads_buildings`
- **Status**: completed
- **Task type**: classification (change-labeled)
- **Samples**: 856 label patches (64×64, UTM, 10 m)

## Source & access

SpaceNet 8 Flood Detection Challenge, hosted on the public AWS Open Data bucket
`s3://spacenet-dataset/spacenet/SN8_floods/` — anonymous/unsigned access, no credentials
required (License **CC-BY-SA-4.0**). Two AOIs carry public labels:

- `Germany_Training_Public` — 2021 Western-Europe floods (Ahr/Erft valleys, Rhineland).
- `Louisiana-East_Training_Public` — Hurricane Ida (SE Louisiana, Aug 2021).

`Louisiana-West_Test_Public` is imagery-only (no `annotations/`) and is **excluded**.

Labels are per-tile GeoJSONs (WGS84 / CRS84) of **building footprints** (Polygon,
`building=yes`) and **road centerlines** (LineString, `highway=<type>`), each with a
`flooded` attribute (`"yes"` = post-event inundated; null/`"no"` = not flooded). The paired
imagery is Maxar VHR (~0.3–0.8 m). **Only the small GeoJSON labels are downloaded**
(801 files, ~a few MB total); no VHR imagery TIFFs are pulled — pretraining supplies its
own S2/S1/Landsat imagery.

## Class scheme

Unified building×road × flooded/non-flooded (spec §5 multi-target → one class map):

| id | name | source |
|----|------|--------|
| 0 | non_flooded_building | `building=yes`, not flooded |
| 1 | flooded_building | `building=yes`, `flooded=yes` |
| 2 | non_flooded_road | `highway=*`, not flooded |
| 3 | flooded_road | `highway=*`, `flooded=yes` |

nodata/ignore = 255. **Positive-only** dataset (spec §5): non-structure ground stays
nodata (we do not fabricate a background class); assembly supplies negatives from other
datasets.

Tiles-per-class (a tile counts toward every class it contains):
`non_flooded_building=525, flooded_building=137, non_flooded_road=744, flooded_road=212`.
Tiles-per-AOI: `Germany=202, Louisiana-East=654`. `flooded_building` is the sparsest class
but retained per spec §5 (downstream assembly drops classes that end up too small).

## VHR → 10 m handling (spec §4)

Each source label tile is ~350–650 m across (≈ one 64×64 tile at 10 m). Features are
reprojected from WGS84 into a local-UTM 10 m grid (UTM zone from the tile centroid) and
rasterized with `all_touched=True`, so every touched 10 m pixel is marked even though an
isolated building (~10–20 m) or road centerline (~5–10 m wide) is at/under one pixel. The
few tiles wider than 64 px are cut into a ≤64×64 grid (856 patches from 801 tiles). Paint
order burns flooded classes **last** so the flood signal wins overlaps
(non_flooded_road < non_flooded_building < flooded_road < flooded_building).

Individual buildings and narrow roads are genuinely under-resolved at 10 m, but **flooded
structures cluster**, so the flooded classes aggregate into contiguous flood-extent patches
— the salvageable signal for 10 m S2/S1. This is documented rather than a reason to reject
(the manifest explicitly asks to "co-locate with S2/S1 at event coords/dates").

## Change label / time range (spec §5 change-timing rule)

Flooding is a **transient/event** state (water recedes within days–weeks), so a
persistent-state recast is NOT valid — the dated-event approach is used instead. Both
events are resolvable to well within the ~1–2-month requirement:

- Germany → `change_time = 2021-07-15` (Ahr/Erft flood peak, mid-July 2021).
- Louisiana-East → `change_time = 2021-08-30` (Hurricane Ida landfall 2021-08-29/30).

`time_range` is a **360-day window centered on** `change_time` (e.g. Germany
2021-01-16 → 2022-01-11), so the sampled pretraining input window spans the flood; the
label is the where-mask of flooded/non-flooded structures. All labels are post-2016.

## Verification (spec §9)

- Opened multiple output tifs: single band, `uint8`, 64×64, UTM at 10 m (EPSG:32632 for
  Germany, EPSG:32615 for Louisiana), values ⊆ {0,1,2,3,255}. Global scan of all 856 tifs:
  union of values = `[0,1,2,3,255]`, zero files with unexpected values.
- Every `.tif` has a matching `.json` (856/856) with a 360-day `time_range` and
  `change_time` set; `metadata.json` covers all 4 class ids.
- **Georeferencing sanity**: tile centers reproject to ~6.9°E/50.5°N (Rhineland/Ahr flood
  zone) and ~90.0°W/29.8°N (SE Louisiana/Ida zone) — exactly the two flood events. Labels
  come from authoritative WGS84 source geometries, so placement on the S2 grid is exact.
- Re-running is idempotent (existing `{id}.tif` skipped).

## Caveats

- Individual buildings / narrow roads are under-resolved at 10 m; the reliable signal is
  flood **extent** where flooded structures cluster. `flooded_building` (137 tiles) and
  `flooded_road` (212 tiles) are the flood-signal classes; `non_flooded_*` mark dry
  structures in the same scenes.
- Positive-only (no background class); negatives come from assembly.

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.spacenet_8_flooded_roads_buildings
```
Outputs: `datasets/spacenet_8_flooded_roads_buildings/{metadata.json, locations/{id}.tif,.json}`
on weka; raw labels under `raw/spacenet_8_flooded_roads_buildings/{AOI}/annotations/`.
