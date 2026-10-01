# GLAKES — global lake-water segmentation

- **Slug:** `glakes`
- **Task:** classification (binary per-pixel segmentation) — `0 = background`, `1 = lake water`
- **Family:** water · **Region:** global · **License:** CC-BY-4.0
- **Status:** completed · **num_samples:** 25,000 (20,408 lake tiles, 4,592 background-only tiles)

## Source

GLAKES global lake polygon product, from Pi et al., "Mapping global lake dynamics reveals
the emerging roles of small lakes" (Nature Communications, 2022).
Zenodo record **7016548** (CC-BY-4.0), file `GLAKES.rar` (1.8 GB).

`GLAKES.rar` unpacks to 7 per-continent ESRI shapefiles (all EPSG:4326):
`GLAKES_{af, as, eu, na1, na2, oc, sa}.shp`, totaling **3,426,389 lake polygons**
(> 0.03 km²). Each polygon is a lake footprint at maximum water extent over 1984–2019,
with attributes `Lake_id`, `Area_bound` (km²), `Continent`, `Lat`/`Lon` (centroid), and
glacier/permafrost/endorheic/reservoir flags. Polygons were derived and validated with a
U-Net water-segmentation pipeline. Lakes are small-dominated: median `Area_bound`
≈ 0.085 km², ~87% fit within a single 64×64 tile (0.41 km²).

## Access method

Downloaded `GLAKES.rar` from the Zenodo files API (unauthenticated, public) to
`raw/glakes/GLAKES.rar`; extracted with `bsdtar` to `raw/glakes/extract/`. No credentials
required. Per-tile intersecting polygons read directly from the shapefiles via a
`pyogrio` bbox spatial filter (uses the `.sbn` index), so no global in-memory index is
built and the write phase parallelizes over `multiprocessing.Pool(64)`.

## Label construction

Binary segmentation into 64×64 uint8 tiles in local UTM at 10 m/pixel (nodata 255, unused):
- **0 background** — any surface outside a GLAKES lake polygon.
- **1 lake water** — inside a GLAKES lake polygon.

Because the source is a huge global vector, sampling is **bounded and geographically
stratified**: a round-robin over 1-degree lon/lat cells across all 7 continents, capped at
**25,000 tiles total** (spec hard cap). Two tile kinds:
- **Positive (~20k):** centered on a stratified sample of lake centroids. All GLAKES
  polygons whose envelope intersects the tile are clipped to the tile, reprojected to the
  tile's UTM grid, and rasterized to class 1 (`all_touched=True`, so small/thin lakes
  register); the remainder is background 0. Nearby lakes and shorelines are captured.
- **Negative (~5k):** background-only tiles, produced by offsetting a stratified sample of
  lake anchors by a random ~3–9 km vector so no lake falls in the tile. The ~8% of anchors
  whose offset still clipped a lake are rasterized correctly and counted as lake tiles
  (final: 4,592 pure-background, 20,408 lake tiles).

Verified water-coverage of lake tiles (400-tile sample): median 0.22, min 0.08, max 1.0,
only 4% fully-water — most tiles contain a genuine land/water boundary. Continent spread:
as 9,276 · na2 5,270 · af 2,837 · eu 2,741 · sa 2,501 · oc 1,417 · na1 958.

## Time range

Lakes are quasi-static; GLAKES covers 1984–2019 and the manifest window is 2016–2021.
Each tile gets a 1-year window with start year sampled uniformly in **2016–2020** (Sentinel
era). No `change_time` (not a change dataset).

## Metadata / classes

`metadata.json`: task_type `classification`; classes `[{0: background}, {1: lake water}]`;
`nodata_value` 255. Per-sample JSON carries `crs`, `pixel_bounds`, `time_range`,
`classes_present`, and `source_id` (`{continent_file}:{row_index}`).

## Caveats

- Negatives are labeled all-background even though GLAKES maps **lakes only**; a negative
  tile could contain an unmapped small pond/river/stream, or (rarely, near coasts) ocean.
  Positives are reliable (rasterized directly from validated polygons).
- Lake polygons are the **maximum** extent over 1984–2019; a given year's actual water may
  be smaller (seasonal draw-down / long-term shrinkage). The label is "lake basin water
  extent", not a single-date shoreline.
- Very large lakes (> tile size) centered on their centroid yield all-water tiles (4% of
  lake tiles); the bulk are small lakes giving mixed land/water tiles.
- Spatial alignment is guaranteed by construction (polygons rasterized into each sample's
  exact CRS/bounds via the shared `rasterize` module). An automated Sentinel-2 overlay
  eyeball check was not run — the S2 rslearn data source needs an index cache/credentials
  not available in this environment.

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.glakes --workers 64
```
Idempotent: existing `locations/{id}.tif` are skipped. Outputs under
`/weka/dfive-default/helios/dataset_creation/open_set_segmentation/datasets/glakes/`.
