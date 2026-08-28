# ChinaPV

- **Slug:** `chinapv`
- **Status:** completed
- **Task type:** classification (per-pixel, rasterized polygons)
- **Num samples:** 2000 (1000 `pv_rural`, 1000 `pv_urban`)
- **Family / region:** solar / China
- **License:** CC-BY-4.0 (usable)

## Source

ChinaPV, "the spatial distribution of solar photovoltaic installation dataset across China
in 2015 and 2020" (Sci Data; Zenodo record **14292571**,
https://zenodo.org/records/14292571). PV installations across China were mapped from
**Landsat-8** imagery (30 m) with **manual adjustment/refinement** and vectorized as
polygons for two epochs (2015 and 2020). Downloaded via `download.download_zenodo` (the
`.shp/.shx/.dbf/.prj` parts for both epochs); no credentials needed.

Shapefiles are EPSG:4326 (3D Polygon). Per-polygon attributes: `Lat`, `Lon` (centroid),
`Area` (km²), `Perimeter`, `Province` (str), and `urban` (int: 0 = rural / ground-mounted,
1 = urban / distributed PV).

- `ChinaPV_2020_v1.1.shp` — 10,985 polygons, 2020 — **USED**
- `ChinaPV_2015_v1.1.shp` — 1,645 polygons, 2015 — **DROPPED** (entirely pre-2016; a PV
  panel visible only in the 2015 epoch cannot be placed confidently in the Sentinel era, spec §2).
- `PV_test_samples.shp` — author sample points, not needed.

## Class mapping

The source `urban` attribute is a genuine appearance/context split (rural utility-scale
ground-mount vs urban rooftop/distributed PV) and the manifest names the class
"PV installation (urban/rural)", so two foreground classes are kept:

| id | name          | meaning                                             |
|----|---------------|-----------------------------------------------------|
| 0  | background    | non-PV land within the tile (real surroundings)     |
| 1  | pv_rural      | source `urban == 0` — rural / ground-mounted PV     |
| 2  | pv_urban      | source `urban == 1` — urban / distributed PV        |

nodata = 255. Source distribution (2020): rural 7,283, urban 3,702.

## Encoding (polygons, spec §4)

Each polygon → ONE ≤64×64 UTM tile at 10 m/pixel (local UTM zone from centroid lon/lat).
Tile centered on the geometry's `representative_point` (guaranteed inside the polygon),
sized to the footprint + 8 px background margin, capped at 64×64. About 33% of polygons
span >640 m and are represented by a 64×64 crop around the representative point (local
footprint + boundary). `all_touched` rasterization so thin/small installations stay visible
at 10 m. Positive-only foreground classes — no fabricated negatives (spec §5); background
pixels are the genuine surroundings within the tile.

## Sampling & time range

- **Sampling:** up to 1000 tiles per foreground class via `balance_by_class` (key
  `fg_class`), giving 1000 `pv_rural` + 1000 `pv_urban` = 2000 tiles, well under the 25k cap.
- **Time range:** 1-year window anchored on **2020** (`[2020-01-01, 2021-01-01)`). PV
  installations are persistent, so a static-label year window is appropriate;
  `change_time = null`.

## Verification

- 2000 `.tif` + 2000 matching `.json`. Spot-checked tiles: single band, uint8, EPSG:326xx
  UTM at 10 m, size ≤64×64, nodata 255. Values across all tiles = {0, 1, 2} (rural tiles
  {0,1}, urban tiles {0,2}) — all valid class ids, covered by `metadata.json`.
- Every sample JSON has a 1-year `time_range` on 2020, `change_time = null`.
- Georeferencing: tile centers reproject back to within ~0.001° of the source polygon
  centroids (small offset = representative-point vs centroid centering) — placement correct.
- Idempotent: re-running skips existing `{id}.tif`.

## Caveats

- The urban/rural split is the source `urban` attribute; at 10 m the panels themselves may
  not always be visually separable by appearance alone (the distinction is partly contextual).
- The 2015 epoch is dropped entirely (pre-2016). Only 2020 labels are used.

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.chinapv
```
