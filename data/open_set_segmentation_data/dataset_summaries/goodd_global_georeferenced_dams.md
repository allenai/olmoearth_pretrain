# GOODD (Global Georeferenced Dams)

- **Slug:** `goodd_global_georeferenced_dams`
- **Task type:** classification (single-class, positive-only object detection)
- **Status:** completed — 1500 samples (1000 dam-positive tiles + 500 background-only negatives)
- **License:** CC0

## Source

Mulligan, M., van Soesbergen, A. & Sáenz, L. *GOODD, a global dataset of more than
38,000 georeferenced dams.* Scientific Data 7, 31 (2020).
https://doi.org/10.1038/s41597-020-0362-5 — distributed by Global Dam Watch,
https://www.globaldamwatch.org/goodd.

Downloaded `GOODD_data.zip` (~18.9 MB) to
`raw/goodd_global_georeferenced_dams/`. It contains two ESRI shapefiles:

- `Data/GOOD2_dams.shp` — **38,667 dam-wall POINTS** (EPSG:4326), attributes
  `DAM_ID`, `Count_ID`, `Latitud`, `Longitud`. Digitized by manual photointerpretation
  of Landsat/SPOT imagery.
- `Data/GOOD2_catchments.shp` — one upstream drainage-**catchment POLYGON** per dam.

## Decisions / class mapping

- Built a **single-class positive-only object-detection** dataset of dam walls
  (`label_type` "points that mark presence", spec §4). Class scheme:
  `0 = background`, `1 = dam`; `255 = nodata/ignore` (detection buffer rings).
- **Catchment polygons dropped.** They delineate the full upstream hydrological
  drainage basin of each dam (frequently thousands of km²), which is not a feature
  observable or segmentable *at the dam location* from S2/S1/Landsat at 10–30 m, and is
  not a coherent per-pixel land-cover class. Only the dam points are used.
- The source records no dam-type attribute, so a single foreground class is correct.

## Detection encoding (tunable, spec §4)

- 1 px positive at each dam wall, ringed by a **10 px nodata (255) buffer** to absorb the
  coordinate imprecision of manual Landsat/SPOT digitizing, with background (0) filling
  the rest of a **32×32 (320 m) context tile**. Parameters: `tile_size=32`,
  `positive_size=1`, `buffer_size=10`.
- **All GOODD dams falling inside a tile are marked positive** (KD-tree neighbor lookup),
  so clustered dams on the same river reach are labeled correctly. Observed: 1001 dam
  pixels across the 1000 positive tiles (one tile caught a 2nd dam).
- **500 background-only negative tiles** emitted per spec §4, placed 3–20 km from a dam
  and guaranteed ≥1 km from any dam (KD-tree check) so they are spatially-meaningful
  negatives.

## Sampling

- 1000 of 38,667 dams sampled (seeded shuffle) as positive tile centers — spec §5
  per-class cap of 1000. Well under the 25k per-dataset cap. Note: this leaves ~37.7k
  dams unused; re-running with a higher `PER_CLASS` could scale up if more dam signal is
  wanted later.
- **Time range:** dams are persistent, undated structures → per spec §5 (static labels)
  each sample gets a 1-year window at a representative Sentinel-era year, spread
  pseudo-randomly across **2016–2022** for temporal diversity.

## Outputs

- `datasets/goodd_global_georeferenced_dams/metadata.json`
- `datasets/goodd_global_georeferenced_dams/locations/{000000..001499}.tif` (+ `.json`)
- Each `.tif`: single-band uint8, local UTM at 10 m, 32×32, nodata 255.

## Verification

- 1500 `.tif` + 1500 `.json`. Global value histogram = {0, 1, 255} only. Positives have
  1 dam px + 440 buffer + 583 background; negatives are all-background (1024 px).
- CRS is per-sample UTM (e.g. EPSG:32721/32649/32631), res 10 m, size ≤64.
- Georeferencing check: tile 000000 center reprojects to (-56.839, -29.9266), matching
  source `DAM_ID 1003102` at (-56.8389, -29.9266) — exact to sub-pixel.
- Every `.json` has a 1-year `time_range`; `metadata.json` classes cover all tif values.

## Caveats

- Dam walls of small dams may be sub-10 m; the point + 10 px ignore ring makes this a
  weak presence-detection target rather than a precise footprint segmentation.
- Positive-only: the dataset supplies its own background/negative tiles (detection
  exception in spec §5); no per-pixel foreground beyond the dam marker.

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.goodd_global_georeferenced_dams
```
Idempotent (skips already-written tiles). Raw zip already staged at
`raw/goodd_global_georeferenced_dams/GOODD_data.zip` (extract `Data/` before running).
