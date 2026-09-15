# AI4SmallFarms

- **Slug:** `ai4smallfarms`
- **Status:** completed
- **Task type:** classification (2-class field-vs-background segmentation)
- **Num samples:** 1000 label patches (64x64, 10 m, UTM)

## Source

Persello, C., Grift, J., Fan, X., Paris, C., Hansch, R., Koeva, M., & Nelson, A. (2023).
"AI4SmallFarms: A Dataset for Crop Field Delineation in Southeast Asian Smallholder Farms",
IEEE GRSL 20, 2505705. https://doi.org/10.1109/LGRS.2023.3323095

Distributed as a **fiboa GeoParquet** on Source Cooperative
(https://source.coop/fiboa/ai4sf, **CC-BY-4.0**), converted with fiboa-cli by Matthias Mohr.

### Access
Public HTTP, **no credential required** (S3-compatible data proxy at
`https://data.source.coop/fiboa/ai4sf/`). Downloaded only the single 29 MB GeoParquet
`ai4sf.parquet` to `raw/ai4smallfarms/`. Not downloaded: `ai4sf.pmtiles` (24 MB web map
tiles) — pretraining supplies its own imagery.

### Contents
439,001 manually-digitized smallholder crop-field **polygons** across 62 tiles of ~5x5 km
in **Cambodia** (318,088) and **Vietnam** (120,913). Fields are small: median ~1,702 m^2
(~4 px across at 10 m), p10 ~412 m^2 (~2 px), p90 ~6,782 m^2. Geometry stored in EPSG:32648
(UTM 48N) metres. Every polygon: `determination_datetime = 2021-08-01`,
`determination_method = auto-imagery` (digitized from 2021 Sentinel-2 composites),
`group` id 0–61 (the 5x5 km tile), `country`. All polygons (438,989 Polygon, 12
MultiPolygon).

## Label encoding decision

Encoded as a **2-class field-vs-background mask** (task spec §4 polygons; the "field
boundary" / "non-field" manifest classes recast as field extent vs background):

| id | name       | meaning |
|----|------------|---------|
| 0  | non-field  | background: land not delineated as a smallholder crop field |
| 1  | field      | interior/extent of a digitized crop-field polygon |

`nodata_value = 255` (unused here; every pixel is 0 or 1).

**Why not a boundary-line class?** Fields are small (median ~4 px, p10 ~2 px at 10 m) and
the physical field boundaries (bunds/dikes/paths, ~1–5 m wide) are **sub-pixel** at
Sentinel-2's 10 m GSD — not separable as their own spectral class. A dilated 1-px boundary
line would consume most of these small fields, leaving no interior. The field **extent** is
observable at 10 m, so the delineation signal is encoded as field vs background. Because
AI4SmallFarms exhaustively digitized every field inside each 5x5 km tile, the non-field (0)
pixels are a **genuine negative** (not undeclared/missing fields), unlike AI4Boundaries.

## Processing

Per group (5x5 km tile): pick the local UTM (48N for centroid lon<108, else 49N — 4 groups
are in Vietnam east of 108°E → 49N), reproject polygons into that UTM's 10 m pixel grid, and
tile the group extent into **non-overlapping <=64x64** windows (spec cap 64). Rasterize field
polygons (value 1, fill 0) via `rasterize.rasterize_shapes` with an STRtree per-window index.
Keep windows with >=1 field pixel; drop <32 px edge slivers. Reprojection/rasterization use
the shared `rasterize`/`io` utils; scan and write both use `multiprocessing.Pool(64)`.

- **Candidates:** 4,538 field-containing windows across the 62 groups.
- **Selection:** tiles-per-class balanced (`sampling.select_tiles_per_class`, per_class=1000,
  total_cap=25,000). Every candidate window contains both classes, so selection yields
  **1000** windows (both classes at 1000).
- **Country split of selected tiles:** cambodia 519, vietnam 481.
- **Field pixel fraction (selected):** 0.71 (dense farmland tiles).

### Time range
1-year **2021** window `[2021-01-01, 2022-01-01)` for every sample — labels anchored on 2021
Sentinel-2 composites (`determination_datetime = 2021-08-01`; post-2016 Sentinel era). Static
field extent, **not** a change dataset (`change_time = null`).

## Verification (§9)

- 1000 `.tif` + 1000 `.json`. All tifs: single band, uint8, EPSG:326xx UTM, 64x64, 10 m res,
  pixel values ∈ {0,1} (matches the class map). All jsons: ≤1-year time_range, change_time null.
- Georeferencing sanity: sampled tile centers fall within the source bbox
  (lon 102.8–109.3, lat 9.6–21.4), spanning both Vietnam (~lat 20.8) and Cambodia (~lat 11.7).
- Idempotent: re-running skips existing `{sample_id}.tif` and reuses `scan_cache.pkl`.
- Full Sentinel-2 image overlay not performed; alignment is inherent since the polygons were
  themselves digitized on Sentinel-2 imagery, and centers verified against the source bbox.

## Caveats
- 2-class dataset; downstream assembly supplies negatives from other datasets as needed
  (though non-field here is already a real negative). No rare-class concerns.
- Selection capped at 1000/class per spec — the source has 439k polygons; only a
  representative 1000 tiles are emitted.

## Reproduce
```
# download (once): curl -L https://data.source.coop/fiboa/ai4sf/ai4sf.parquet \
#   -o /weka/.../raw/ai4smallfarms/ai4sf.parquet
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.ai4smallfarms
```
