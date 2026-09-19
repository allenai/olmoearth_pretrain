# Thermokarst Lakes & Ponds, Qinghai-Tibet Plateau

- **Slug**: `thermokarst_lakes_ponds_qinghai_tibet_plateau`
- **Status**: completed
- **Task type**: classification (positive-only, 2 classes)
- **Family / region**: permafrost / Qinghai-Tibet Plateau
- **Num samples**: 1628 label tiles (64x64, UTM 10 m)
- **License**: CC-BY-4.0

## Source

Zenodo record [5509325](https://zenodo.org/records/5509325) — "Thermokarst lake and
pond dataset of the Qinghai-Tibet Plateau (QTP)". Wei, Z., Du, Z., Wang, L., Lin, J.,
Feng, Y., Xu, Q., & Xiao, C. (2021), "Sentinel-based inventory of thermokarst lakes and
ponds across permafrost landscapes on the Qinghai-Tibet Plateau", *Earth and Space
Science*, 8(11), e2021EA001950, https://doi.org/10.1029/2021EA001950.

Five ESRI polygon shapefiles (`QTP_Perm_TL_2020_1..5.shp`), one per sub-region, each in
its own UTM projection (EPSG:32644 / 32645 / 32647 and an equivalent custom UTM-44N WKT).
Together they cover the entire QTP permafrost landscape with **161,341** thermokarst
water-body polygons, mapped from **2020** Sentinel-2 imagery by a random-forest model plus
manual visual vectorization, ranging from ~467 m² to 3.09 × 10⁶ m². The attribute table
carries `Area` (m²), DMS `Long`/`Lati` strings, `Perm_Type`, `Elevation`, `Basin`, and
climate/soil covariates. **There is no lake/pond class field** — the split is by size.

## Access

Downloaded the five zips via `download.download_zenodo("5509325", ...)` (~300 MB total,
no credentials) into `raw/{slug}/`, extracted into `raw/{slug}/extracted/`. Each `.shp`
ships with an `.sbn` spatial index, so bounded per-tile bbox reads are fast.

## Class mapping (size split)

The source paper distinguishes ponds as standing water **< 10,000 m² (0.01 km²)**; larger
bodies are lakes. That threshold is applied here:

| id | name             | rule            | source polygons |
|----|------------------|-----------------|-----------------|
| 0  | thermokarst lake | Area ≥ 10,000 m² | 33,933 (21%)   |
| 1  | pond             | Area < 10,000 m² | 127,408 (79%)  |
| 255 | nodata/ignore   | non-water pixels | —              |

**Positive-only foreground** (spec §5): the product maps only water bodies, so non-water
is left as nodata (255), not a fabricated background class. The assembly step supplies
negatives from other datasets.

## Processing

- Polygon centroids snapped to a 640 m grid (= a 64 px × 10 m output tile) in each file's
  UTM CRS; occupied cells deduped (161,341 polygons → 86,584 candidate cells). This
  collapses the extremely dense pond clustering into distinct tile footprints.
- Each cell tagged with the classes of the centroids it contains; **tiles-per-class
  balanced** selection (`select_tiles_per_class`, rarest class = lakes filled first),
  `per_class=1000` → 1629 cells selected.
- Each selected cell → one 64×64 tile in local UTM at 10 m, centered on the cell. Every
  water polygon intersecting the tile (bbox read from the source `.shp`) is rasterized
  with its area-derived class over a 255 background. `all_touched=True` so the smallest
  ponds (~500 m² ≈ 5 px) stay visible. 1628 tiles written (1 selected cell produced no
  resolvable water pixels and was skipped).
- A tile counts toward every class actually present after rasterization. Final:
  **1169 tiles contain lake pixels, 1045 contain pond pixels** (586 contain both). Both
  classes exceed the 1000 target because boundary tiles capture more classes than their
  centroid-based estimate; well under the 25k per-dataset cap.

## Time range

Annual 2020 product → each tile gets a 1-year window `[2020-01-01, 2021-01-01)`.
Persistent-landform presence classification; `change_time = null`.

## Resolvability

All polygons are ≥ ~467 m² (≥ ~5 px at 10 m), so **none are sub-pixel** at 10 m. The
smallest ponds (~500 m²) are near the resolution limit and are kept via
`all_touched=True`. Large lakes (up to 3 km²) exceed the 640 m tile and are clipped to the
window (homogeneous interior).

## Verification

- Structural: all 1628 `.tif` are single-band uint8, 64×64, UTM at 10 m, nodata=255;
  pixel values across the whole set are exactly {0, 1, 255}, matching `metadata.json`.
  Every `.tif` has a matching `.json` with a 1-year `time_range`.
- Georeferencing (independent cross-check): for sampled tiles the source polygons' own DMS
  `Long`/`Lati` attributes fall inside the tile footprint (e.g. tile 000001 center
  82.328°E/30.372°N; contained polygons' DMS ≈ 82.33°E/30.37°N), and class assignment
  matches area (tile 000800: a 170,854 m² polygon → lake=0, an 8,092 m² → pond=1). Tile
  centers lie on the QTP (82–97°E, 30–35°N).
- Spatial overlay: EOX s2cloudless-2020 imagery fetched for several lake tiles overlays
  cleanly on the labels — e.g. the lake label in tile 001200 sits exactly on the dark
  water body in the S2 image. Minor edge blur is from the low-res cloudless mosaic, not
  label misregistration (the DMS cross-check is exact).
- Idempotent: re-running skips existing `locations/{id}.tif`.

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.thermokarst_lakes_ponds_qinghai_tibet_plateau
```

## Caveats

- The lake/pond boundary is a single hard size threshold (10,000 m²) from the source
  paper; there is no morphological/genetic class in the data.
- The tile is centered on the 640 m grid cell, not on an individual polygon, so a tile may
  clip water bodies at its edges (standard for the bounded-polygon-sampling recipe).
- Non-water pixels are ignore (255), by design (positive-only); do not interpret them as a
  mapped "no water" class.
