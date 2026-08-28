# Circum-Antarctic Icebergs (Sentinel-1)

- **slug:** `circum_antarctic_icebergs_sentinel_1`
- **status:** completed
- **task_type:** classification (binary iceberg vs ocean/sea-ice segmentation)
- **num_samples:** 25,000 label tiles (64x64, uint8, local UTM/UPS @ 10 m)

## Source

"A Six-year circum-Antarctic icebergs dataset (2018-2023)" — Zenodo record
[17165466](https://doi.org/10.5281/zenodo.17165466) (ESSD, **CC-BY-4.0**). Icebergs were
detected from **Sentinel-1 SAR** by a semi-automated random-forest classifier with manual
correction. Region: Southern Ocean south of ~55S.

Downloaded file: `Iceberg vector outline.zip` (346 MB) → six GeoPackages
`{2018..2023}10_distribution.gpkg`, one per year, each the **October** distribution of
detected icebergs. All layers are in **EPSG:3031** (Antarctic Polar Stereographic).
Per-year feature counts: 2018=34,825; 2019=39,261; 2020=38,066; 2021=51,420; 2022=36,186;
2023=44,537 (**244,295 total**). Each feature is one iceberg outline `Polygon` with
attributes `lon, lat` (centroid), `area_km2`, `area_uncertainty_km2`, `perimeter_km`,
`long_axis_km`, `short_axis_km`, `mass_gt`, `mass_uncertainty_gt`.

(The record's other files — `Iceberg detection code.zip`, `Iceberg sample set.zip` — were
not needed and not used.)

## Label mapping

Binary per-pixel segmentation:

| id | name       | meaning |
|----|------------|---------|
| 0  | background | open ocean / sea ice — any surface outside a mapped iceberg outline |
| 1  | iceberg    | inside a mapped iceberg outline polygon |

nodata = 255 (unused; every tile is fully observed). The rich per-iceberg
geometric/mass attributes are **not** expressible as a per-pixel raster target, so they are
collapsed to a single `iceberg` class and preserved only in the metadata description. Task
type is therefore classification, not regression.

## Processing

Modeled on `glakes.py` (bounded, geographically-stratified polygon → tile rasterization).

- **Sampling:** the product is a large circum-Antarctic vector, so we do BOUNDED sampling
  capped at the 25,000-tile per-dataset limit. Centroids from all six years are pooled and
  selected by **round-robin over 1-degree lon/lat cells** (geographic stratification), which
  also naturally mixes years. Realized per-year counts: 2018=3,426; 2019=3,262; 2020=3,612;
  2021=5,669; 2022=3,578; 2023=5,453.
- **Tiles:** each tile is 64x64 @ 10 m in the sample's **local UTM/UPS** projection
  (`get_utm_ups_projection`), centered on a sampled iceberg centroid. All iceberg polygons
  intersecting the ~640 m tile are read via a per-tile pyogrio bbox spatial filter (in
  EPSG:3031), reprojected to UTM pixel space, and rasterized to class 1 (`all_touched=True`);
  the rest is background 0. Source polygons wrapped as `EPSG:3031_1_1` Projection (mirrors
  the repo's `WGS84_PROJECTION`).
- **Time range:** each GeoPackage is an October snapshot, so each tile gets the **1-month**
  window `[Oct 1, Nov 1)` of its source year. A full-year window would be ill-posed because
  icebergs drift; the monthly window is the tightest anchor the product supports.
- **Negatives:** per spec §5 (positive-only dataset), **no synthetic background-only tiles
  are fabricated**. The within-tile ocean around each berg is genuine, spatially-meaningful
  background; the assembly step adds further negatives from other datasets.

## Class balance / stats

- 25,000 tiles, **all** contain class 1 (iceberg). In a 300-tile sample: 293 mixed
  (background + iceberg), 7 all-iceberg; mean iceberg-pixel fraction ≈ 0.50 — well balanced
  between the two classes at the pixel level.

## Verification

- Sampled tifs: single-band, uint8, UTM CRS (e.g. EPSG:32717/32724/32728/32703), 64x64,
  10 m resolution, nodata 255, values ⊆ {0, 1}. Every `.tif` has a matching `.json` with a
  1-month `time_range` and `classes_present`.
- Georeferencing round-trip: tile-center lon/lat reprojected from `crs`/`pixel_bounds`
  matches the source berg centroid to 4 decimals (e.g. src 64.4232/-60.0346 vs tile
  64.4231/-60.0345). All 200 sampled tile centers lie south of 55S (Southern Ocean),
  matching the stated region.
- **No S2 overlay check performed:** icebergs are SAR-detected; Sentinel-2 coverage over the
  Southern Ocean in austral October is poor (low sun, cloud, sea ice), so an optical overlay
  would be misleading rather than confirmatory. Georeferencing was validated analytically
  (round-trip above) instead.

## Caveats / judgment calls

- **Drift within the month:** icebergs move ~km/day, so even the 1-month time window carries
  positional label noise. Pretraining should pair these labels preferentially with
  Sentinel-1 imagery near the October acquisition. Flagged here rather than narrowing
  further (no per-berg acquisition dates are available in the product).
- **Giant tabular bergs** (area up to ~5,700 km2, far larger than a 640 m tile) yield
  all-iceberg tiles with no background (~2% of tiles). These are valid but uninformative for
  boundary learning; they are kept for large-berg representation.
- **`background` = "no mapped iceberg":** sub-detection-threshold bergs (below the product's
  ~0.04 km2 minimum) may fall in background pixels.
- **Binary collapse:** the source's per-iceberg area/mass attributes are discarded at the
  pixel level (documented in class metadata) — they cannot be a per-pixel target.

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.circum_antarctic_icebergs_sentinel_1 --workers 64
```
Raw source: `raw/circum_antarctic_icebergs_sentinel_1/extract/Iceberg vector outline/*.gpkg`
(see `raw/.../SOURCE.txt`). Idempotent: existing `locations/{id}.tif` are skipped.
```
