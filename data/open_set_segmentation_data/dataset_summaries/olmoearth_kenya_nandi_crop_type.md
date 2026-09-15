# OlmoEarth Kenya Nandi crop type

- **Slug**: `olmoearth_kenya_nandi_crop_type`
- **Status**: completed
- **Task type**: classification (sparse points, spec §2a)
- **Num samples**: 8,568
- **Region**: Nandi County, Kenya
- **License**: internal (olmoearth)

## Source

Local rslearn eval dataset at
`/weka/dfive-default/rslearn-eai/datasets/crop/kenya_nandi/20250625`
(`have_locally: true`; wrote `raw/{slug}/SOURCE.txt`, no copy). It is the existing
olmoearth Kenya-Nandi crop-type eval (nandi_base/mm/aef variants). Two window groups:

- `groundtruth_polygon_split_window_32` (6,924 windows): manual field-survey crop-type
  reference. Field polygons were sampled on a ~10 m grid; one 32×32 window is built
  centered on each reference point. Categories: Coffee, Trees, Grassland, Maize,
  Sugarcane, Tea, **Legumes, Vegetables** (the manifest listed only the first six).
- `worldcover_window_32` (2,000 windows): homogeneous ESA-WorldCover-derived context
  points — **Water** (1,000) and **Built-up** (1,000).

Each window's `metadata.json` carries the `category` and a UTM projection + bounds. The
label is a single center pixel (verified (16,16) in 400/400 windows with a materialized
`label_raster`). 722 windows have `metadata.json` (valid category + bounds) but no
materialized layers; they are still usable as labeled points and were included.

## Access / processing

Point-only dataset → one dataset-wide `points.geojson` (spec §2a); no per-sample GeoTIFFs.
Run:
```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.olmoearth_kenya_nandi_crop_type
```
Idempotent; parallel metadata scan with `multiprocessing.Pool(64)`.

### Coordinates
The window-**name** lon/lat string is **unreliable** (up to ~175° divergence from the true
location on some windows) — it is NOT used. Coordinates are derived from each window's own
UTM CRS + bounds center pixel:
`ux=(b0+16.5)*10`, `uy=-(b1+16.5)*10`, then reproject to WGS84. Windows span **two UTM
zones** — EPSG:32636 (36N, 8,359 windows) and EPSG:32736 (36S, 565 windows) — so each
window is reprojected with its own CRS. This formula was verified to match rasterio's own
pixel-center transform exactly for both zones. Result: lon 34.74–35.42, lat −0.058–0.552
(Nandi County). Because these are the same georeferenced windows the eval used to extract
its Sentinel-2/1 imagery, spatial alignment with pretraining imagery is exact by
construction.

### Class scheme (unified, ids 0–9)
Manifest crop types first (0–5), then extra source crops (6–7), then WorldCover land-cover
context (8–9):

| id | class | count |
|----|-------|-------|
| 0 | Coffee | 977 |
| 1 | Trees | 926 |
| 2 | Grassland | 1000 |
| 3 | Maize | 1000 |
| 4 | Sugarcane | 964 |
| 5 | Tea | 979 |
| 6 | Legumes | 440 |
| 7 | Vegetables | 282 |
| 8 | Water | 1000 |
| 9 | Built-up | 1000 |

Balanced to ≤1000/class via `balance_by_class` (Grassland, Maize, Water, Built-up capped
at 1000; the rest kept in full). Total 8,568, well under the 25k cap. No classes dropped
(10 ≪ 254-class cap). Legumes/Vegetables are the sparser crop classes (downstream assembly
may filter very small classes).

### Time range
All reference points were observed in the **2023** growing season (window metadata
`time_range` = 2023-03; planting dates cluster in 2023 with older years for perennials
like Coffee/Tea/Trees, which are still present in 2023). Assigned a static 1-year window
`[2023-01-01, 2024-01-01)` per point (seasonal-crop rule, §5). This is post-2016 (Sentinel
era). Note: the manifest listed `time_range: [2024, 2025]`, but the on-disk data is 2023 —
2023 is used. No change labels.

## Caveats

- Water/Built-up are derived-product (ESA WorldCover) rather than in-situ, but restricted
  to homogeneous single-pixel samples (§5 map fallback); crop types are manual field
  survey.
- Manifest listed 6 crop classes; source actually has 8 crop categories + 2 WorldCover
  land-cover classes. All combined into one unified scheme (§5).
- Single-pixel (1×1) point labels; the labeled pixel is the window center.

## Verification

- `points.geojson`: FeatureCollection, 8,568 Point features, `task_type=classification`,
  `count=8568`; labels 0–9 match `metadata.json` class ids/counts.
- Coordinates within Nandi County bbox; uniform 2023 one-year `time_range`; no per-sample
  tifs (correct for point-only).
- Coordinate formula validated against rasterio's authoritative pixel-center transform for
  both UTM zones (exact match).
