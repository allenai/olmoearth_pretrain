# DARTS (Retrogressive Thaw Slumps)

- **Slug:** `darts_retrogressive_thaw_slumps`
- **Status:** completed
- **Task type:** classification (positive-only polygon segmentation)
- **Samples:** 25,000 label tiles (64×64, single class)
- **Label type:** polygons

## Source

DARTS: *Multi-year database of AI detected retrogressive thaw slumps (RTS) in hotspots of
the circum-arctic permafrost region — v1.2*. Nitze, I., Heidler, K., Nesterova, N.,
Küpper, J., Schütt, E., Hölzer, T., Barth, S., Lara, M. J., Liljedahl, A. & Grosse, G.
(2025), NSF **Arctic Data Center**, DOI [10.18739/A22B8VD7C](https://doi.org/10.18739/A22B8VD7C).
Companion data paper: *Scientific Data* **12**, 1512 (2025),
[10.1038/s41597-025-05810-2](https://doi.org/10.1038/s41597-025-05810-2). License **CC-BY-4.0**.

DARTS maps footprints of active retrogressive thaw slumps (RTS) — hillslope thermokarst
mass-wasting landforms triggered by thawing ice-rich permafrost — and, lumped into the same
detection target, active-layer detachment slides (ALD). Footprints are produced by a U-Net++
deep-learning model segmenting ~3 m PlanetScope imagery (with ArcticDEM slope / relative
elevation and Landsat trend layers), followed by review/validation, across circum-arctic
hotspots (NW Canada — Banks Island, Peel Plateau — Siberia, Novaya Zemlya, …).

## Access method

Fully open, no credentials. Files enumerated via the DataONE Solr API on `arcticdata.io`
and pulled with `download.download_http`. We use only the **Level 2** feature file (annual
maximum RTS extent per calendar year; L1 per-image footprints dissolved on the `year`
attribute):

- `DARTS_NitzeEtAl_v1-2_features_2018-2023_level2.gpkg` (152 MB, 77,405 polygons, EPSG:4326)
  — `https://arcticdata.io/metacat/d1/mn/v2/object/urn%3Auuid%3Af1169dfd-0b3e-405f-9c56-4ff3c4827316`

Level 1 (per-image, 125,250 features), coverage files, and the styling/description sidecars
were not used. Raw is under `raw/darts_retrogressive_thaw_slumps/` (+ `SOURCE.txt`).

## Class / label mapping

One unified foreground class (positive-only):

| id | name | notes |
|----|------|-------|
| 0 | retrogressive thaw slump / active-layer detachment slide | any active RTS footprint or active area within a larger RTS landform |
| 255 | *nodata / ignore* | everything outside a slump footprint |

**Why one class, not the manifest's two.** The DARTS v1.2 release carries **no per-feature
RTS-vs-ALD attribute** — the model detects RTS and ALD as a single "active slumping" target
class. So the two manifest classes (thaw slump / active-layer detachment slide) collapse to
one expressible class. Per spec §5 this is a positive-only / no-background dataset: non-slump
pixels are left as nodata (255); we do **not** fabricate synthetic negatives (assembly draws
negatives from other datasets).

## Time-range & change handling

DARTS L2 is **annually resolved** (each feature is the max extent for one calendar `year`,
2018–2023). Per the spec §5 change-timing rule, this is **not** treated as a change label:
annual resolution is coarser than the ~1–2 month change-timing bar, and a thaw-slump scar is
a *persistent* geomorphic feature that stays visible long after any one year's headwall
retreat. It is therefore handled as **presence/state classification**:

- `change_time = null`
- `time_range` = static **1-year window** anchored on the feature's observation year
  (`[Jan 1 year, Jan 1 year+1)`).

A slump observed in multiple years yields **one separate annually-resolved sample per year**
— that is the intended annual resolution. All labels fall 2018–2023 (fully in the Sentinel
era; no pre-2016 filtering needed).

## Processing

- One 64×64 tile per selected feature, in local **UTM at 10 m/pixel**, centered on the
  polygon (centroid, or an interior representative point for concave shapes).
- All **same-year** L2 polygons intersecting the tile bbox are rasterized to class 0
  (`all_touched=True` so tiny slumps survive — median footprint ~2,188 m² ≈ 22 px; 99th pct
  ~90,000 m²; max ~637,000 m²). Same-year-only keeps the mask temporally consistent with the
  tile's 1-year window. Rest of tile = 255.
- Geometries read on demand per tile with a pyogrio bbox filter (GPKG R-tree index), so both
  scan and write phases parallelize over a `multiprocessing.Pool(64)` + `star_imap_unordered`.

## Sampling

77,405 L2 features exceed the 25,000 per-dataset hard cap (§5), so we take a **geographically
stratified round-robin over 1-degree lon/lat cells** (seed 1) to keep dense hotspots from
dominating; one tile per selected feature → 25,000 tiles.

Selected-sample year distribution: 2018 = 383, 2019 = 395, 2020 = 456, 2021 = 7,503,
2022 = 7,232, 2023 = 9,031 (2021–2023 dominate, matching the annual-coverage core period).
24,800 / 25,000 tiles contain the slump mask (200 were pre-written in a smoke test);
foreground coverage per tile: min 0.2%, mean 2.8%, max 26% of pixels.

## Verification (§9)

- 25,000 `.tif` each with a matching `.json`; 0 orphans.
- Random tiles: single band, `uint8`, UTM CRS (EPSG:326xx/327xx), 10 m, 64×64, nodata 255,
  pixel values ∈ {0, 255} (valid class id + nodata).
- All sample JSONs: `time_range` ≤ 1 year, `change_time = null`.
- `metadata.json` declares the single class 0; all tif values are covered.
- Spatial sanity: labels are burned from the authoritative georeferenced GPKG via the shared
  WGS84→UTM rasterizer (same validated path as the mining/aquaculture polygon datasets), and
  99.2% of tiles land foreground on the intended slump — confirming alignment. A live
  Sentinel-2 overlay was not run (arctic hotspots, small features); rasterization is exact
  from source coordinates.

## Caveats

- Only one foreground class (RTS/ALD unified) — no RTS-vs-ALD split available in v1.2.
- Labels are model-derived (U-Net++ on PlanetScope) with review, not purely manual in-situ;
  a per-feature probability (`pval_*`) exists in the source but was not used to filter — all
  L2 features (model threshold 0.5) are eligible.
- Slumps are small; many tiles have only a few percent foreground (expected for a sparse,
  positive-only phenomenon; downstream assembly supplies negatives).

## Reproduce

```bash
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.darts_retrogressive_thaw_slumps --workers 64
```

Idempotent: existing `locations/{id}.tif` are skipped; the raw GPKG is downloaded once.
