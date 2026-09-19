# fMoW-Sentinel (Functional Map of the World)

- **Slug:** `fmow_sentinel_functional_map_of_the_world`
- **Status:** completed
- **Task type:** classification (sparse points, spec §2a)
- **Samples:** 23,479 points across 62 classes
- **Source:** Stanford / IARPA — fMoW-Sentinel (Cong et al. 2022, *SatMAE*), Stanford Digital
  Repository <https://purl.stanford.edu/vg497cb6002> (DOI 10.25740/vg497cb6002).
- **License:** fMoW Challenge Public License applies to the locations/categories in the
  metadata CSVs (the Sentinel-2 imagery, which we do **not** use, is under the Sentinel-2
  license).

## What the source is

fMoW-Sentinel pairs the Functional Map of the World facility locations with Sentinel-2
image time series. The metadata (which is all we need — pretraining supplies imagery) is
distributed as three small CSVs: `train.csv` (712,874 rows), `val.csv` (84,939), and
`test_gt.csv` (84,966), plus a `README.md`. Each row is one Sentinel-2 composite image for
a facility location with columns:

- `category` — one of **62** fMoW functional/land-use classes.
- `location_id` — fMoW location index (unique within a category **and split**).
- `image_id` — image index within the location; `<100` = a real fMoW acquisition,
  `>=100` = a synthetic composite placed at a 6-month-interval midpoint.
- `timestamp` — UTC center of the 90-day composite interval (`YYYY-MM-DDThh:mm:ssZ`,
  some real acquisitions carry fractional seconds).
- `polygon` — WGS84 lat/long **bbox** of the location (axis-aligned rectangle).

The 77 GB `fmow-sentinel.tar.gz` image tarball is **not downloaded**; only the ~220 MB of
CSVs are pulled (from `https://stacks.stanford.edu/file/druid:vg497cb6002/<file>`).

## Access method

Public, no credential. `download.download_http` from the Stanford stacks file endpoint.
Reproduce end-to-end with:

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.fmow_sentinel_functional_map_of_the_world
```

(idempotent; re-runs skip regeneration. `--force` regenerates.)

## Triage decision (accept)

- **Georeferencing recoverable:** yes — the `polygon` column gives the WGS84 bbox of every
  location, so labels place cleanly on the S2 grid. (This is the common fast-reject failure
  mode for "ML-ready" releases; fMoW-Sentinel passes.)
- **Observable at 10–30 m:** yes — these are functional facilities/land uses (golf course,
  stadium, port, airport, surface mine, solar/wind farm, …) largely discernible at 10 m.
- **Post-2016:** yes — timestamps span 2015–2020; the pre-2016 rows are filtered out and
  only 11 of 82,012 locations (which have no post-2016 image) are dropped.

## Encoding decision (why points, not tiles)

A fMoW category labels a **facility at a location**, not a homogeneous land-cover patch.
The facility bbox is small (max-side: median ~0.40 km, p90 ~0.50 km, max ~5 km; ~93% fit
inside a 640 m / 64-px tile), and the pixels *surrounding* a facility are generally **not**
the same class, so painting a uniform-class tile over the bbox would overclaim. Per spec
§4 (scene-level) + §2a, we therefore emit **one 1×1 sparse point** at the facility bbox
**centroid** with the category class, written to a single dataset-wide
`points.geojson` — not tens of thousands of tiny per-facility GeoTIFFs (which spec §2
warns cripple weka). This mirrors the land-use reference-point pattern (LUCAS / LCMAP).

## Processing details

- **Deduplication:** the CSVs are a per-location image *time series* (882,779 rows over
  82,012 unique `(split, category, location_id)` locations). The facility land use is
  static, so each location collapses to **one** point.
- **Time range (spec §5):** for each location, pick a representative **post-2016** image —
  preferring a real fMoW acquisition (`image_id<100`), else the earliest synthetic
  composite center — and set a **1-year window centered on that timestamp** (±180 days =
  360-day span, within the pretraining cap). Faithful to the image-acquisition-based
  ~1-year window; `change_time` is null (static land use).
- **Class scheme:** the 62 categories map to ids **0–61 by descending unique-location
  frequency** (well under the 254-class uint8 cap; no classes dropped). `source_id` records
  `split/category/location_id/img<image_id>` for provenance.
- **Balancing (spec §5):** `balance_by_class(per_class=1000, total_cap=25000)` →
  effective 403/class. 53 classes reach the 403 cap; the rest keep all available:
  `border_checkpoint` 373, `zoo` 340, `lake_or_pond` 261, `impoverished_settlement` 237,
  `debris_or_rubble` 234, `shipyard` 151, `nuclear_powerplant` 70, `space_facility` 51.
  Sparse classes are retained (downstream assembly filters too-small ones).

## Verification (spec §9)

- `points.geojson`: 23,479 features, `task_type=classification`, labels 0–61 (all 62
  classes present), every label id covered by `metadata.json` classes.
- Every feature has a `time_range` of exactly 360 days (≤ 1 year), `change_time=null`.
- Coordinates span the globe (lon −179.7…177.6, lat −55.1…74.2), consistent with fMoW's
  global coverage.
- **Spatial consistency:** for sampled points the emitted centroid falls inside the source
  `polygon` bbox (5/5 checked), and spot-checked coordinates land on plausible facility
  sites (e.g. `airport` points over Kirkuk IQ, Surabaya ID, Pretoria ZA airfields).
  Because points are exact WGS84 bbox centroids, S2 georeferencing alignment is exact by
  construction; no per-point S2 tile fetch was performed (point encoding, not a raster).

## Caveats

- Point (facility-centroid) labels only; the facility footprint/extent is not encoded.
- fMoW category noise (manual/crowdsourced annotation) carries through as label noise.
- A minority of large facilities (bbox > 640 m; ~7%) are still represented by a single
  centroid point.
