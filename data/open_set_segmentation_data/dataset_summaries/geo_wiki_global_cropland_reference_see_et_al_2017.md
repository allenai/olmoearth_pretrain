# Geo-Wiki Global Cropland Reference (See et al. 2017)

- **Slug**: `geo_wiki_global_cropland_reference_see_et_al_2017`
- **Status**: completed
- **Task type**: classification (binary: cropland / non-cropland)
- **Label type**: points (sparse point segmentation -> `points.geojson`, spec 2a)
- **Num samples**: 2000 (1000 cropland + 1000 non-cropland, class-balanced)

## Source

PANGAEA dataset doi [10.1594/PANGAEA.873912](https://doi.pangaea.de/10.1594/PANGAEA.873912)
(See, Linda 2017; companion to the *Scientific Data* paper). A global crowdsourced cropland
reference database collected via the **Geo-Wiki** platform. Over ~3 weeks in **September
2016**, volunteers visually interpreted VHR (Google/Bing) imagery at ~36k systematically
sampled global locations (lat/lon grid intersections, densified where cropland probability
was 25-75%) and marked which grid cells within each frame were cropland.

**Access**: open direct HTTP download from `store.pangaea.de` — no credentials required.
Files are small zipped tab-delimited text. The dataset record also lists control (1793,
trained students) and expert (60) validation subsets; we use the full participant reference
set.

License: **CC-BY-3.0** (PANGAEA landing page; the manifest listed CC-BY-4.0 — noted
discrepancy, both permit use with attribution).

### Files (raw/)
- `crop_all.zip` — every grid cell marked cropland, per user, with per-submission timestamps
  and skip reasons (35866 locations; 1.08M rows).
- `loc_all.zip` / **`loc_all_2.zip`** — mean cropland per (location, user). `loc_all_2` is
  the **updated** version (skipped users' crop value blanked) and **adds a `timestamp`
  column**; this is the file we process.
- `crop_con` / `loc_con` (control, 1793), `crop_exp` / `loc_exp` (expert, 60), and
  `See_2017_info.pdf` (field dictionary) — downloaded for reference, not used for labels.

### `loc_all_2.txt` fields
`location_id, userid, timestamp, sumcrop, loc_cent_X, loc_cent_Y` where `sumcrop` is the
mean cropland **percentage (0-100)** the user assigned at that location and
`loc_cent_X/Y` are the frame-centroid lon/lat (decimal degrees, WGS84).

## Processing / label mapping

1. Parse `loc_all_2.txt` (203,515 rows). For each of the **35,866 unique locations**,
   average `sumcrop` across all users who did *not* skip it (blank `sumcrop` rows excluded)
   -> a mean cropland fraction (%). Every location retained a cropland judgement.
2. **Binary classification** (manifest classes are binary). Majority threshold:
   - mean cropland % **>= 50 -> `cropland` (id 0)**
   - else -> `non-cropland` (id 1)
   The raw continuous mean is preserved per point as an auxiliary `cropland_fraction`
   property (0-100), so downstream can re-threshold or treat it as a regression target.
3. Pre-balance distribution: **7825 cropland / 28041 non-cropland**. Balanced to
   <=1000/class (spec 5) via `sampling.balance_by_class` -> 1000 + 1000 = **2000** points
   (well under the 25k cap).
4. Written as one dataset-wide `points.geojson` (WGS84 `[lon, lat]`); no per-point GeoTIFFs
   (pure 1x1 sparse points).

### Threshold rationale
The `sumcrop` value is a continuous cropland fraction; ~44% of locations are 0%, ~34% are in
(0,50), ~22% are >=50. A **majority (>=50%)** threshold labels a point `cropland` only when
the frame is predominantly cropland, giving a cleaner single-pixel label than a `>0` rule
(which would tag any trace of cropland). The middle band is inherently ambiguous at a single
10 m pixel; keeping `cropland_fraction` lets the assembly step revisit this choice.

## Time range
Campaign submissions are dated **Sept 2016** (Sentinel era). The interpreted VHR imagery is
of **unknown / various years**, so per spec 5 (seasonal/annual labels) every point is
assigned a representative **1-year window anchored on the 2016 campaign year**
(`[2016-01-01, 2017-01-01)`). `change_time` is null (not a change dataset).

**Caveat**: because the underlying VHR imagery years are not recorded, the label may reflect
land state from a year other than 2016; cropland presence is however largely persistent, so
a 2016 window is a reasonable anchor. All labels are >= 2016, so none are dropped under the
pre-2016 rule.

## Metadata (`metadata.json`)
- `classes`: `[{0: cropland}, {1: non-cropland}]` with descriptions.
- `nodata_value`: 255 (uint8 classification convention; not used in the point table).
- `class_counts`: `{cropland: 1000, non-cropland: 1000}`.

## Verification (spec 9)
- `points.geojson`: FeatureCollection, `task_type=classification`, `count=2000`, 2000 Point
  features; label counts `{0:1000, 1:1000}`. No `locations/` dir (correct for point-only).
- All coordinates valid (lon in [-160.2, 177.8], lat in [-45.5, 77.7]); truly global spread
  (32 x 13 ten-degree bins).
- Every point's `label` is consistent with its `cropland_fraction` vs the 50% threshold.
- All `time_range`s are a single calendar year (366 days, leap year 2016 — matches the
  `io.year_range` convention used by the worked `olmoearth_lcmap_land_use` example).
- **Spatial sanity (eyeball)**: sampled points are semantically coherent — a Sahara point
  (`-9.75, 22.45`, Mauritania) has fraction 0 -> non-cropland; a Brazilian cerrado point
  (`-47.25, -16.55`) has fraction 95.2 -> cropland; a Cote d'Ivoire point 0% non-cropland.
  This confirms lon/lat are not swapped and labels are meaningful. (No full S2 raster overlay
  was pulled; points are 1x1 so alignment reduces to correct lon/lat placement.)
- Idempotent: re-running skips the existing raw download/extract and overwrites the (small)
  output tables deterministically (seeded balancing).

## Reproduce
```bash
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.geo_wiki_global_cropland_reference_see_et_al_2017
```
Outputs to
`/weka/dfive-default/helios/dataset_creation/open_set_segmentation/datasets/geo_wiki_global_cropland_reference_see_et_al_2017/`
(`metadata.json`, `points.geojson`, `registry_entry.json`); raw source under
`.../raw/geo_wiki_global_cropland_reference_see_et_al_2017/`.

## Caveats summary
- VHR interpretation years unknown -> representative 2016 window (see above).
- Crowdsourced labels carry interpreter noise; the paper ships control/expert subsets for QA
  (not merged here). Averaging `sumcrop` across users mitigates individual error.
- Binary threshold at 50% is a modeling choice; `cropland_fraction` is retained for
  flexibility.
