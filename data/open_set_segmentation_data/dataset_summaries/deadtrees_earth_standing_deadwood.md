# deadtrees.earth (standing deadwood)

- **Slug:** `deadtrees_earth_standing_deadwood`
- **Registry status:** completed
- **Task type:** regression (per-pixel fractional cover)
- **Regression target:** `standing_deadwood_fractional_cover` — fraction (0–1) of each 10 m
  pixel's *observed* area covered by standing deadwood
- **num_samples:** 307 label tiles (from 204 orthophoto label sets)
- **Family:** tree_mortality · **Region:** Global (concentrated Central Europe) · **License:** CC BY 4.0

## Source

deadtrees.earth — an open-access global database of centimeter-scale drone/aerial
orthophotos with expert-delineated standing-deadwood polygons (Univ. Freiburg /
Wageningen; Mosig, Schiefer, Kattenborn et al., *Remote Sensing of Environment* 2025,
doi:10.1016/j.rse.2025.115027; portal https://deadtrees.earth). Each manual label set is
tied to one orthophoto and to an **Area-Of-Interest (AOI) multipolygon**: inside the AOI,
area not delineated as deadwood is known to be alive tree / non-tree, so the AOI defines
the *observed* extent (outside the AOI is unobserved). Per-orthophoto acquisition dates and
CC BY licensing are recorded in the platform metadata.

## Access method (label-only, no credential)

The data were read from the platform's **authentication-free, self-hosted Supabase REST
API** (`https://supabase.deadtrees.earth/rest/v1`, anon key extracted from the deployed
frontend) — the same public endpoint the website uses ("download without an account"). We
downloaded **only the labels**: deadwood polygons (`v2_deadwood_geometries`), AOI
multipolygons (`v2_aois`), and dataset metadata (`v2_full_dataset_view_public`). The
cm-scale RGB orthomosaics (COGs) were **not** downloaded — pretraining supplies its own
Sentinel imagery. Total label download is small (a few hundred MB of vectors across 204
datasets). The prepackaged `standing-deadwood-aerial-global-conservative` GeoPackage bundle
exists on the API but its download requires an authenticated account token (not in
`.env`), so the public Supabase read path was used instead.

> Paging note: PostgREST range-paging is only stable with an explicit `ORDER BY`; without
> one, multi-page queries silently skip/repeat rows (observed dropping ~25% of the dataset
> view). All paged queries page over `order=id`.

## Resolution / observability decision (the crux)

Native labels are centimeter-scale; an **individual** standing-dead tree is sub-pixel at
10 m Sentinel resolution, so encoding presence/absence of individual dead trees at 10 m
would be dishonest. Instead we aggregate the cm-scale deadwood mask into the honest 10 m
signal that the manifest itself calls out ("Deadwood fractional cover predictable at 10 m")
and that deadtrees.earth's own satellite models regress:

**fractional standing-deadwood cover per 10 m pixel = deadwood area / observed area**,
restricted to the labeled AOI. This is a **regression** target in [0, 1].

Aggregation method (VHR → 10 m):
1. Reproject the WGS84 deadwood polygons and the AOI multipolygon into the sample's local
   UTM zone and rasterize them onto a **0.5 m sub-grid** (`SUB=20` sub-pixels per 10 m cell).
2. Clip deadwood to the AOI (`dead ∧ aoi`).
3. Average each 20×20 sub-block down to one 10 m pixel: `frac = dead_subpixels / aoi_subpixels`.
4. A 10 m pixel is **observed** (kept) only if ≥ 50 % of its area lies inside the AOI; other
   pixels are nodata (`-99999`). Within the AOI, absence of a deadwood polygon → fraction 0
   (known alive/background).

This encodes contiguous stands of deadwood as a meaningful 10 m fraction rather than
pretending to resolve individual sub-pixel dead trees.

## Label selection & filtering

- **Only manual expert delineations** (`label_source = visual_interpretation`,
  `label_type = semantic_segmentation`, `label_data = deadwood`, `is_active = true`). The
  platform's ~12 k **SegFormer auto-prediction** label sets are a derived ML product and are
  **excluded** to keep this a high-confidence reference bank (SOP §: prefer reference over
  derived maps).
- Datasets restricted to **public**, **CC BY**, **non-archived** records with a **complete
  acquisition date ≥ 2016** (Sentinel era). Of 205 manual deadwood label sets, 54 belong to
  datasets not exposed in the public view and 1 lacks a complete date → **204 qualifying**.
- Platform breakdown of the 204: 203 drone, 1 airborne; label quality 3/3 for 182, 2/3 for
  21, 1/3 for 1 (kept — quality is downstream-filterable and these are still expert labels).

## Tiling, time range, change handling

- AOIs are small drone footprints (median ~230 m), so **most datasets contribute a single
  footprint-sized tile** (e.g. 11×11 px ≈ 110 m); larger airborne AOIs are tiled into a
  grid of ≤ **64×64** tiles at 10 m in local UTM. Tiles with < 25 observed (in-AOI) 10 m
  pixels are dropped. 204 label sets → **307 tiles** (sizes: 131 are 11×11, 76 are 64×64,
  rest intermediate).
- **`change_time = null`** — a single-date deadwood *state*, not a dated change event.
- **`time_range`** = 1-year window anchored on the orthophoto acquisition year (≤ 360 days).
- Acquisition-year distribution of tiles: 2017:53, 2018:5, 2019:61, 2020:34, 2021:121,
  2022:12, 2023:16, 2024:5.

## Value distribution

Deadwood fractional cover is heavily **zero-inflated** (most 10 m pixels are alive /
background). Per-pixel values span the full [0.0, 1.0]. **81.8 %** of tiles contain some
deadwood. Per-tile mean-fraction: median 0.0043, mean 0.0175; histogram (per-tile mean):
`[0,0.001):114  [0.001,0.01):92  [0.01,0.05):68  [0.05,0.1):23  [0.1,0.2):7  [0.2,0.6):3`.
No bucket-balancing was applied (307 ≪ the 5000 regression cap; every tile is kept).

## Output contract

- `datasets/deadtrees_earth_standing_deadwood/metadata.json` — dataset metadata (regression block).
- `datasets/.../locations/{id}.tif` — single-band **float32**, local UTM, 10 m/pixel, ≤ 64×64,
  values = deadwood fraction 0–1, nodata `-99999`.
- `datasets/.../locations/{id}.json` — `crs`, `pixel_bounds`, `time_range` (≤ 1 yr),
  `change_time=null`, `source_id = dataset_{ortho}_label_{label}`.
- `raw/deadtrees_earth_standing_deadwood/labels/{label_id}.json` — cached AOI + deadwood
  polygons + metadata per label set (idempotent).

## Verification (spec §9)

- 307 `.tif` each with a matching `.json`; all float32, single-band, UTM CRS, 10 m
  resolution, size ≤ 64 on both axes.
- All per-pixel values in [0, 1]; only nodata value present is `-99999`.
- All `change_time = null`; all `time_range` ≤ 366 days.
- Spatial sanity: tile-center lon/lat round-trips to inside the source AOI for sampled tiles
  across Germany and South Africa (exact georeferencing). A full Sentinel-2 overlay was not
  rendered, but coordinate round-trip confirms alignment; sites are forested drone captures.

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.deadtrees_earth_standing_deadwood --workers 64
```
Idempotent: cached raw label files and existing output `.tif`s are skipped. (Changing the
qualifying-label set changes the running sample-id → label mapping, so clear
`datasets/<slug>/locations` before a fresh full rebuild.)

## Caveats

- Manual-only reference bank (307 tiles) — modest size by design; the excluded SegFormer
  auto-prediction labels could expand coverage to thousands of tiles as a lower-confidence
  fallback if ever needed.
- Fractional cover is zero-inflated; downstream regression training should expect a sparse
  positive signal.
