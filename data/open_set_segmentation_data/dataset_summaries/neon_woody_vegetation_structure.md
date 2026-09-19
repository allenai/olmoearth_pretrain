# NEON Woody Vegetation Structure — REJECTED (needs-credential)

- **Slug:** `neon_woody_vegetation_structure`
- **Manifest name:** NEON Woody Vegetation Structure
- **Source:** NSF NEON, data product **DP1.10098.001** ("Vegetation structure")
- **URL:** https://www.neonscience.org/data-products/DP1.10098.001
- **Family / label_type:** tree_species / points (in-situ field-plot woody-stem measurements)
- **Region / time_range:** United States (42 terrestrial sites) / 2014–2025 (Sentinel-era subset 2016+)
- **Final status:** `rejected` — `notes: needs-credential`
- **Task type (intended if creds obtained):** regression (plot-level canopy height / biomass)

## Rejection reason (blocking): NEON now requires an authenticated account/API token

As of **2026-06-30**, NEON changed its access policy: downloading data via the NEON API
(or neonUtilities) now **requires a NEON user account / API token**, and the data license
moved from **CC0 to CC BY 4.0**. See
https://www.neonscience.org/impact/observatory-blog/upcoming-required-logins-and-data-licensing-updates .

Today (2026-07-11) this is already in effect. Empirically verified from this environment:

- `GET https://data.neonscience.org/api/v0/products/DP1.10098.001` → **200** (metadata is
  still open; 42 sites, monthly availability, `availableDataUrls` all resolve to the
  `/data/…` endpoint below).
- `GET https://data.neonscience.org/api/v0/data/DP1.10098.001/{SITE}/{YYYY-MM}` → **403**
  `{"error":{"status":403,"detail":"Access Denied"},"data":null}` for **every** site/month
  tried (ABBY 2016-08, 2023-08, 2025-10; BART 2018-08; also product DP1.10003.001) —
  regardless of `release=` param or browser headers. Rate-limit headers show
  `x-ratelimit-remaining: 198`, so it is **not** rate-limiting.
- The same `/data/` URL returns **403** via an independent egress (Anthropic WebFetch),
  confirming it is a **NEON server-side auth gate**, not an IP block or transient outage.

This is a **permanent access gate requiring a credential we do not have** — the
`needs-credential` rejection case in the task spec (§8), not `temporary_failure`. The file
listing/download endpoints cannot be reached without a NEON token, so no raw stems can be
pulled and placed on the S2 grid.

**To unblock (retry recipe):** create a free NEON account, generate an API token
(https://data.neonscience.org/data-api tokens page), and re-run with the token available
(e.g. env `NEON_TOKEN`, passed as the `X-API-Token` request header), OR supply a
pre-downloaded copy of DP1.10098.001 (basic package, RELEASE-2026) on weka under
`raw/neon_woody_vegetation_structure/`.

## Suitability assessment (why regression, not per-stem species)

DP1.10098.001 is **individual-tree stem measurements** in NEON field plots: per stem a
`taxonID` (species), `stemDiameter` (DBH), `height`, `plantStatus`/`growthForm`, and a
mapped location (via `pointID` + `stemDistance`/`stemAzimuth`, or plot centroid).

- **Per-stem species labels are not usable at 10–30 m.** A single tree stem is sub-pixel
  at Sentinel-2/Landsat resolution; a "one species per point" encoding (like GlobalGeoTree)
  would be almost pure noise here. Reject that framing.
- **Plot-level aggregate regression IS a good fit.** NEON plots (20×20 m tower / 40×40 m
  distributed, i.e. ~2×2 to 4×4 S2 pixels) can be reduced to a single canopy-structure
  value at the plot centroid — directly comparable to S2/GEDI canopy-height and biomass
  products, which are the intended OlmoEarth regression comparison. NEON plot coordinates
  are **not coordinate-fuzzed** (unlike FIA's ~1 mi swap); `vst_perplotperyear` carries
  `decimalLatitude`/`decimalLongitude` + `coordinateUncertainty` per plot.

## Intended processing recipe (once a token / local copy is available)

1. **Download** (token in `X-API-Token`): iterate `siteCodes[*].availableDataUrls` from the
   products endpoint, keep months `>= 2016-01` (drop the small pre-2016 subset, Sentinel
   era), pull the `.csv` tables per site-month:
   - `vst_perplotperyear` → `plotID`, `eventID`/year, `decimalLatitude`, `decimalLongitude`,
     `coordinateUncertainty`, `plotType`, `totalSampledAreaTrees`.
   - `vst_apparentindividual` → `individualID`, `plotID`, `eventID`, `height`,
     `stemDiameter`, `plantStatus`, `growthForm`.
   - (`vst_mappingandtagging` for `taxonID`/finer geolocation — not needed for the plot
     aggregate.)
2. **Aggregate per (plotID, year)** over live woody stems (`plantStatus` contains "Live"):
   choose the regression target — recommended **`canopy_height`** = mean (or 95th-pct) of
   measured `height` per plot; alternative **`aboveground_biomass`** via species allometry
   from `stemDiameter` (more assumptions). Attach the plot's `decimalLatitude/Longitude`.
   Drop plots with too few measured stems or `coordinateUncertainty` above a threshold
   (e.g. > 30 m).
3. **Write** a sparse-point **regression** table `points.geojson` via
   `io.write_points_table(slug, "regression", points)` — one `Point` feature per
   (plot, year): `properties.label = <canopy_height meters>`, `time_range =
   io.year_range(year)` (annual window anchored on the sampling year), `source_id =
   "{plotID}_{year}"`. Expected O(few thousand) plot-years across 42 sites × ~2016–2024,
   comfortably under the 5000-sample regression cap (bucket-balance only if skewed).
4. `metadata.json` with a `regression` block (`name: "canopy_height"`, `unit: "meters"`,
   `dtype: float32`, `value_range` from data, `nodata_value: -99999`).

No `datasets/` label outputs, `metadata.json`, or `points.geojson` were written (rejected);
only `datasets/neon_woody_vegetation_structure/registry_entry.json` was written on weka.

## Reproduce (the check that produced this rejection)

```bash
curl -s "https://data.neonscience.org/api/v0/products/DP1.10098.001"        # 200 (metadata open)
curl -s "https://data.neonscience.org/api/v0/data/DP1.10098.001/ABBY/2016-08"  # 403 Access Denied (token required)
```
Then, when a NEON token or local DP1.10098.001 copy is provided, implement the recipe above
as `olmoearth_pretrain/open_set_segmentation_data/datasets/neon_woody_vegetation_structure.py`
and run `python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.neon_woody_vegetation_structure`.
