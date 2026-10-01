# US Forest Inventory & Analysis (FIA)

- **Slug:** `us_forest_inventory_analysis_fia`
- **Manifest name:** US Forest Inventory & Analysis (FIA)
- **Family / label_type:** tree_species / points
- **Source:** USDA Forest Service, FIA DataMart (https://research.fs.usda.gov/products/dataandtools/fia-datamart). License: public domain.
- **Status: REJECTED** — fundamental reason: **no recoverable geocoordinates (coordinate-fuzzed + swapped points).** This is the canonical FIA rejection called out in the spec (§8, "coordinate-fuzzed points like FIA ~1 mi").

## What the source is

The USDA Forest Service Forest Inventory & Analysis program maintains a national network
of ~500k+ permanent field plots. Each plot records tree-level measurements (species,
diameter, height, condition, mortality, etc.) plus plot-level attributes (forest type,
forest/nonforest, live biomass, carbon). The public database (FIADB) is distributed via
the FIA DataMart as CSV/SQLite by state. Task fit would otherwise be excellent: a genuine
per-plot tree-species / forest-condition label in the Sentinel era (manifest time_range
2016–2026), expressible as sparse-point classification (§2a/§4) — hundreds of US tree
species — or as a plot-level forest/nonforest or biomass regression target.

## Why it is rejected

The public FIA release **does not publish placeable geocoordinates.** To protect
landowner privacy (required by the Food Security Act of 1985 / 7 U.S.C. 2276), FIA
deliberately degrades the public LAT/LON in the PLOT table by two mechanisms, both still
in force for the current DataMart release (verified July 2026):

- **Fuzzing:** most plot coordinates are randomly relocated within **0.5 mile** of the
  true location, with the remainder moved **up to 1 mile** — i.e. the true plot is masked
  within roughly a 500-acre area.
- **Swapping:** for up to **20% of privately-owned forested plots per county**, the
  coordinates are exchanged with a *different similar plot in the same county*, so the
  published point may not even be near the plot it describes.

Actual (unfuzzed) plot coordinates are confidential and shared only rarely under a
specific, limited USFS Spatial Data Services agreement — not available through any
unauthenticated/public path.

For this effort a label must be co-located with imagery on the 10 m Sentinel-2 grid. A
positional error of up to ~1 mile (~1600 m ≈ **160 pixels** at 10 m) is more than 2× the
maximum 64×64 tile footprint (640 m), so a plot's species/condition label cannot be pinned
to any specific 10 m pixel or even reliably placed within a single tile. Swapped plots can
be arbitrarily wrong. This is exactly the situation the spec names as a rejection: "phenomenon
not observable … coordinate-fuzzed points like FIA ~1 mi, and no aggregate/mask
representation salvages it" (§8) → **no recoverable geocoordinates** (§1a fundamental reason).

## Judgment calls

- **Considered a salvageable plot-level aggregate** (per task note: forest/nonforest or
  biomass at coarse cells). Rejected: to be robust to ~1 mile fuzzing + county-scale
  swapping, an aggregation cell would have to be several km across — far larger than the
  64×64 (640 m) tile cap and the 10 m point grid this effort targets. There is no coarse
  representation that both respects the output contract and survives the positional noise.
- **Better georeferenced alternatives already exist in the manifest** for the salvageable
  aggregates: `annual_nlcd_reference_data` (NLCD land cover incl. forest classes) and
  derived canopy-height / biomass map products cover forest cover/structure with true
  georeferencing, so the fuzzed FIA aggregate adds nothing.
- **`rejected`, not `temporary_failure` or `needs-credential`.** The source is fully
  accessible without credentials; the blocker is a *permanent, deliberate property of the
  public release* (legally mandated coordinate degradation), not a transient outage or a
  simple access gate. (True coordinates could in principle be obtained via a USFS Spatial
  Data Services agreement, but that is a rare, restricted research arrangement, not a
  credential the user can readily supply — noted for completeness.)

## Reproduce / verify the rejection

- FIA DataMart: https://research.fs.usda.gov/products/dataandtools/fia-datamart — public
  FIADB downloads; PLOT table LAT/LON are the fuzzed/swapped public coordinates.
- Privacy methodology & confidentiality: USFS FIA Spatial Data Services,
  https://research.fs.usda.gov/programs/fia/sds — documents the fuzzing (0.5 mi, up to
  1 mi) and swapping (≤20% of private forested plots per county) applied to all public
  coordinates, and that actual locations are shared only under limited agreement.
- FIADB Database Description (v9.2, Apr 2024): documents that public plot coordinates are
  fuzzed/swapped.

No outputs written to weka `datasets/` beyond the required `registry_entry.json`.
