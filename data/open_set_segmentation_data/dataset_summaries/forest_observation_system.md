# Forest Observation System (FOS) — aboveground biomass (regression)

- **Slug**: `forest_observation_system`
- **Status**: completed
- **Task type**: regression (point table, spec §2a)
- **Num samples**: 247 points
- **Family / region**: biomass / Global
- **License**: CC-BY-4.0

## Source

The [Forest Observation System](http://forest-observation-system.net/) (FOS) is an
international initiative compiling a global in-situ reference of forest **aboveground
biomass (AGB)** and canopy height from permanent research plots, to calibrate/validate
Earth-observation biomass products (Schepaschenko et al. 2019, *Sci Data* 6:198,
doi:10.1038/s41597-019-0196-1).

The live portal offers an extended, accurately-geolocated dataset behind **free
registration** (no credential in `.env`). Per the task, we instead used the **open,
peer-reviewed Sci Data data package** (CC-BY-4.0), hosted by IIASA:

- Download: `https://pure.iiasa.ac.at/id/eprint/17619/1/FOS_data_v2019.04.10.zip`
- DOI: `10.22022/ESM/03-2019.38` (IIASA eprint 17619)

The zip ships one Excel workbook (`FOS_data_v2019.04.10.xlsx`, sheet `Plots`): 1,645
sub-plots across 274 plots. Each sub-plot row carries the plot center lon/lat, census
year, plot area, AGB (Mg/ha) under three allometries (`AGB_local` / `AGB_Chave` /
`AGB_Feldpausch`), Lorey's and max canopy height (m), wood density, basal area, stem
density, etc. **In this open package coordinates are rounded to 2 decimal places (~1 km);**
the exact-geolocation version is portal-only.

## Access method

Anonymous HTTPS download of the IIASA-hosted zip → `raw/forest_observation_system/`
(`download.download_http` + `extract_zip`). No credential required.

## Label mapping / target

- **Regression target** = **plot-mean AGB (`AGB_local`), Mg/ha (= t/ha)** — the source's
  most direct estimate (local allometric equations / Chave 2014 eq.4 with local H–D
  curves). The `metadata.json` `regression` block holds this single quantity
  (`name=aboveground_biomass`, `unit=Mg/ha (t/ha)`, `dtype=float32`,
  `value_range=[0.15, 609.3]`, `nodata=-99999`).
- **Canopy height** (the secondary quantity in the manifest) is *available but not
  regressed* — carried per point as `canopy_height_lorey_m` / `canopy_height_max_m`.
  `AGB_Chave` / `AGB_Feldpausch` are likewise attached as auxiliary point properties, plus
  `census_year`, `plot_area_ha`, `n_subplots`, `country`, `network`.

## Plot-level aggregation (key decision)

The open package rounds coordinates to 2 dp (~1 km), so all sub-plots of a plot collapse
onto a single ~1 km cell (240/274 plots share one 2dp coord; up to 136 sub-plots at one
coord). Emitting sub-plots would place contradictory AGB values on the same 10 m pixel, so
each plot is **aggregated to one point at its center with the plot-mean AGB** (FOS itself
recommends "use a plot average" to avoid within-plot spatial autocorrelation). Of 274
plots, **247** have a valid plot-mean `AGB_local > 0` with real coordinates and are kept.

## Sampling

247 samples is well under the 5,000-sample regression cap (spec §5), so **all valid plots
are kept — no downsampling and no bucket-balancing** (the AGB distribution is only mildly
right-skewed). AGB histogram (Mg/ha, 100-wide bins 0–800):
`[35, 66, 86, 47, 8, 4, 1, 0]`. Distribution spans boreal Russia (~60 Mg/ha), temperate
Europe (~130), and tropical Amazonia/Africa/SE-Asia (250–600); one near-zero
disturbed/savanna plot in Gabon (0.1).

## Time range / change handling

- Plots censused **2016+** (92 plots) use a 1-year window on their census year
  (233 features fall in the 2016 window, 12 in 2017, 2 in 2018 after the pre-2016 remap).
- Plots censused **before 2016** (median census ~2010) get a representative Sentinel-era
  1-year window anchored on **2016** (earliest full Sentinel-2 year → minimizes the gap to
  the field measurement). Justified because AGB at established permanent research plots is
  slowly-varying. No change labels (`change_time=null`).

## Tile size

Sparse-point regression → **point table** (`points.geojson`), no per-sample GeoTIFFs
(spec §2a). Plot footprints (~0.25 ha) are ~1–3 px at 10 m, so a 1×1 point is appropriate.

## Verification

- `points.geojson`: 247 `Point` features, `task_type=regression`, all `time_range`s ≤ 1
  year, labels in [0.15, 609.3] Mg/ha, coords global (lon −83.58…148.92, lat −36.52…64.51).
- Geolocation sanity: spot-checked country vs coordinates for 17 countries — all consistent
  (e.g. Brazil −55.94/−9.60 in S Amazon; Russia 42.91/64.51 boreal at ~60 Mg/ha; Cameroon
  12.72/3.33 at ~581 Mg/ha; Australia NT tropical savanna at ~22 Mg/ha).
- `metadata.json` regression block present with value range covering all labels.
- Script is idempotent (download skips existing; outputs overwritten deterministically).

## Caveats

1. **~1 km coordinate rounding** in the open package — points are only approximately
   located relative to a 10 m pixel (the FOS portal has exact geolocation behind
   registration if a more precise version is needed later).
2. Most census dates **predate Sentinel-2**; pre-2016 plots rely on the quasi-static-AGB
   assumption for old-growth/permanent plots.

## Reproduce

```bash
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.forest_observation_system
```
