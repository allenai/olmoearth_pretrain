# FLUXNET flux-tower eval from ERA5-Land daily

Turns the eddy-covariance flux tower archives of the FLUXNET network
(ONEFlux v1.3 FLUXNET-format products from AmeriFlux, the ICOS Carbon Portal
and TERN, plus older FLUXNET2015-style FULLSET archives some sites still use)
into an rslearn dataset for the ERA5 daily encoder evals
(`scripts/era5_supervised/v0`, tasks `fluxnet_*_eval`).

The question each sample asks: given the 448 days of ERA5-Land daily weather
at a tower's 0.1 deg cell, what were the tower's mean fluxes over the last 8
days? GPP needs the phenological state (temperature and moisture history) as
well as the current radiation and VPD, so it exercises the whole history, and
it is continuous, global and has real headroom. The encoder sees only
meteorology, so the targets are not functions of the input: GPP is not an
ERA5-Land variable. LE is partly circular (ERA5-Land's `e` is modelled
evaporation); it is kept as a secondary eval.

Root on weka: `/weka/dfive-default/rslearn-eai/datasets/fluxnet/`.

## Stages

| Stage | Module | Output |
|---|---|---|
| 1 | `download` | `raw/catalog.csv`, `raw/<hub>/*.zip`, `raw/download_status.csv` |
| 2 | `targets` | `meta/daily/<site>__<hub>.parquet`, `meta/sites.parquet` |
| 3 | `era5` | `meta/era5/<site>.npy`, `meta/era5_dates.npy`, `meta/era5_cells.parquet` |
| 4 | `write_windows` | `meta/candidates.parquet`, `meta/sites_split.parquet`, `meta/windows_{all,eval}.parquet`, `meta/label_stats.json`, `rslearn_dataset/` |

`scripts/era5_supervised/v0/fluxnet/launch_build.sh` runs all four in one
CPU-only Beaker job; every stage is resumable or idempotent.

### Download (stage 1)

Does what the FLUXNET Shuttle (github.com/fluxnet/shuttle, announced May 2026)
does, with plain HTTP calls: AmeriFlux `amfcdn.lbl.gov/api/v2` (downloads are
logged with the Shuttle's anonymous `fluxnetshuttle` id, no personal
information), the ICOS SPARQL endpoint (`miscFluxnetArchiveProduct`: ICOS,
European Fluxes Database, JapanFlux, ChinaFlux, KoFlux, SAEON and the FLUXNET
legacy collection; the `licence_accept` link accepts CC-BY-4.0 per object) and
TERN's THREDDS catalogue. No account is needed. All data are CC-BY-4.0:
`raw/catalog.csv` keeps every product's citation, and any publication must
cite each site's product.

### Targets (stage 2)

From each archive's daily file (`*_FLUXMET_DD_*`, or `*_FULLSET_DD_*` in
legacy archives) and BIF site metadata. A day is valid for GPP when
`GPP_NT_VUT_REF` is present and `NEE_VUT_REF_QC` (the daily fraction of
measured or good-quality gap-filled half-hours) is >= 0.5; for LE likewise
with `LE_F_MDS_QC`.

### ERA5-Land series (stage 3)

From the weka mirror `datasets/era5_land_global.zarr` of the EarthDataHub
ERA5-Land daily UTC store that the pretraining corpus and the other ERA5 evals
were ingested from (rechunked to 10x10 cells x 3650 days, so a site is five
chunk reads). `era5 validate` compares it with the remote store value by value:
identical, except that **the mirror has no 29 February** (12 days over
1980-2026); those days are filled with the mean of 28 February and 1 March, so
at most one day of a window is synthetic.

Each site takes the nearest ERA5-Land land cell (up to 3 cells away for coastal
towers; the offset is recorded) and its own values, not interpolated ones.
rslearn-materialized windows from UTM projections (DFMC, the pretraining
corpus) carry resampled values instead; they correlate > 0.99 with the nearest
cell in flat terrain and less in mountains.

### Windows (stage 4)

Anchor dates `d` sit on a global 8-day grid, so target periods never overlap.
The ERA5 window is `[d - 448, d)` (as for DFMC: it ends the day before the
anchor) and the targets are means over its last 8 days `[d - 8, d)`, with at
least 6 valid days:

| Property | Target |
|---|---|
| `gpp` | night-time-partitioned GPP (`GPP_NT_VUT_REF`), gC m-2 d-1 |
| `gpp_dt` | day-time-partitioned GPP on the same days |
| `le` | latent heat `LE_F_MDS`, W m-2 |
| `nee` | `NEE_VUT_REF` on the GPP days |
| `gpp_anom` | `gpp` minus the site's leave-one-year-out climatology at the same day of year (8-day GPP within +-7 days in every other year; >= 2 other years) |

Splits are spatial: towers within 50 km of each other (single linkage) form a
group, and groups go to train / val / test (70 / 15 / 15 % of sites) within
each coarse region (North and South America, Europe, Africa, Asia, Oceania).
Towers sharing a landscape or an ERA5 cell never straddle splits.

Two window groups, one window per (site, anchor):

- `all`: every candidate with `gpp`, on a 16-day grid, for supervised training
  (registry `fluxnet_gpp`, `fluxnet_le`, `fluxnet_gpp_anom`; tags `has_le`,
  `has_anom` select the windows defining each target).
- `eval`: a site-balanced subset (4000 train / 1500 val / 1500 test; sites are
  visited round-robin, so every site contributes before any contributes twice)
  of the candidates where `gpp`, `gpp_dt`, `le` and `nee` are all defined; the
  registry `fluxnet_*_eval` tasks. A separate group so the in-loop evaluator does
  not scan the full set.

Window layout: WGS84 0.1 deg/px, 1x1 bounds on the pixel holding the tower
(rslearn only reads label features inside the window), `era5_daily` raster
`(14, 448, 1, 1)` float32 `NumpyRasterFormat` with per-day timestamps and
nodata -9999 in raw ECMWF units (normalization happens in the eval loader), and
a `label` point with the defined targets as properties. The task configs read
one property each with `allow_invalid: false`, so a missing target fails
instead of reading as 0.

## Provenance

See `scripts/era5_supervised/v0/fluxnet/` for the launchers and the record of
what ran (filled in after the build).
