# Snow melt-timing eval from ERA5-Land daily

Turns 26 years of MODIS/Terra 8-day snow cover (MOD10A2 collection 6.1,
2000-02-18 onward) into an rslearn dataset for the ERA5 daily encoder evals
(`scripts/era5_supervised/v0`, tasks `snow_*`).

The question each sample asks: given the 448 days of ERA5-Land daily weather
at a seasonal-snow 0.1 deg cell, when will (or did) this year's snow cover
disappear, relative to the cell's usual date? Snow timing integrates snowfall,
freeze / thaw, the temperature and radiation history and rain-on-snow over the
whole cold season, with no vegetation biology involved, so it checks whether an
encoder learned temporal weather structure independently of the plant-stress
signals that drive LFMC, GPP and yield. Absolute timing is dominated by each
cell's static climate (elevation, latitude), so the targets are anomalies
against the cell's own other years, and the headline is a forecast: the
window ends a fixed lead before the cell's usual melt-out.

The ERA5-Land input bands (`common.ERA5L_BANDS`) hold no snow variable, but
three carry ERA5-Land's own snow model: `ssr` / `ssrd` is its modelled albedo,
`ro` includes snowmelt runoff, `e` includes sublimation. A window that spans
the melt therefore shows ERA5-Land's melt-out; the `post` (hindcast) family
measures that, the forecast families mostly cannot use it.

Root on weka: `/weka/dfive-default/rslearn-eai/datasets/snow/`.

## Stages

| Stage | Module | Output |
|---|---|---|
| 1 | `modis screen`, `modis counts` | `raw/modis/tiles.csv`, `raw/modis/parts/h<HH>v<VV>_<YYYY>.npz` |
| 2 | `phenology` | `meta/scf/<tile>.npz`, `meta/phenology/<tile>.parquet`, `meta/phenology.parquet`, `meta/modis_cells.parquet` |
| 3 | `cells` | `meta/cells.parquet`, `meta/era5/<cell_id>.npy`, `meta/era5_dates.npy` |
| 4 | `write_windows candidates` | `meta/cells_split.parquet`, `meta/candidates.parquet` |
| 5 | `write_windows windows` | `meta/eval_keys.parquet`, `meta/windows_{eval,all}_<family>.parquet`, `meta/label_stats.json`, `rslearn_dataset/` |
| - | `partner_labels` | `usdm_snow.parquet`, `fluxnet_snow.parquet` (snow labels for the USDM / FLUXNET evals) |
| - | `stations` | `meta/stations_vs_modis.parquet`, `meta/stations_summary.json` (GHCN-Daily check of the labels) |

`scripts/era5_supervised/v0/snow/launch_build.sh` runs any subset of the
stages (`STEPS=...`) in one CPU-only Beaker job; every stage is resumable or
idempotent.

### MODIS snow counts (stage 1)

The Planetary Computer serves every MOD10A2 granule as per-asset COGs
(`modiseuwest/modis-061-cogs`, anonymous read with a SAS token). Every
`Maximum_Snow_Extent` composite of every kept sinusoidal tile is downloaded
whole and each 500 m pixel is binned into the ERA5-Land 0.1 deg cell holding
its centre (pixel centres from the sinusoidal grid definition): per cell and
composite, land pixels with snow (any snow day in the 8 days), land pixels seen
snow-free, and land pixels of unknown state (cloud on every day, polar night,
no decision). Water (inland water, ocean, lake ice) is left out.

`screen` keeps the tiles of 70 N - 20 N and 20 S - 60 S with at least 3 cells
half snow-covered on 10 / 26 February (NH) or 5 / 21 August (SH) of 2011 or
2018.

The token endpoint answers 429 to the AI2 clusters' shared egress address
whenever other jobs use the Planetary Computer, so the build reads tokens from
a relay file (`raw/modis/sas_token.json`) that
`scripts/era5_supervised/v0/snow/token_relay.py` refreshes from a laptop every
15 minutes, and falls back to the endpoint.

### Phenology (stage 2)

Per cell, snow cover fraction SCF = snow / (snow + no snow) where at least 30 %
of the land pixels were seen, gap-filled linearly in time and filtered with a
3-composite running median (removes the single-composite commission errors of
the maximum-extent product; a monotonic melt curve is unchanged). Water years
run October-September (NH) and April-March (SH), named by the calendar year of
the melt. Per water year: the snow season is the longest run of composites
with SCF >= 0.5 (at least 4); `sdd` is when SCF falls through 0.5 at its end
(linear between composite centres; about the melt-out date of the cell's
median pixel) and `sad` when it rises through 0.5 at its start; `scd` is the
cell-mean count of snow-covered days. Leave-one-year-out climatologies (at
least 8 other years) give the anomalies `*_anom`. A cell split between tiles is
kept by the tile holding at least 80 % of its expected pixels.

### Label check against GHCN-Daily snow depth (`stations`)

12,275 GHCN-Daily stations with daily snow depth (`SNWD`, NOAA's public AWS
copy) lie in seasonal-snow cells; 2,674 of them (North America and Europe;
GHCN has no Russian snow depth) give 45,575 station-years in eligible cells
with both a station and a MODIS melt-out anomaly. A station's SDD is the day
after its longest run of >= 1 cm snow depth. Agreement of the anomalies:
median per-station correlation 0.74 (IQR 0.54-0.87, 2,488 stations with >= 10
years), pooled r = 0.60 (0.63 without the 2 % beyond +-60 days), RMSE 16.7
days; stations melt out 7 days before their cell's median pixel on average
(valley sites). The labels track the in-situ season, with point-to-cell noise
of the order of the anomaly SD.

### Cells and ERA5-Land series (stage 3)

Eligible cells: a snow season in at least 90 % of water years, an SDD in at
least 15, a climatological season of at least 30 days, land in ERA5-Land. One
random eligible cell per 0.5 deg block. Series come from the weka ERA5-Land
mirror (`datasets/era5_land_global.zarr`, the store the pretraining corpus and
the other ERA5 evals were ingested from) from 2000-01-01, one read per 10 x 10
cell chunk column, with the band order, no-data value and 29 February filling
of `fluxnet.era5.Era5Mirror`.

### Targets, families and splits (stages 4-5)

One label per (cell, water year), with an observed melt (bracketing observed
composites at most 24 days apart), in cells with a stable seasonal snowpack:
inter-annual SDD SD at most 20 days and a climatological season of at least 60
days. In short or intermittent snow climates (much of the Tibetan Plateau,
maritime lowlands) the longest snow run jumps between separate snow episodes
from year to year, which produced SDD "anomalies" of +-60-120 days: over all
sampled cells the anomaly kurtosis was 15.5 (1.1 % beyond 60 days); the filter
keeps 87 % of the cell-years and 25,123 of 29,026 cells, brings the kurtosis to
2.7 (SD 15.9 -> 10.4 days) and leaves the station agreement unchanged.
Window families place the 448-day window `[d - 448, d)`:

| Family | Anchor `d` | Targets |
|---|---|---|
| `l0`, `l30`, `l60`, `l90` (forecast) | the cell's climatological melt-out date (leave-one-year-out mean SDD) minus 0 / 30 / 60 / 90 days | `sdd_anom` (= days from `d` to melt-out minus the lead), `sdd_dowy`, `days_to_melt` |
| `post` (hindcast) | first day of the next water year | `sdd_anom`, `scd_anom`, `sad_anom`, `season_anom` (tag `has_sad`), `sdd_dowy`, `scd` |

Splits cross space and time as in the USDM eval: 3 deg blocks go to train /
val / test (70 / 15 / 15 % of cells within south / America / Europe /
north Asia / south Asia), water years by `WY % 5` (2 test, 4 val, else train),
and a window's split is its block's when its water year has the same split,
else `none`.

- `eval_<family>`: the same 4000 / 1500 / 1500 (cell, water year) keys in every
  family (`meta/eval_keys.parquet`), round-robin over cells; registry
  `snow_<target>_<family>_eval`.
- `all_<family>`: every candidate of the train / val / test splits, for
  supervised training; registry `snow_<target>_<family>`.

Window layout as in the FLUXNET / USDM evals: WGS84 0.1 deg/px, 1x1 bounds
whose north-west corner is the cell centre, `era5_daily` raster
`(14, 448, 1, 1)` float32 `NumpyRasterFormat` with per-day timestamps and
nodata -9999 in raw ECMWF units, and a `label` point (at the pixel centre)
with the family's targets as properties.

## Provenance

Filled in as the dataset is built (Beaker experiment, commit, what it did).
