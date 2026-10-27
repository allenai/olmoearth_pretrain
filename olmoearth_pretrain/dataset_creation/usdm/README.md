# USDM drought eval from ERA5-Land daily

Turns the weekly U.S. Drought Monitor maps (2000-01-04 onward, D0-D4) into an
rslearn dataset for the ERA5 daily encoder evals (`scripts/era5_supervised/v0`,
tasks `usdm_*`).

The question each sample asks: given the 448 days of ERA5-Land daily weather
at a CONUS 0.1 deg cell ending the day before a weekly map's valid date, what
is the cell's drought state on that map, how has it evolved over the previous
weeks, and how will it evolve over the next weeks? USDM maps are an expert
synthesis of precipitation, soil moisture, streamflow, vegetation and impact
indicators over time scales from weeks to years, so the state is not a
function of the input: the encoder sees one year of local weather and no
location or climatology, while the USDM's indices are percentiles against
each place's own record. Changes over the next weeks depend on weather the
encoder never sees (a forecast); changes over the previous weeks are inside
the input window (the trajectory).

Root on weka: `/weka/dfive-default/rslearn-eai/datasets/usdm/`.

## Stages

| Stage | Module | Output |
|---|---|---|
| 1 | `download` | `raw/maps/USDM_<YYYYMMDD>_M.zip`, `raw/cb_2018_us_state_20m.zip`, `raw/download_status.csv` |
| 2 | `rasterize` | `meta/usdm_sum.npy`, `meta/usdm_inside.npy`, `meta/usdm_dates.npy`, `meta/grid.json`, `meta/usdm_national.csv` |
| 3 | `cells` | `meta/cells.parquet`, `meta/era5/<cell_id>.npy`, `meta/era5_dates.npy` |
| 4 | `write_windows candidates` | `meta/cells_split.parquet`, `meta/candidates.parquet` |
| 5 | `write_windows windows` | `meta/eval_keys.parquet`, `meta/windows_{all,eval}.parquet`, `meta/label_stats.json`, `rslearn_dataset/` |
| - | `fluxnet_labels` | `fluxnet_drought.parquet` (USDM state at the FLUXNET eval's US tower-anchors) |

`scripts/era5_supervised/v0/usdm/launch_build.sh` runs stages 1-4 (or any
subset, `STEPS=...`) in one CPU-only Beaker job; every stage is resumable or
idempotent.

### Labels (stages 1-2)

Each weekly map is one zipped shapefile (EPSG:4326, one multipolygon per
category, categories mutually exclusive; land outside every polygon is "no
drought"). Maps are rasterized onto a grid of 0.1 deg cells centred on the
ERA5-Land cells over CONUS (50 N to 24 N, 125 W to 66.5 W), each split into
4 x 4 sub-cells. A sub-cell inside CONUS (Census 1:20M state boundaries, lower
48 + DC) takes the category of its centre, as severity 0 (none), 1 (D0), ...,
5 (D4); a cell's severity is the mean over its inside sub-cells, i.e. the
cell's DSCI / 100 (USDM's Drought Severity and Coverage Index weights the
cumulative D0-D4 areas equally, which is category + 1 per unit area). Cells
with fewer than 12 of 16 sub-cells inside CONUS are outside the domain.

Check against the official CONUS DSCI (USDM data services API, integer
values): over all 1,399 weekly maps (2000-01-04 to 2026-10-20, no missing
week) r = 0.99997, mean difference -0.42, max |difference| 1.43 (e.g.
2012-08-14: 214.6 vs 215).

Check against in-situ fluxes: at the FLUXNET eval's 266 CONUS towers
(`fluxnet_labels`), the mean May-September 8-day GPP anomaly (FLUXNET
`gpp_anom`, gC m-2 d-1) falls monotonically with the USDM class of the week:
none +0.19, D0 +0.07, D1 -0.07, D2 -0.26, D3-D4 -0.53 (SE 0.03-0.06, n 1,213
to 7,381 tower-anchors per class).

### Cells and ERA5-Land series (stage 3)

USDM polygons are hundreds of km across, so neighbouring cells carry nearly
the same label and weather: one cell per 0.5 deg block (a random domain cell
that is land in ERA5-Land, seeded) keeps the label field's spatial detail.
Series come from the weka ERA5-Land mirror (`datasets/era5_land_global.zarr`,
the store the pretraining corpus and the other ERA5 evals were ingested from),
one read per 10 x 10 cell chunk column, with the band order, no-data value and
29 February filling of `fluxnet.era5.Era5Mirror` (the mirror has no 29
February; those days are the mean of 28 February and 1 March).

### Targets and splits (stage 4)

Anchors are the map dates (Tuesdays). The ERA5 window is `[d - 448, d)`; the
map valid on `d` is drawn from data up to that morning. Per (cell, anchor):

| Property | Target |
|---|---|
| `sev` | severity at the anchor, 0-5 |
| `sev_m{4,8,13,26}`, `sev_p{4,8,13}` | severity 4-26 weeks before / 4-13 weeks after |
| `dpast{4,8,13,26}` | `sev - sev_m<L>`: recent evolution (inside the input window) |
| `dfut{4,8,13}` | `sev_p<L> - sev`: coming evolution (a forecast) |
| `tpast8`, `tfut8` | `dpast8` / `dfut8` as improve / stable / worsen (`|change| >= 0.5`) |
| `drought` | `sev >= 2` (D1 or worse) |

Every candidate has every lag defined (no gap in the weekly record).

Splits cross space and time, so no test window shares a cell, a landscape or a
season with training: 3 deg blocks go to train / val / test (70 / 15 / 15 % of
cells within west / central / east CONUS), water years (October-September) by
`WY % 5` (2 test, 4 val, else train), and a window's split is its block's when
its water year has the same split, else `none` (unused). Test water years:
2002, 2007, 2012, 2017, 2022; val: 2004, 2009, 2014, 2019, 2024.

### Windows (stage 5)

- `all`: every candidate of the train / val / test splits on a 12-week anchor
  grid (random phase per cell), for supervised training (registry `usdm_sev`,
  `usdm_dpast8`, `usdm_dfut8`).
- `eval`: 4000 train / 1500 val / 1500 test windows for the in-loop linear
  probes (registry `usdm_*_eval`), fixed by `meta/eval_keys.parquet`. A uniform
  draw would be mostly wet, steady cell-weeks, so each split is drawn by
  stratum (drought state wet / dry / drought x 8-week dynamics steady /
  moving, `write_windows.STRATUM_SHARES`), round-robin over cells and water
  years within a stratum.

Window layout as in the FLUXNET eval: WGS84 0.1 deg/px, 1x1 bounds whose
north-west corner is the cell centre, `era5_daily` raster `(14, 448, 1, 1)`
float32 `NumpyRasterFormat` with per-day timestamps and nodata -9999 in raw
ECMWF units, and a `label` point (at the pixel centre) with every target as a
property.

## Provenance: how the dataset on weka was built (2026-10-27)

CPU-only Beaker jobs on one Jupiter node each (48 workers), from
`scripts/era5_supervised/v0/usdm/launch_build.sh`.

| Step | Beaker experiment | Commit | What it did |
|---|---|---|---|
| Build | 01M5ZNKXJG2465H95B9XFNWR3J | 706bf8e1a | Download (1,399 maps 2000-01-04..2026-10-20; 2026-10-27 not yet published), rasterize (81,751 domain cells), cells (3,457 sampled, 912 mirror chunk columns), candidates (4.63M: train 1.95M / val 125k / test 154k / none 2.40M). 15 min. |
| Eval windows | 01M5ZPMFDSC04V1Z13FCBJHKE9 | 8ce5917e8 | `eval` group: 4000 / 1500 / 1500 windows over 3,137 cells, keys in `meta/eval_keys.parquet`. |
| All windows | 01M5ZPTZXP9KFG803Y3V811XYH | 8ce5917e8 | `all` group: 185,772 windows (train 162,526 / val 10,417 / test 12,829). |
| Tower labels | (weka session) | 4670fdcb1 | `fluxnet_drought.parquet`: 33,098 FLUXNET windows at 266 CONUS towers. |

Final state: 3,457 cells, `eval` 7,000 windows, `all` 185,772 windows;
`meta/label_stats.json` holds the per-split target statistics (registry
`target_mean` / `target_std`), `meta/baselines_windows_eval.json` the reference
baselines.
