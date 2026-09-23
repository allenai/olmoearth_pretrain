# CY-Bench yield eval from ERA5-Land daily

Turns the [CY-Bench](https://github.com/WUR-AI/AgML-CY-Bench) crop-yield
benchmark (maize + wheat, sub-national yields, Zenodo record 17279151, v1.10)
into an rslearn dataset for the ERA5 daily encoder evals
(`scripts/era5_supervised/v0`, tasks `cybench_maize_eval` / `cybench_wheat_eval`).

CY-Bench's own predictors are six AgERA5-derived variables (tmin, tmax, tavg,
prec, cwb, rad); our encoder needs the 14 raw ERA5-Land bands at exactly 448
daily steps. So we take CY-Bench's **labels, admin polygons and crop
calendars** and build the ERA5-Land input ourselves, reproducing CY-Bench's
aggregation (crop-area-fraction weighted mean over the admin polygon) on the
ERA5-Land daily UTC Zarr that the pretraining corpus was ingested from.

No rslearn ingest / tile store: Zarr chunks are streamed and reduced in memory
(a 500 GB tile store would otherwise be needed for ~10k chunks).

## Stages

| Stage | Module | Output |
|---|---|---|
| 0 | (manual) CY-Bench download | `raw/{cybench-data,polygons,centroids}` |
| 1 | `build_weights` | `meta/weights.parquet` (region x ERA5 cell: `w_area`, `w_crop`), `meta/regions.parquet` |
| 2 | `aggregate_era5` | `agg/partials/t{tc}_y{latc}_x{lonc}.npz` (per-region means / edge sums per Zarr chunk), `agg/dates.npy` |
| 3a | `write_windows series` | `agg/series_crop/<ridx>.npy` — one `(T_all, 14)` series per region |
| 3b | `write_windows windows` | `rslearn_dataset/` — one window per (adm_id, harvest_year), + `meta/windows_summary_*.csv`, `meta/label_stats_*.json` |

| 4 | `tag_eval_subset` | `oep_eval` tag on a stratified 3k/1k/1k subset per crop (the `cybench_<crop>_eval` registry entries); `meta/eval_subset_oep_eval.csv` |

Root on weka: `/weka/dfive-default/helios/dataset/cybench/`.

Registry: `cybench_<crop>` = full dataset (too large for the in-loop evaluator,
which embeds the whole train split every eval interval); `cybench_<crop>_eval`
= the `oep_eval` subset, like `lfmc_woody` vs `lfmc_woody_eval`.

### Weights (stage 1)

Each admin polygon is rasterized on a 0.01 deg sub-grid (10x10 per ERA5-Land
0.1 deg cell). `w_area` = fraction of the cell inside the polygon; `w_crop` =
coverage-weighted WorldCereal crop area fraction (maize AFI for maize,
winter+spring cereals AFI for wheat; both tifs ship in the CY-Bench repo under
`data_preparation/global_crop_AFIs_ESA_WC`). Regions with (near-)zero crop
weight fall back to `w_area` (`crop_fallback` flag). Polygons that cross the
antimeridian are split at +-180.

### Aggregation (stage 2)

Uses rslearn's `ERA5LandDailyUTCv1` for auth + chunk geometry
(`EARTHDATAHUB_TOKEN` required; Beaker secret `RSLEARN_EARTHDATAHUB_TOKEN`).
Per `(time, lat, lon)` chunk: sparse `(regions x cells) @ (cells x days*bands)`
for the weighted sums and, separately, for the *valid-cell* weight totals, so
ocean / lake cells (NaN in ERA5-Land) drop out of the mean per band and day.
Regions fully inside a chunk get final means; regions on a chunk boundary emit
partial sums that stage 3a merges. Resumable (skips existing partials),
shardable (`--shard i --num-shards n`), and `--limit-to-label-years` restricts
each spatial chunk to `[min label year - 2, max label year]`.

### Windows (stage 3)

For each label `(crop, cc, adm_id, harvest_year)` with a crop-calendar row:
window = 448 days ending at the calendar end of season (`eos` DOY in the
harvest year, clamped to Dec 31), i.e. the same anchor CY-Bench uses before it
truncates at a forecast lead time. Windows with any fully-missing day are
skipped (counted in `windows_summary`). Output window:

- group `<crop>`, name `<crop>_<cc>_<adm_id>_<year>`, WGS84 0.1 deg/px, bounds
  = the ERA5 cell holding the polygon's representative point (layer is 1x1);
- `era5_daily`: `(14, 448, 1, 1)` float32 `NumpyRasterFormat`, per-day
  timestamps, nodata -9999, raw ECMWF units (normalization happens in the eval
  loader via `computed.json`, as for every other ERA5 task);
- `label`: one point feature with `yield` (t/ha), read by rslearn's
  `RegressionTask(property_name="yield")`;
- tags: `crop, country_code, adm_id, harvest_year, split` with
  `split` = train (<= 2017) / val (2018-2019) / test (>= 2020).

Only harvest years >= 2000 are written by default (`--min-year`, CY-Bench's
`MIN_INPUT_YEAR`); the aggregation range defaults to 1998-09-01 .. 2025-07-01.

### Protocol caveats

- CY-Bench scores leave-one-year-out per country with normalized RMSE / MAPE /
  R2 and truncates inputs at a lead time (default middle of season). Our
  in-loop probe uses the fixed year split above on full-season inputs. Numbers
  comparable to the CY-Bench tables need an offline LOYO pass on cached
  embeddings plugged into CY-Bench's `run_benchmark`.
- EU statistics stop in 2020, India in 2017: a year-based test split
  under-represents them.
- `target_mean` / `target_std` in `direct_registry.json` must be filled from
  `meta/label_stats_crop.json` (`train` split) after the export finishes.

## Running on Beaker

`scripts/era5_supervised/v0/cybench_export.yaml` runs stages 1-3 in one 0-GPU
job (resumable; resubmit after preemption). Extra aggregation shards can run
concurrently with `SHARD`/`NUM_SHARDS`/`RUN_WINDOWS=0`; the main job waits on
`aggregate_era5 --check-complete` before stage 3.
