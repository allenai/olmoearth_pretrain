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

Per-country last-year-out tasks (`cybench_<crop>_{US,DE,AR}_lyo_eval`):
`tag_eval_subset --per-country US DE AR` writes `loyo_split=test` on every window
of the country's last label year and `loyo_split=train` on a 3k stratified
sample of its earlier years; the probe is scored on the held-out year (reported
as `val`). This is one fold of CY-Bench's leave-one-year-out protocol (they hold
out every year in turn); pooled-subset and per-country scores are both in
`direct_registry.json`. Countries were chosen for labels through 2021+, several
hundred sub-provincial regions, both crops, and a spread of CY-Bench baseline
R2 (DE maize 0.10 .. AR wheat 0.68).

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

## Provenance: exactly how the dataset on weka was built

Every step ran as a Beaker job whose spec is committed under
`scripts/era5_supervised/v0/cybench/`. `__HELIOS_REF__` in a spec is
substituted with the commit to run (`sed "s/__HELIOS_REF__/<sha>/"`) at submit
time; the shas below are what actually ran.

| Step | Spec | Beaker experiment | Commit | Exact invocation / args |
|---|---|---|---|---|
| Download CY-Bench v1.10 + label coverage count | `download_and_count.yaml` | 01M328XCQD9TW25CNPS4WRER14 (2026-09-21) | n/a (inline) | Zenodo record 17279151 -> `raw/{cybench-data,polygons,centroids}`; resume loop because Zenodo drops long transfers; python `zipfile` (image has no unzip) |
| Stages 1-3 (weights, aggregate, series+windows) | `export.yaml` | 01M34H6E1YM5RA0QD1SNMD3344 (2026-09-22, 2h40) | f94f55a13 | `build_weights --polygons-root raw/polygons/polygons --afi-dir <AgML-CY-Bench clone>/data_preparation/global_crop_AFIs_ESA_WC --labels-root raw/cybench-data/cybench-data --out-dir meta --workers 8` ; `aggregate_era5 --meta-dir meta --out-dir agg --start 1998-09-01 --end 2025-07-01 --limit-to-label-years --workers 16 --shard 0 --num-shards 1` ; `write_windows all --agg-dir agg --labels-root raw/cybench-data/cybench-data --ds-path rslearn_dataset --variant crop --min-year 2000 --workers 16` |
| Pooled eval subset tags | `tag_pooled_subset.yaml` | 01M36NPPEX876WN98P5XDY00ZV (2026-09-23) | 9acb7b13f | `tag_eval_subset --labels-root ... --ds-path rslearn_dataset --n-train 3000 --n-val 1000 --n-test 1000 --tag oep_eval --seed 0 --workers 32` -> 10,000 windows |
| Per-country last-year-out tags | `tag_last_year_out.yaml` | 01M390ZTNQKZ3ASFE2A7A37FYV (2026-09-24) | 471f45cc8 | `tag_eval_subset --labels-root ... --ds-path rslearn_dataset --per-country US DE AR --n-train 3000 --lyo-tag loyo_split --seed 0 --workers 32` -> 21,190 windows |
| Training run with the evals | `launch_train_cybench_lyo.sh` | 01M3918WPJ07SZGTVAFP70BMK1 (`..._cybench_lyo`, 11 evals); 01M36VHFRFDS9MNHD9M33N82XV (`..._cybench`, 7 evals) | 471f45cc8 / 060faf414 | see the script (clone of hadriens' era5enc_1306_halo75_nogate, Beaker 01M28KK8ENQF6N8DAXHKPK2J9Q) |

Outputs that record what was produced: `meta/windows_summary_crop.csv`
(per crop/country labels vs windows written vs skips), `meta/label_stats_crop.json`
(yield mean/std per crop and split; the registry `target_mean/std`),
`meta/eval_subset_oep_eval.csv` and `meta/eval_subset_loyo_split.csv` (the exact
windows carrying each tag), `meta/weights.parquet` + `meta/regions.parquet`.

Operational notes from the first launch: the pretraining windows
(`era5enc_pretrain`) had gone cold on weka after ~11 idle days and the run
crawled at ~1 step/min until a 192-way parallel `cat` of all 1.58M files
(Beaker 01M36WTG7Y4V11XDSDEVE3Z9BE, ~85 min) rehydrated them; and the very first
attempt (01M36P66YPDT5774C1X9BMFA02) died on the regression label extractor
not being picklable (fixed in 060faf414).

`export.yaml` runs stages 1-3 in one 0-GPU job (resumable; resubmit after
preemption). Extra aggregation shards can run concurrently with
`SHARD`/`NUM_SHARDS`/`RUN_WINDOWS=0`; the main job waits on
`aggregate_era5 --check-complete` before stage 3.
