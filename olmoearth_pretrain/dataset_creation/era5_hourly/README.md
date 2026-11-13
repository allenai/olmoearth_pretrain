# ERA5-Land hourly pretraining corpus

Hourly ERA5-Land series, 365 days long, for every window of the daily ERA5
pretraining corpus (`/weka/dfive-default/helios/dataset/era5enc_pretrain`),
packed into `.npy` shards of up to 10,000 windows so a sample is one contiguous
read. The corpus itself is only read: each window is taken as defined, and
its hourly series is the 8,760 hours ending at the window's end time.

Root on weka: `/weka/dfive-default/helios/dataset/era5enc_pretrain/era5L_hourly_365d/`.

## Source

EarthDataHub's hourly ERA5-Land store (`era5-land-v0.zarr`, zarr v3, 1950 to
the end of the previous month, 90N to 57.1S on the 0.1 deg grid of the weka
daily mirror, chunks of 1440 h x 50 x 100 cells). There is no local copy:
each shard job streams the store tiles its windows fall in, so a tile is
downloaded once. Values are bit-rounded to 13 mantissa bits by EarthDataHub.

## Layout

| File | Content |
|---|---|
| `shards/shard_XXX.npy` | `(n_windows, 8760, 27)` float32, raw ECMWF units, `-9999` where the store has no value |
| `index.parquet` | one row per corpus window: `window`, `status`, `shard`, `row` (-1 when not stored), centre and ERA5-Land cell (`ilat`, `ilon`), `first_hour` / `end_hour` / `first_time`, the corpus time range and options (role, Koppen class, ...) |
| `bands.json` | band order, units, accumulation and time conventions |
| `shards.json` | the store tiles and time chunks of each shard |
| `static/static.npy` | `(11, 1472, 3600)` time-invariant fields (`static.json` gives the order) indexed by `ilat, ilon` |
| `progress/` | per-tile markers, per-shard build and validation summaries |

Row `k` of a window is store hour `first_hour + k` (UTC, hours since
1950-01-01T00). Instantaneous bands are the value at that hour; accumulated
bands are the amount over the hour ending there. The store holds running
totals from 00 UTC (the 00 UTC value closes the previous day), which
`common.deaccumulate` turns into hourly amounts.

Shards are made of whole store tiles ordered by longitude, so a shard covers
one longitude band and is not a random sample of the corpus: sample through
the index.

## Bands

The 14 daily bands in their `ERA5L_DAY_10` order (`d2m e pev ro sp ssr ssrd
str swvl1 swvl2 t2m tp u10 v10`), then `sf strd sshf evavt evabs sro swvl3
swvl4 skt stl1 stl4 sd smlt`.

ERA5-Land's archive has three evaporation components swapped (ECMWF known
issue, checked on this store: in the Sahara its `evabs` field is zero and
`evaow` equals total evaporation). Bands are stored under the quantity they
hold: `evavt` (transpiration) is read from the store's `evabs` field and
`evabs` (bare-soil evaporation) from its `evaow` field.

## Windows without data

A window is stored when its centre's ERA5-Land cell has data. It is not
(`status`) when the cell is south of the store (`south_of_store`, every
Antarctic window) or is sea in ERA5-Land (`not_land_in_era5_land`: coasts and
small islands, since the corpus took land from Natural Earth). There is no
fallback to a neighbouring cell.

## Stages

| Stage | Module | Output |
|---|---|---|
| 1 | `plan` | `index.parquet`, `shards.json`, `bands.json`, `README.md` |
| 2 | `shards --shard N` | `shards/shard_NNN.npy`, `progress/shard_NNN/` |
| 3 | `static` | `static/` |
| 4 | `validate --shard N` | `progress/shard_NNN.validate.json` |

`scripts/era5_supervised/v0/era5_hourly/launch_build.sh` runs them in a
CPU-only Beaker job (one job per shard; `plan` once, before the others) on a
jupiter, neptune or saturn node: one job per node, at most 64 workers, at
most 50 jobs at once. Every stage resumes or skips finished work. The token
comes from the workspace secret `RSLEARN_EARTHDATAHUB_TOKEN` as
`EARTHDATAHUB_TOKEN`.
