# Every-Capture ("allcap") Pretraining Corpus

An OlmoEarth pretraining corpus built from the **same 100K-window subset of the v1
`osm_sampling` locations and timestamps** as the perceiver-experiment corpus, but with
**every individual Sentinel-2 L2A, Sentinel-1 RTC and Landsat capture** over a 360-day
span per window instead of 12 monthly least-cloudy mosaics, and **no cloud filtering**
(cloud information is recorded as data, not used to drop scenes).

Branch: `joer/corpus-allcaptures` (this repo). Built 2026-08-18 → 2026-09-25.

## Where it is

| stage | path |
|---|---|
| rslearn dataset (windows, `items.json`, materialized layers, manifests, scan logs) | `/weka/dfive-default/helios/dataset_creation/osm_allcaptures` |
| OlmoEarth GeoTIFFs + consolidated CSVs (13 TB) | `/weka/dfive-default/helios/dataset/osm_allcaptures` |
| **training h5 (395,660 samples)** | `/weka/dfive-default/helios/dataset/osm_allcaptures/h5py_data_w_missing_timesteps_zstd_3_128_x_4/cdl_landsat_l2_openstreetmap_raster_sentinel1_sentinel2_l2a_sentinel2_scl_srtm_worldcereal_worldcover_wri_canopy_height_map/395660` |
| validated 1K pilot (same layout, 4,000 samples) | `/weka/dfive-default/helios/dataset/osm_allcaptures_pilot1k/h5py_data_w_missing_timesteps_zstd_3_128_x_4/.../4000` |

`/weka/dfive-default/olmoearth_pretrain` is a symlink to `helios`, so either prefix works.

## What is in it

- **Windows**: 100,000 windows (256×256 px at 10 m, 2.56 km), a seeded subset of the
  285,288 v1 `osm_sampling` windows, copied verbatim (same names, grid geometry and
  14-day window time range). Manifest: `selected_windows.json` in the rslearn dir. The
  first 1,000 (`osm_allcaptures_pilot1k`) were the validation pilot.
- **Time span**: 360 days, −180 d … +180 d around the window center time (same as v1).
- **Every-capture modalities** (rslearn `space_mode: CONTAINS`, i.e. only scenes whose
  data footprint fully covers the window; chronological; no cloud sort/filter):
  - `sentinel2_l2a` — 12 bands (B02,B03,B04,B08 @10 m; B05,B06,B07,B8A,B11,B12 @20 m;
    B01,B09 @40 m), harmonized, from Planetary Computer `sentinel-2-l2a`.
    Typical 60–140 captures/window (median ≈ 72).
  - `sentinel2_scl` — the L2A scene-classification band (20 m), stored as its own
    modality sharing the S2 timestamps.
  - `sentinel1` — VV/VH RTC (float32, dB in h5) from PC `sentinel-1-rtc`; captures
    containing nodata are dropped at conversion. Typical 20–120 captures/window.
  - `landsat_l2` — Landsat 8/9 Collection-2 Level-2 bands B1–B7 + B10 at 20 m from PC
    `landsat-c2-l2` (no pan/cirrus/B11 — those are Level-1 only). Typical 20–90
    captures/window.
- **Static map layers** (as in v1): `worldcover`, `srtm`, `openstreetmap_raster`,
  `worldcereal`, `wri_canopy_height_map`, `cdl` (CONUS only).
- **Same-pass duplicates removed**: adjacent S2 MGRS tiles (same instant) and adjacent
  Landsat rows (~24 s apart) both fully cover small windows; captures within 120 s of
  the previously kept one are dropped (`dedup_allcap_items.py`). This removed ~46% of S2
  and ~19% of Landsat item groups.
- **Cloudiness**: no filtering. Per-subtile SCL cloud fraction is stored; on the pilot
  the median capture was 56–69% cloudy and ~51–54% of captures were majority-cloud.

### Completeness

| layer | windows complete | genuinely no data | missing (unrecoverable) |
|---|---|---|---|
| Sentinel-2 | 99,865 | 96 | 39 |
| Sentinel-1 | 97,785 | 2,181 | 34 |
| Landsat | ~97,600 | 348 | ~2,000 |

98,915 windows are complete in all three; those are the ones in the h5 (× 4 subtiles
= 395,660 samples). The Landsat gap is mostly corrupt blobs in the Planetary Computer
`landsat-c2-l2` mirror (truncated ~14 KB files that return HTTP 200) plus a few 404s
that could not be attributed to a scene from logs.

## h5 format (differences from the v1 corpus)

Same directory convention (`h5py_data_*_zstd_3_128_x_4/<sorted modalities>/<N>`,
one `sample_{i}.h5` per 128×128 subtile, `sample_metadata.csv`,
`latlon_distribution.npy`, `compression_settings.json`) — the training loader keys
off this path. Per file:

- Multitemporal datasets are `(128, 128, T_mod, bands)` with **T varying per sample
  and per modality**; there is no shared 12-slot grid and no `missing_timesteps_masks`.
- Per-modality timestamps: `timestamps_sentinel2_l2a`, `timestamps_sentinel2_scl`,
  `timestamps_sentinel1`, `timestamps_landsat_l2` — each `(T, 3)` int `[day, month-1,
  year]` like v1's `timestamps`, in chronological order.
- `sentinel2_scl_cloud_fraction` — `(T_scl,)` float32 fraction of subtile pixels in
  SCL classes {3, 8, 9, 10} (cloud shadow, medium/high cloud, cirrus).
- `latlon` as before. Static modalities as in v1.
- `sample_metadata.csv` marks per-sample modality presence; OSM raster is absent for
  windows whose OSM tile is empty (~77%), `cdl` outside CONUS, etc.

The **training-side loader has not been adapted yet**: `MAX_SEQUENCE_LENGTH=12`,
`_get_max_t_within_token_budget`, `get_valid_start_ts` and the masking code in
`olmoearth_pretrain/data/dataset.py` assume 12 aligned monthly steps. Irregular,
per-modality timelines need loader/collation changes before this corpus is trainable.
Extra files in the h5 dir (`sample_manifest.json`, `samples.pkl`, `progress/`) are
build artifacts and can be ignored or deleted.

## How it was built

Config: `olmoearth_pretrain/dataset_creation/rslearn_configs/corpus_allcap.json`
(`max_matches` 400/200/150 for S2/S1/Landsat, `sort_by: datetime`).

```
# 1. windows (copied from v1), config, source_data symlinks
python -m olmoearth_pretrain.dataset_creation.create_windows.from_existing_windows \
    --src_ds_path /weka/.../dataset_creation/osm_sampling --ds_path $DS --group res_10 \
    --num_windows 100000 --seed 0 --config-path .../corpus_allcap.json
# 2. prepare -> dedup -> ingest -> materialize, sharded on Beaker (CPU-only jobs)
python scripts/data/corpus_pipeline.py launch-rslearn --window-manifest $DS/selected_windows.json \
    --group res_10 --rslearn-dir $DS --rslearn-config $DS/config.json --num-shards N \
    --workers 32 --jobs-per-process 8 --steps prepare dedup ingest materialize \
    --clusters ai2/neptune-cirrascale --gpus 0 --priority urgent
# 3. fixup loop until convergence: workers tee corrupt-scene errors to
#    $DS/progress/badscenes/shard_N.log; prune them; re-materialize incomplete windows
python -m olmoearth_pretrain.dataset_creation.prune_bad_scenes --ds_path $DS --group res_10 \
    --layers landsat,sentinel2_l2a,sentinel1 --from-log <concatenated badscenes logs>
# 4. convert (per-window GeoTIFF stacks + per-window CSVs), sharded
python scripts/data/corpus_pipeline.py launch-convert --window-manifest <complete windows> \
    --group res_10 --allcap --rslearn-dir $DS --num-shards 15 --workers 32 ...
# 5. once, centrally
python -m olmoearth_pretrain.dataset_creation.pipeline --ds_path $DS --olmoearth_path $OLM \
    --groups res_10 --allcap --workers 32 --only metadata
python -m olmoearth_pretrain.dataset_creation.pipeline ... --only rasterize_osm --workers 96
# 6. h5: distributed bad-modality scan, then prepare + sharded writers
python scripts/data/corpus_pipeline.py launch-scan --olmoearth-dir $OLM --allcap \
    --h5-tile-size 128 --scan-modalities sentinel1 openstreetmap_raster --num-shards 15 ...
python scripts/data/corpus_pipeline.py prepare-h5 --olmoearth-dir $OLM --allcap \
    --h5-tile-size 128 --bad-modalities-dir $OLM/h5_scan
python scripts/data/corpus_pipeline.py launch-h5 --h5py-dir <h5 dir> --num-h5-shards 40 ...
# 7. validate
python -m scripts.data.validate_allcap --olmoearth_path $OLM --h5_dir <h5 dir> --num_h5 300
```

Run pattern on the dev box: `PYTHONPATH=/root/repos/oe_pretrain_corpus_v2` with
`/root/dev/.venv/bin/python`; Beaker jobs use `uv.lock` (rslearn 0.1.11).

## Operational lessons (so the next run is one pass, not eight)

- **rslearn commits a layer all-or-nothing**, and every-capture layers have 40–140
  scenes, so one unreadable scene fails the whole layer for that window on every retry.
  The materialize → prune → re-materialize loop exists only because of this; the
  right fix is an rslearn option to skip unreadable items instead of failing the layer.
- Materialize is **network-bound** (Azure West-Europe blobs, ~490 sequential reads per
  window), ~2 windows/h/worker. 256 workers per IP is the safe ceiling; 768 caused
  connection-level throttling of the S2 blob host (no 429s, just timeouts).
- Long-lived rslearn workers **leak file handles**; use `--jobs-per-process 8`
  (`maxtasksperchild`) or nodes hit system-wide fd exhaustion after ~17 h.
- Use **CPU-only Beaker jobs without a `cpuCount` reservation** (`--gpus 0`, no
  `--cpus`); reserved-CPU jobs are counted against slot-linked capacity and sit
  pending when Neptune is busy, and 8-GPU whole-node jobs may never schedule.
  `--workers 32` bounds the real footprint.
- Never rely on downloading Beaker job logs at scale — hours per log. Anything a
  later step needs (bad scenes, progress) is written to Weka by the worker.
- Do not run per-shard metadata consolidation (`--finalize-metadata`); run it once.
- Beaker `experiment.create` occasionally 409s on client retries; launch shards one at
  a time with existence checks, and prefer a single launcher invocation when possible.

## Loose ends

- Delete the rslearn intermediates (`$DS/windows/*/layers`, ~23 TB) once validation
  passes; keep `items.json`/`metadata.json` (small) for provenance.
- ~1% of windows lack a layer (see Completeness); acceptable for pretraining, could be
  recovered by re-fetching Landsat from AWS instead of Planetary Computer.
- Training loader work for variable-T inputs (above).
