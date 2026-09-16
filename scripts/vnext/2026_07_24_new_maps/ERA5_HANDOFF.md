# ERA5 climate-awareness experiments — handoff

_Last updated: 2026-09-15 evening. Written for a fresh agent picking this up._

## Goal (why this exists)

Make the pretrained embeddings **climate/weather aware** by bringing in the
`era5_10` modality (ERA5 monthly reanalysis: 12 months × 6 vars, non-spatial 1×1
"pixel" per scene at ~9 km). Two arms explore two hypotheses:

1. **`era5in`** — ERA5 as an **encoder input / conditioning** signal (not supervised).
2. **`era5clim`** — ERA5 as a **decode-only supervision target**: the model must
   predict the per-scene **12-month climate normal** (the temporal PATTERN, 72-dim)
   from the mean-pooled register grid. This is the "climate-aware task".

The guardrail throughout: this must **not** homogenize the registers and hurt the
dense spatial probes (PASTIS ps=1). Keep the ERA5 supervision weight low and always
read climate-zone metrics next to PASTIS.

There was an earlier discussion; key decisions:
- **Signature = temporal pattern**, NOT annual mean. Seasonality (Mediterranean vs
  monsoon vs continental) is what separates climate zones. So both the supervision
  target and the eval clustering use the flattened `[12 months × 6 vars] = 72`-dim
  vector in calendar/time-major order (`[m0_c0..m0_c5, m1_c0..]`). No observation-
  timestamp alignment needed because month index == calendar month.

---

## Current state (what's DONE and validated)

### Code changes (all lint-clean)

1. **`olmoearth_pretrain/nn/supervision_head.py`** — added a `temporal_reduction`
   knob on `SupervisionModalityConfig` for **non-spatial multitemporal** regression
   targets. It maps a `[B, T, C]` target onto the non-spatial head's flat
   `[B, num_output_channels]` output:
   - `"mean"` → `[B, C]` (annual level), via `_reduce_time_mean` (missing-aware).
   - `"flatten"` → `[B, T*C]` (full monthly pattern), via `_flatten_time`
     (`rearrange("b t c -> b (t c)")`, time-major). **This is what era5clim uses.**
   - `None` → target untouched.
   Validation in `__post_init__` allows only `{"mean", "flatten", None}`, regression-
   only, mutually exclusive with `time_conditioned`. Docstring updated.

2. **`scripts/vnext/2026_07_24_new_maps/regbtl_v1_2_gdyn_d768_wideread_regsup_w1_newsampling_psuniform_landsat_refl_era5clim.py`**
   (the **pooled climate-prediction** arm):
   - Adds `era5_10` to `training_modalities`, forces it **decode-only** (masking
     `only_decode_modalities` in both dataloader + train-module configs, and
     `mask_negatives_for_modalities` in the loss config).
   - Adds an `era5_10` regression supervision head: `num_output_channels=72`
     (`ERA5_SIGNATURE_DIM = 12*6`), `regression_loss_type="l1"`,
     `temporal_reduction="flatten"`, `weight=ERA5_SUPERVISION_WEIGHT=0.1`.

3. **`scripts/vnext/2026_07_24_new_maps/regbtl_v1_2_gdyn_d768_wideread_regsup_w1_newsampling_psuniform_landsat_refl_era5in.py`**
   (the **input-conditioning** arm):
   - Adds `era5_10` to `training_modalities` but does **NOT** make it decode-only and
     does **NOT** supervise it → it is an encoded input.

4. **`scripts/tools/era5_climate_zone_eval.py`** — standalone climate-zone eval:
   - `build-zones`: sample `era5_10` from the new h5, reduce each scene to its 72-dim
     climate-normal signature (`_era5_signature`), standardize all 72 dims, KMeans → K
     zones, write NPZ (`indices, zones, signatures, band_mean, band_std, centers, bands`).
   - `score`: given an embeddings NPZ (`indices`, `embeddings`), report NMI/ARI
     (KMeans on embeddings vs zones), a logistic-regression linear-probe acc + macro-F1,
     and same-zone vs diff-zone cosine gap. Supports `--koppen-npz` for real labels.

### Tests

Persistent pytest coverage lives in
`tests/unit/nn/test_supervision_head.py::TestTemporalReduction` — covers the
`flatten`/`mean` helpers (shape, time-major order, missing-aware), config validation,
the non-spatial head output width (72), and end-to-end pooled-register → era5 pred with
loss backprop into the register grid (both `flatten` and `mean`, plus missing-month
drop). It mirrors the existing `TestTimeConditionedSupervision` end-to-end pattern.

- `mean` path: full end-to-end unit test **passed earlier** (config build, non-spatial
  pooled head, L1 regression, weighted backprop, missing-timestep masking).
- `flatten` reshape (the only new logic): verified time-major ordering + shape
  `[B,12,6]→[B,72]` + missing-month drop (dependency-free check, passed).
- The new `TestTemporalReduction` pytest was **lint-clean but NOT yet executed** —
  torch/olmoearth imports off weka were being starved by the running conversion job.
  Run it once I/O frees up (see TODO 3).

---

## The h5 dataset build (IN PROGRESS)

**Command** (running as nohup, pid was `23927`, cwd `/weka/dfive-default/yawenz/olmoearth_pretrain`):
```
nohup .venv/bin/python -m olmoearth_pretrain.internal.run_h5_conversion \
  --tile-path=/weka/dfive-default/helios/dataset/osm_sampling_landsat_refl \
  "--supported-modality-names=[sentinel2_l2a,sentinel1,landsat,worldcover,glo30,openstreetmap_raster,meta_canopy_height,cdl,worldcereal,era5_10]" \
  --compression=zstd --compression_opts=3 --tile_size=128 \
  > era5_h5_conversion.log 2>&1 &
```
Log: `/weka/dfive-default/yawenz/olmoearth_pretrain/era5_h5_conversion.log`

**Progress at last check (~8:38pm):** ~72% of the **filtering pass**
(`206572/285288`), ETA ~1.5h for that pass. Then it writes the `sample_*.h5` files
(the "Creating H5 files" phase), which is additional time.

**Expected output dir** (NOT created until `set_h5py_dir` fires, i.e. after filtering):
```
/weka/dfive-default/helios/dataset/osm_sampling_landsat_refl/
  h5py_data_w_missing_timesteps_zstd_3_128_x_4/
  cdl_era5_10_glo30_landsat_meta_canopy_height_openstreetmap_raster_sentinel1_sentinel2_l2a_worldcereal_worldcover/
  <N>
```

### ⚠️ IMPORTANT — the trailing `<N>`
- `<N> = filtered_scenes × num_subtiles`, and `num_subtiles = (256//128)² = 4`.
- `285288` is **scenes before filtering**; `1138828` (the sibling dataset's number)
  is **h5 files = filtered_scenes × 4**. They count different things — there is NO
  real "drop". Expect `<N> ≈ 1.13–1.14M` (very close to the existing `1138828`).
- **Both arm scripts currently HARDCODE `.../1138828`.** Once the log prints
  `Setting h5py_dir to ...`, grab the real `<N>` and fix the `ERA5_H5_DIR` constant in
  BOTH scripts if it differs (it may end up exactly `1138828`, but confirm):
  - `...era5in.py` line ~68
  - `...era5clim.py` line ~91

Norm stats already exist: `olmoearth_pretrain/data/norm_configs/computed_landsat_reflectance.json`
has an `era5_10` entry (6 bands), so no recompute needed.

---

## TODO (in order) for the next agent

1. **Confirm the conversion finished cleanly.** Check the process is gone and the log
   ends with the H5-creation phase completing (not a traceback):
   ```
   grep -iE "Setting h5py_dir|Creating H5|Traceback|Error" era5_h5_conversion.log | tail
   ```
   Grab the real `Setting h5py_dir to <path>/<N>`.

2. **Patch `<N>`** in `ERA5_H5_DIR` in both era5 scripts if it differs from `1138828`.
   Also sanity-check the h5 dir has the expected file count and that a `sample_*.h5`
   actually contains an `era5_10` dataset of shape reducible to `(12, 6)`:
   ```python
   import h5py, glob
   p = sorted(glob.glob(f"{H5DIR}/sample_*.h5"))[0]
   f = h5py.File(p); print(f["era5_10"].shape)  # expect something folding to (12,6)
   ```

3. **Run the persisted unit tests** now that I/O is free:
   ```
   .venv/bin/python -m pytest tests/unit/nn/test_supervision_head.py::TestTemporalReduction -q
   ```
   (Also good to run the whole `test_supervision_head.py` to confirm no regressions.)

4. **CPU config-build smoke test of BOTH arms** (no training) to catch wiring issues
   before GPUs. Use the project venv `.venv/bin/python`. Something like building the
   experiment configs via each script's builders and asserting the model +
   dataloader + train-module configs construct. Watch for:
   - era5clim: `era5_10` present in `supervision_head_config.modality_configs` with 72
     outputs + `temporal_reduction="flatten"`; `era5_10` in `only_decode_modalities`
     (both dataloader and train module) and in `mask_negatives_for_modalities`.
   - era5in: `era5_10` in `training_modalities`, NOT decode-only, NOT supervised.

5. **Launch the two arms on GPUs** (plus the existing no-ERA5 baseline
   `regbtl_v1_2_gdyn_d768_wideread_regsup_w1_newsampling_psuniform_landsat_refl` as the
   control). Keep everything else identical so the comparison is clean.

6. **Evaluate climate-awareness** once there are checkpoints:
   - `build-zones` once from the new h5 (pick K, e.g. 16).
   - Produce pooled scene-level embeddings NPZ (`indices`,`embeddings`) for baseline +
     both arms with whatever inference harness the repo uses.
   - `score` each and compare NMI/ARI, linear-probe acc/F1, cos gap.
   - **ALWAYS** read these next to PASTIS ps=1 / crop evals — the whole point is the
     climate-awareness vs spatial-detail trade-off. If era5clim tanks PASTIS, lower
     `ERA5_SUPERVISION_WEIGHT` (it's the knob) and/or prefer the era5in arm.

## Open questions / things to watch
- **Homogenization risk (era5clim):** if the low weight (0.1) still hurts PASTIS ps=1,
  sweep the weight down. The pooled-then-predict design already limits per-cell pull.
- **ERA5 NaN drops:** ~14% of scenes lose `era5_10` over ocean/edge (NaN → converter
  removes the modality). Fine for a decode-only target and for an optional input
  (missing modalities are handled), but note the effective supervised subset is smaller.
- **T=12 assumption:** `flatten` requires exactly 12 months. The converter drops
  `era5_10` entirely on any NaN, so stored ERA5 is a complete 12-month stack. If a
  future data change breaks that, `flatten` will shape-mismatch the 72-dim head — the
  loss will error loudly rather than silently mis-train.

## Key file map
- Supervision head + temporal_reduction: `olmoearth_pretrain/nn/supervision_head.py`
  (`_flatten_time`, `_reduce_time_mean`, `SupervisionModalityConfig.temporal_reduction`)
- Arms: `scripts/vnext/2026_07_24_new_maps/..._era5in.py`, `..._era5clim.py`
- Baseline (control): `scripts/vnext/2026_07_24_new_maps/regbtl_v1_2_gdyn_d768_wideread_regsup_w1_newsampling_psuniform_landsat_refl.py`
- Climate-zone eval: `scripts/tools/era5_climate_zone_eval.py`
- H5 converter: `olmoearth_pretrain/dataset/convert_to_h5py.py`;
  runner: `olmoearth_pretrain/internal/run_h5_conversion.py`
- ERA5 modality spec: `olmoearth_pretrain/data/constants.py` (`ERA5_10`, lines ~442-462)
- Norm stats: `olmoearth_pretrain/data/norm_configs/computed_landsat_reflectance.json`
- Conversion log: `era5_h5_conversion.log` (repo root)
