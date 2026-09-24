# HP-LSP (HLS-PhenoCam Land Surface Phenology)

- **slug:** `hp_lsp_hls_phenocam_land_surface_phenology`
- **manifest name:** HP-LSP (HLS-PhenoCam Land Surface Phenology)
- **status:** **completed**
- **task type:** regression (greenup onset day-of-year)
- **num_samples:** 5000
- **source:** ORNL DAAC, DOI [10.3334/ORNLDAAC/2248](https://doi.org/10.3334/ORNLDAAC/2248) —
  "Phenology derived from Satellite Data and PhenoCam across CONUS and Alaska, 2019-2020"
- **CMR collection:** `C2775078742-ORNL_CLOUD` (short name `Landsat8_Sentinel2_Phenocam_2248`)
- **license:** open (EOSDIS / ORNL DAAC)

## What the dataset is

30 m land-surface-phenology (LSP) product fusing Harmonized Landsat-Sentinel (HLS) EVI2
time series with PhenoCam ground-camera observations. For each of 78 PhenoCam sites and
growing seasons 2019/2020, per-pixel phenological transition dates were derived over the
site's HLS/MGRS (UTM) tile (transition-date accuracy <= ~5 days per the source guide).

Files (CMR `GET DATA` URLs, Earthdata-protected path): per site-year there is a
`..._LSP_Date.tif` (12-band Int16 transition dates) and a `..._LSP_EVI2.tif` (122-band
EVI2, ancillary — not used). Filename pattern
`HLS_PhenoCam_A{YEAR}_{SITE}_T{MGRS}_LSP_Date.tif`. We use the **156 `LSP_Date`** files.

**Georeferencing (§8.2): passes** — real georeferenced GeoTIFFs on MGRS/UTM tiles (30 m,
native CRS = the tile's UTM zone), directly placeable on the S2 grid.

### The 12 `LSP_Date` bands and the cycle-2 finding (key judgment call)

The 12 bands are 4 transition types x up to 3 growing cycles:
`1-3 Greenup`, `4-6 Maturity`, `7-9 Senescence`, `10-12 Dormancy` (onset), cycles 1/2/3.
Values are day-of-year; fill = 32767.

Empirically over all 156 files, **cycle 2 is the primary annual growth cycle, not cycle
1.** Greenup **cycle 2 (band 2)** has ~92% valid coverage with a physically-correct
seasonal progression (greenup DOY ~102 -> maturity ~176 -> senescence ~252 -> dormancy
~314). Cycle 1 (~3% valid, mostly negative DOY) and cycle 3 (~4% valid, DOY ~300+) are
sparse early/late partial cycles that straddle the calendar boundary. So the canonical
start-of-season is **band 2**, and that is the regression target used here
(`greenup_onset_doy`). (Using band 1, "Greenup Cycle 1", would have been wrong — it is
almost all fill and negative; this was caught and corrected during processing.)

## Processing

- **Task:** regression. Target = greenup onset DOY, cycle 2 (band 2 of `LSP_Date`).
  Maturity/senescence/dormancy onset (bands 4-12) and EVI2 are available in the source and
  could be shipped as separate regression datasets later; only greenup onset is emitted
  here (per the task instruction).
- **Access:** files are Earthdata-protected. Credentials from
  `.env` (`NASA_EARTHDATA_USERNAME` / `NASA_EARTHDATA_PASSWORD`,
  user-authorized) were written to `~/.netrc` (`machine urs.earthdata.nasa.gov`, chmod
  600). Download via the new shared `download.download_earthdata` (requests + netrc,
  follows the URS OAuth redirect). Granule URLs enumerated from CMR (no auth). 156 files,
  ~150 MB total.
- **Reprojection:** native 30 m UTM -> local UTM 10 m via
  `GeotiffRasterFormat().decode_raster(..., resampling=Resampling.nearest)`. **Nearest**
  (not bilinear): the int16 32767 fill is a hard sentinel that bilinear would smear into
  real DOY. At 30 m->10 m nearest simply replicates each source pixel exactly.
- **Tiles:** 64x64 (~640 m) single-band float32; 32767 (and out-of-range) -> nodata
  `-99999` (`io.REGRESSION_NODATA`).
- **Sampling (§5):** bounded-tile sampling across all 156 site-year tiles. Candidate tile
  centers gathered on a ~21-px (one-tile) decimated grid from band 2 (valid pixels only) —
  32,224 candidates — then `bucket_balance_regression` across greenup-DOY deciles to the
  regression cap of **5000** (the distribution is peaked around spring greenup, so bucket
  balancing ensures early- and late-season pixels both appear).
- **Time (§5):** greenup onset is an **annual per-pixel value, not a dated change event**,
  so `change_time = null` and `time_range` = the labeled calendar year (2019 or 2020) via
  `io.year_range`. Selected samples: 2562 in 2019, 2438 in 2020.

## Output stats

- 5000 GeoTIFFs + 5000 sample JSONs in
  `datasets/hp_lsp_hls_phenocam_land_surface_phenology/locations/`.
- Single-band float32, local UTM (many zones, e.g. 32613/32615/32617), 10 m/pixel, 64x64,
  nodata -99999.
- Greenup-DOY value range across tiles: [1, 365]; bucket edges (deciles)
  [1, 64, 77, 87, 94, 102, 115, 128, 137, 155, 365]; per-sample percentiles p5~40,
  p50~102, p95~180.

## Verification (§9)

- Opened multiple tifs: all single-band float32, UTM at 10 m, 64x64, nodata -99999, values
  in valid DOY range with high valid fractions.
- All 5000 tifs have a matching JSON; `time_range` = 1 year; `change_time` null;
  `metadata.json` `regression.value_range` covers observed values.
- Georeferencing round-trip: tile-center values match the source `LSP_Date` band 2 at the
  tile's lon/lat (exact for most; where a tile center landed in a high-gradient wetland the
  value is still one of the immediate 3x3 source neighbors — confirms correct registration,
  nearest-resampled). Latitudes/regions sensible (e.g. Florida sites early greenup DOY
  ~16-120, Virginia later); higher-latitude/later greenup as expected.
- Idempotent: `_write_one` skips any existing `{sample_id}.tif`; selection is seeded and
  order-independent (`sampling._stable_order`).

## Reproduce

```
# ensure ~/.netrc has: machine urs.earthdata.nasa.gov login <user> password <pass>  (chmod 600)
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.\
hp_lsp_hls_phenocam_land_surface_phenology                 # download + process
# or, if raw/ already populated:
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.\
hp_lsp_hls_phenocam_land_surface_phenology --skip-download
```

## Caveats

- Only greenup onset (cycle 2) is shipped; maturity/senescence/dormancy onset are left for
  potential future datasets.
- Cycle-1/cycle-3-only pixels (sparse, ~3-4%) are nodata in the label.
- `time_range` is the labeled calendar year; a small number of cycle-2 greenup pixels near
  DOY 1 effectively started at the very start of the year — acceptable for a 1-year pairing
  window.
- Reusable addition: `download.download_earthdata` was added to the shared module for
  Earthdata/URS-protected DAAC downloads.
