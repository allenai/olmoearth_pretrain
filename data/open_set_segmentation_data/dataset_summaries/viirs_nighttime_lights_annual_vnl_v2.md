# VIIRS Nighttime Lights (Annual VNL V2)

- **Slug**: `viirs_nighttime_lights_annual_vnl_v2`
- **Task type**: regression (per-pixel continuous night-time radiance)
- **Status**: completed — **5000** samples
- **Family**: nightlights · **Region**: Global · **License**: CC-BY-4.0 (public domain per CSM)

## Source

**Annual VIIRS Nighttime Lights V2** — Earth Observation Group (EOG), Payne Institute for
Public Policy, Colorado School of Mines (https://eogdata.mines.edu/products/vnl/). Global
annual cloud-free composites of Day/Night Band radiance at ~15 arc-second (**~500 m**)
native resolution, in **nW/cm²/sr**. It is the standard remote-sensing proxy for human
settlement / electrification / economic activity.

### Access decision (why not eogdata.mines.edu directly)

The canonical EOG distribution now sits behind a **Keycloak/SSO login gate** (every
`/nighttime_light/annual/...` URL 302-redirects to `eogauth.mines.edu`). There is **no EOG
credential** in `.env`, and the authorized GEE service-account path
referenced by `TEST_GEE_SERVICE_ACCOUNT_CREDENTIALS` (`/etc/credentials/gee_key.json`) **does
not exist** on this host, so the `NOAA/VIIRS/DNB/ANNUAL_V2x` Earth Engine assets were also
unreachable. Rather than reject on a credential gate, the **identical VNL V2 product** was
sourced from a public, ungated CC-BY-4.0 mirror on the Hugging Face Hub:

> **`Major-TOM/Core-VIIRS-Nighttime-Light`**
> (2016–2021 = Annual VNL **V2.1**, 2022–2024 = Annual VNL **V2.2**; band = annual **median**)

Other public mirrors were considered and rejected: AWS `s3://globalnightlight/` ("Light
Every Night") holds only raw nightly/monthly VIIRS through 2020, not the finished annual
masked composites; CREODIAS `s3://eodata/.../nightlights_average_viirs_v21/` needs CDSE S3
keys we don't hold.

The Major TOM mirror is well-suited to our contract: the global product is pre-diced into
~1056×1056 single-band **float32 GeoTIFF** patches, **each already in a local UTM zone at
exactly 10 m/pixel** (one patch per Major TOM grid cell, evenly distributed worldwide), and
each patch is byte-addressable within its year shard via an `(offset, size)` range in
`INDEX.parquet`.

### Download mechanics

Anonymous per-patch HTTP **Range** reads against the Hub are aggressively rate-limited
(HTTP 429). So the script instead downloads a **curated subset of the 16 year shards** for
2020 that together span every inhabited continent (shards `001,005,006,007,008,013,015`
≈ **31 GB**) via `huggingface_hub` (CDN + automatic 429 backoff), then reads each sampled
patch by **seeking to its INDEX byte offset in the local shard** (the stored `.tif` is
uncompressed, so a `seek()+read()` yields the exact tif bytes). All measure/write work is
therefore local and fast; only the shard pulls touch the network.

## Label → regression target

- **Quantity** (`metadata.regression.name`): `nighttime_radiance`
- **Unit**: nW/cm²/sr · **dtype**: float32 · **nodata**: −99999 (`io.REGRESSION_NODATA`)
- **Year**: **2020** (post-2016, Annual VNL V2.1), one composite year.
- Radiance is an *intensity* (resolution-invariant), so it is stored **as-is** — no unit
  conversion or normalization. Non-finite pixels → nodata; all observed values are ≥ 0.
- **Observed per-pixel value range across tiles: [0.052, 635.9] nW/cm²/sr.** Dark areas sit
  at the sensor **noise floor (~0.1–0.4)**, not exactly zero, because this is the (un-masked)
  **median** product.

## ⚠️ Resolution caveat (important)

VNL is **natively ~500 m**. The Major TOM mirror already resampled it onto a 10 m grid, so a
64×64 (**640 m**) tile carries only **~1–2 native VIIRS pixels** of real information — a
smooth/upsampled field, essentially near-constant within a tile. This is an **intentionally
coarse regression probe** (a settlement/economic-activity proxy), **not** a 10 m-native
signal. Recorded in `metadata.regression` as `native_resolution_m: 500`.

## Tiling & sampling

- **One center 64×64 window per sampled Major TOM cell**, reusing the mirror's local-UTM
  10 m grid directly (**no reprojection** — georeferencing round-trips exactly).
- **Candidate pool (11,500)**, stratified so the value range is well populated:
  - `bright` (4000): highest-`socio:population` land cells, capped ≤200/country → guarantees
    the bright urban tail across many countries.
  - `broad` (6000): land cells stratified across `socio:human_modification` deciles →
    dark-rural → peri-urban spread.
  - `ocean` (1500): ocean/lake cells → genuine near-zero noise floor.
- Radiance measured per candidate (window mean), then **bucket-balanced across
  log10(radiance+1) deciles** to **TOTAL = 5000** (`sampling.bucket_balance_regression`).
- **Time range** = the composite year `[2020-01-01, 2021-01-01)` (≤ 1 year). **`change_time`
  = null** (static annual label).

### Resulting distribution (5000 tiles)

Selected-tile mean-radiance histogram (nW/cm²/sr): `[0,0.5): 4090 · [0.5,1): 454 ·
[1,2): 156 · [2,5): 97 · [5,10): 59 · [10,25): 86 · [25,50): 40 · [50,100): 14 ·
[100,500): 4`. The distribution is dark-dominated — which is realistic (Earth is mostly dark
at night) — with the full bright tail present. Not oversampled toward cities.

Rough continental spread: Africa/MidEast 2051 · Asia 1271 · S.America 786 · N.America 551 ·
Europe 298 · Oceania/other 43 · (Ocean cells 655). Top countries: Brazil 370, India 221,
China 205, USA 178, DR Congo 156, Pakistan 119, Ethiopia 110, Mexico 109, Chad 106,
Nigeria 100. lon −179.7…178.8, lat −64.8…71.4.

## Verification (spec §9)

- 5000 `.tif` + 5000 `.json`, all paired; each tile single-band **float32**, local **UTM**
  CRS at **10 m**, **64×64**, nodata **−99999**; values within the declared range.
- Every sample JSON has a ≤1-year `time_range` and `change_time=null`.
- **Semantic sanity**: brightest tiles land on real cities/industry — Chicago (140.8),
  Kuwait (131.3), Pearl River Delta/Hong Kong (106.1), Zaragoza ES (100.9), Coatzacoalcos MX
  gas flaring (93.8) — while the darkest sit over remote interior New Guinea and Amazon
  (~0.1). Radiance correctly co-locates with settlement.
- Re-running the script is idempotent (skips existing `.tif`; a re-run wrote 0 tiles).

## Outputs

- Weka: `datasets/viirs_nighttime_lights_annual_vnl_v2/{metadata.json, locations/{000000..004999}.{tif,json}}`
- Weka raw: `raw/viirs_nighttime_lights_annual_vnl_v2/{INDEX.parquet, 2020/MAJORTOM-VIIRS-NTL_2020_median_{001,005,006,007,008,013,015}.zip}`
- Repo: `olmoearth_pretrain/open_set_segmentation_data/datasets/viirs_nighttime_lights_annual_vnl_v2.py`

## Reproduce

```bash
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.viirs_nighttime_lights_annual_vnl_v2 --workers 64
# --skip-download to reuse already-downloaded INDEX.parquet + shards
```

## Caveats / notes

- **~500 m → 10 m resolution mismatch** (see above): the label is a coarse upsampled field.
- The **median (un-masked)** product retains a low background radiance floor (~0.1–0.4);
  there is no hard zero for "unlit". This is fine for a continuous regression target.
- Sampling covers 7 of 16 year shards (all inhabited continents); it is a bounded global
  sample, not exhaustive global coverage (spec §5, large derived-product rasters).
- Manifest listed classes ("unlit/dim/lit/bright") as a *classification* framing; we instead
  keep the native continuous radiance as **regression**, which preserves more signal and
  matches the SOP's regression recipe. Buckets are recorded in `metadata.regression.buckets`
  for anyone who wants to discretize downstream.
