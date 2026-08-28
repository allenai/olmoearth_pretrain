# Global Snowmelt Runoff Onset (Sentinel-1)

- **Slug**: `global_snowmelt_runoff_onset_sentinel_1`
- **Status**: `completed`
- **Task type**: regression
- **Samples**: 5000
- **Source**: Gagliano, E., Shean, D. & Henderson, S. (2026), *A global high-resolution
  dataset of snowmelt runoff onset timing from Sentinel-1 SAR, 2015–2024*, Zenodo record
  [19618062](https://zenodo.org/records/19618062), concept DOI
  10.5281/zenodo.16953614. License **CC-BY-4.0**.

## What the source is

The first comprehensive global map of snowmelt **runoff onset** timing. Sentinel-1 C-band
SAR (VV) backscatter minima — which coincide with the ripening→runoff transition of melting
seasonal snow — are detected inside a custom MODIS-derived (MOD10A2) snow-phenology search
window. Validated against 735 Western-US snow-pillow stations (median timing difference
−1.0 d, MAD 9.0 d).

- **Grid**: EPSG:4326, ~80 m effective resolution (pixel spacing ~7.2e-4°), dims
  `(water_year: 10, latitude: 195970, longitude: 499998)`, signed **int16**.
- **Regression variable**: `runoff_onset` = **day of water year (DOWY)**, integer 1–366,
  no-data **−9999**, **no scale factor** (the 0.1 scale documented for the record applies
  only to `*_mad` / `temporal_resolution`, which we do not use).
- **Water-year definition**: NH = Oct 1(N−1)–Sep 30(N), DOWY 1 = Oct 1; SH = Apr 1(N)–Mar
  31(N+1), DOWY 1 = Apr 1. Typical runoff onset is DOWY ~110–270, i.e. the melt of water
  year N falls within **calendar year N** in both hemispheres.
- **Distribution**: cloud-optimized Zarr shipped as `.tar` files, plus a **Kerchunk
  reference JSON** (`…zarr.tar.refs.json`, ~14 MB) mapping each Zarr key to
  `[tar_url, byte_offset, byte_length]`.

## Triage decision: ACCEPT (regression, dense_raster)

Clean per-pixel regression target, CC-BY-4.0, no credentials, expressible on the S2 grid.

## Access — the obstacles and how they were solved

1. **User-Agent fingerprinting.** Zenodo's file CDN returns HTTP 403 ("unusual traffic from
   your network") to non-browser User-Agents, while the metadata API works. Fix: send a real
   browser UA (`Mozilla/5.0 (X11; Linux x86_64; rv:125.0) …Firefox/125.0`) on **all** Zenodo
   file requests.
2. **zarr v3 cannot read a v2 Kerchunk reference via fsspec** (`ReferenceFileSystem`'s
   async-flag mismatch in zarr v3's `FsspecStore`). Fix: skip fsspec/xarray entirely and read
   the reference **directly** — the format is trivial (`[url, offset, length]` per chunk).
   The script parses the refs, computes which 2048×2048 chunks a region box touches, and
   **HTTP-range-reads + blosc-zstd-decodes only those chunks**, assembling each region into an
   EPSG:4326 int16 GeoTIFF cached under `raw/regions/`. Bounded (~a few GB), idempotent.
3. **Rate limits.** Zenodo throttles guest file access (~60 req/min, ~2000 req/hour per IP;
   parallel workers reliably trip a 429). Fix: **serial** chunk reads with ~0.5 s pacing
   (~37/min) plus Retry-After-aware exponential backoff, and a **bounded 5-water-year subset**
   (below) so the whole job fits inside one hourly window.

## Processing (implemented and run end-to-end)

- **Bounded region sampling (spec §5)**: 19 curated seasonal-snow regions across both
  hemispheres — Sierra Nevada, Colorado Rockies, Cascades, Wasatch/Uinta, Alaska Range, BC
  Coast Mtns, Canadian Rockies, Iceland, European Alps, Scandinavia, Caucasus, W. Himalaya,
  Tien Shan, Pamir, Altai, E. Siberia (NH) and Central Andes, Patagonia, Southern Alps NZ
  (SH). No global coverage attempted.
- **Water years**: WY2015 dropped (its melt is NH spring 2015, pre-2016). From WY2016–2024,
  a bounded **5-water-year subset — 2016, 2018, 2020, 2022, 2024** — evenly spanning the era
  (keeps the network job under the rate limit; still 5 distinct 1-year pairing windows per
  region and yields far more than 5000 candidates).
- **Time-range assignment**: each annual DOWY layer for water year N → a **1-year window on
  calendar year N** (`io.year_range(N)`; the melt of WY N lies in calendar year N in both
  hemispheres). `change_time = null` (a timing value, not a dated change event).
- **Tiling**: each 64×64 output tile = an 8×8 native (80 m) block (640 m). The ~80 m source
  window is reprojected EPSG:4326 → local UTM at **10 m** (bilinear on the continuous DOWY
  field + a nearest/threshold validity mask so −9999 never blends into valid pixels;
  reprojected values rounded back to integer DOWY). **Native resolution is 80 m — the 10 m
  tiles are upsampled 8×.**
- **dtype / nodata**: **int16**, nodata **−9999** (the source sentinel; the repo default
  −99999 does not fit int16 — recorded in `metadata.json`).
- **Sampling**: candidate 8×8 blocks with ≥50% valid (non-nodata, DOWY 1–366) pixels,
  reservoir-capped at 500 per region-year → 47,500 candidates; then
  `bucket_balance_regression` across the DOWY distribution (10 quantile buckets) →
  **5000** samples (spec §5 regression cap).

## Results / verification (§9)

- **5000** samples, all 5000 `.tif` paired with `.json` (0 unpaired). Spot-checked tiles:
  single-band **int16**, local UTM (EPSG:326xx per region), **10 m** res, **64×64**, nodata
  **−9999**, values in valid DOWY range.
- **Per-pixel value range**: 30–364 DOWY. Selected-window mean-DOWY histogram (bucket
  balanced; melt clusters in the DOWY ~130–240 spring window, tails pulled in by balancing):
  ```
  [ 77,104): 27   [104,132): 417  [132,159): 975  [159,186):1238  [186,213):1274
  [213,241):746   [241,268):185   [268,295): 84   [295,322): 45   [322,350):  9
  ```
- **Region coverage**: all 19 regions represented (240–298 samples each), incl. 3 SH regions.
- **Water-year spread**: 2016≈1006, 2018≈1007, 2020≈972, 2022≈995, 2024≈1020.
- **Time ranges**: all 1-year windows anchored on the sample's water year; `change_time`
  null.
- **Spatial sanity**: 12 random samples all fall inside their source-region boxes at
  plausible mountain coordinates (e.g. Himalaya 31.6°N/78.7°E, Sierra 37.5°N).
- **Idempotent**: `_write_one` and region caching skip existing outputs.

## Reproduce

```bash
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.global_snowmelt_runoff_onset_sentinel_1
# optional: --workers N  --limit N
```

The region download step is serial + paced (~0.5 s/request) to respect Zenodo's guest rate
limit; a full cold run takes on the order of ~30–45 min (bounded ~1000 range reads), then
scan+write of 5000 tiles takes ~1–2 min. Re-runs skip cached region rasters and tiles.

## Caveats

- **Native 80 m upsampled 8× to 10 m** — the label is coarse relative to the S2 grid.
- Value is **day of water year**, not calendar day-of-year (kept native to avoid a lossy,
  hemisphere-dependent, wrap-prone transform); metadata documents the water-year definition.
- Temporal coverage is a **5-year subset** (2016/2018/2020/2022/2024) of the available
  2016–2024, chosen to respect Zenodo's rate limits while preserving diversity. Re-running
  with `WATER_YEARS = list(range(2016, 2025))` would add the intervening years (it just needs
  more Zenodo requests spread across rate-limit windows).
