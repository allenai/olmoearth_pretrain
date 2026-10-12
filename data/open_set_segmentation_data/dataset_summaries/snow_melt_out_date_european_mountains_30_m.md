# Snow Melt-Out Date, European Mountains (30 m)

- **Slug**: `snow_melt_out_date_european_mountains_30_m`
- **Task type**: regression (dense_raster)
- **Status**: completed — 5000 samples
- **Source**: Zenodo record 13151801 — "Gridded snow melt-out date (SMOD) dataset for
  Pyrenees, European Alps and Greater Caucasus at 30-m spatial resolution and two periods,
  1985-1996 and 2011-2022" (Scientific Data). <https://zenodo.org/doi/10.5281/zenodo.13151801>
- **License**: CC-BY-4.0
- **Access**: public Zenodo download (no credentials). Files fetched via
  `download.download_zenodo(...)` into `raw/{slug}/`.

## What the source is
Landsat-derived 30 m rasters of **snow melt-out date** (`calDoy` = calendar day-of-year of
the last day of the continuous snow season), one file per (period x region). Each raster is
a **per-period climatology**: a single value per pixel = mean melt-out DOY over the period.
Regions: Pyrenees (PYRENE), European Alps (EUALPS), Greater Caucasus (GRTCAU). Source CRS
EPSG:3035 (LAEA Europe), 30 m. Two variants per file: `RAW` (int32, 0-365, 0 = no snow) and
`MASKED` (float32, NaN nodata, restricted to reliable snow pixels, valid DOY ~121-243).

## Decisions / judgment calls
- **Regression target = melt-out DOY** regressed directly (float32, "day of year"), nodata
  -99999. Not a change event: melt-out date is an annual per-pixel value, so `change_time`
  is null.
- **Used the MASKED variant** (high-confidence / reliable-snow pixels), per spec §4's
  preference for high-confidence windows of derived-product maps. RAW files were downloaded
  only for the smallest region (PYRENE) during inspection and are not used for labels.
- **Only the 2011-2022 period.** The 1985-1996 period is entirely pre-2016 (pre-Sentinel)
  and is excluded per spec §2/§8 (labels entirely pre-2016). The 2011-2022 climatology
  overlaps the Sentinel era.
- **Time range = a "snow year"** Sep 1 (Y) -> Aug 31 (Y+1) (364 days, <= 1 year), so the
  spring/summer melt-out falls inside the window. Because the value is a multi-year
  climatology, tiles are spread across snow years **2016/17 .. 2021/22** for temporal
  diversity (all within both the Sentinel era and the 2011-2022 product period). ~833 tiles
  per snow year.
- **Bounded-tile sampling** across the 3 regions (spec §5, large derived product, no in-situ
  reference alternative): decimated candidate scan (~1 candidate per 640 m tile) → 101,947
  candidates → bucket-balanced.
- **Bucket-balanced across DOY deciles** (spec §5): the raw DOY distribution is right-skewed
  (median ~150, tail to 243 — late-melt high-alpine/glacier pixels are rare), so balancing
  improves coverage of late-melt values. Bucket edges recorded in `metadata.json`.
- **Reprojection**: EPSG:3035 30 m → local UTM 10 m, **bilinear** resampling (melt-out DOY
  is a smooth continuous field), 64x64 (~640 m) tiles. NaN (masked) pixels → -99999.

## Outputs
- `datasets/{slug}/metadata.json` — regression block (name `snow_melt_out_date`, unit
  "day of year", value_range [121, 243], nodata -99999, bucket edges).
- `datasets/{slug}/locations/{id}.tif` — single-band float32, UTM 10 m, 64x64, nodata -99999.
- `datasets/{slug}/locations/{id}.json` — crs/pixel_bounds/time_range (snow year),
  change_time null, source_id = region.
- Counts: 5000 samples. Regions: EUALPS 3038, GRTCAU 1662, PYRENE 300 (PYRENE is the
  smallest region so contributes fewest candidates). Per-pixel value range across tiles
  [121, 243] DOY. Selected DOY percentiles p5=124, p50=150, p95=190.

## Verification
- Confirmed 5000 `.tif` each with a matching `.json`; sampled tiles are single-band float32,
  UTM at 10 m, 64x64, nodata -99999, values within the DOY range. CRS zones match regions
  (32631 Pyrenees, 32632 Alps, 32638 Caucasus). time_range <= 1 year, change_time null.
- Idempotent: re-running with `--skip-download` skips all existing outputs.

## Reproduce
```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.snow_melt_out_date_european_mountains_30_m
# raw files already present -> add --skip-download
```

## Caveats
- The label is a **multi-year climatology**, not a single-year observation; the assigned
  snow-year window is a representative pairing window, not the exact year the value was
  measured. Pretraining imagery for any 2016-2022 snow year should still align with the
  typical melt-out pattern.
- MASKED product floors valid values near DOY 121 (May 1); the lowest bucket is dense.
