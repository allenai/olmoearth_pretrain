# Bark Beetle Disturbance, Pokljuka (Slovenia)

- **Slug:** `bark_beetle_disturbance_pokljuka_slovenia`
- **Status:** completed
- **Task type:** classification (positive-only, single class)
- **Num samples:** 585
- **Source:** Zenodo record [10.5281/zenodo.15260584](https://doi.org/10.5281/zenodo.15260584) — "Bark beetle geospatial dataset from 2017 to 2021 (Pokljuka, Slovenia)"
- **License:** CC-BY-4.0 (open, no credentials)

## Source

A single GeoTIFF, `Change detection mask 2017-2021.tif` (60 MB), covering the Pokljuka
plateau, Slovenia. Bark-beetle (*Ips typographus*) spruce disturbance was derived from
2017–2021 Sentinel-2 NDVI + NBSI vegetation-index time series, processed with the CuSum
change-detection algorithm; the mask is the intersection of the high-magnitude breakpoint
maps from both indices, overlaid with in-situ ground-truth validation.

- CRS **EPSG:32633** (UTM 33N), **3000×2500 px at 10 m** (~30×25 km), single band, float64.
- Values: **0** = no disturbance / background (99.24%, file nodata=0); **1** = bark-beetle
  disturbance (0.758%, 56,854 px).

## Processing (label_type = dense_raster, single positive class)

Because the source is already local UTM at 10 m, tiles are cropped **directly from its
grid with no reprojection** (spec §2 allows reusing a source window's CRS when already UTM
at 10 m). The raster is partitioned into a non-overlapping **64×64 (640 m) grid** (1794
cells); every cell containing **≥1 disturbance pixel** is kept — **585 tiles** (median 48
disturbance px/tile, range 1–885). 585 < the 1000/class cap, so all positives are kept; no
subsampling.

- Per pixel: source value 1 → **class id 0** ("bark beetle disturbance"); every other pixel
  → **255 (ignore)**.
- **Positive-only** (spec §5): the manifest lists a single class, so no "no-disturbance"
  negative class is fabricated — non-disturbance pixels are left as ignore and downstream
  assembly supplies negatives from other datasets. (The source 0 could alternatively be
  read as a confident negative, but only disturbance was field-validated, so ignore is the
  safer choice.)

Output: `datasets/bark_beetle_disturbance_pokljuka_slovenia/locations/{000000..000584}.tif`
(single-band uint8, EPSG:32633, 64×64, nodata 255) + matching `.json` sidecars, plus
`metadata.json`.

## Class mapping

| id | name | count |
|----|------|-------|
| 0 | bark beetle disturbance | 585 tiles |
| 255 | nodata / ignore (all non-disturbance pixels) | — |

## Time range and change handling (pre/post scheme)

The mask is a **cumulative 2017–2021 disturbance product** with **no per-pixel disturbance
dates** (they are not recoverable from the single aggregate raster) — the disturbance
occurred somewhere in that span. It is encoded under the **pre/post change scheme** by
bracketing the whole span with two **fixed six-month windows** (each ≤ 183 days) and
`time_range` = **null**:

- `pre_time_range` = **summer 2016** (before the disturbance period).
- `post_time_range` = **summer 2022** (after it) — the disturbance occurred somewhere in
  between.
- `change_time` = **null** (the exact date is unknown).

Summer windows avoid snow-cover confusion.

**Previously rejected; now resolved by pre/post windows.** This replaces the earlier
encoding (annual disturbance-presence anchored on 2021: `time_range` 2021-01-01 → 2022-01-01,
`change_time = null`) and the accompanying **rejection** on change-timing grounds (the
disturbance not resolvable to within ~1–2 months). With the disturbance bracketed between a
genuine before/after image pair the timing imprecision is no longer a problem, so the
dataset is **completed / usable**.

## Verification

- Opened 7 output tifs: all single-band uint8, EPSG:32633, 64×64, nodata 255, values ⊆
  {0, 255}.
- 585 tifs each have a matching `.json` with `time_range` = **null**, fixed `pre_time_range`
  (summer 2016) and `post_time_range` (summer 2022) each ≤ 183 days, and `change_time` null;
  `metadata.json` class ids cover all values in the tifs.
- **Georeferencing:** verified byte-exact — for sampled class-0 output pixels, the source
  raster holds value 1 at the identical world coordinate (tiles are a same-grid remap, so
  alignment is exact). A live Sentinel-2 overlay was not fetched because the label is itself
  a field-validated 10 m Sentinel-2 product on the identical grid, making the overlay
  redundant.
- Re-running is idempotent (existing `{id}.tif` skipped).

## Reproduce

```bash
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.bark_beetle_disturbance_pokljuka_slovenia
```

## Caveats

- Single small study area (~30×25 km, one UTM zone) → only 585 samples; one class.
- Cumulative multi-year disturbance bracketed by fixed pre (summer 2016) / post (summer
  2022) windows rather than a dated per-pixel change (see above).
- Positive-only: tiles are label-sparse (~48 disturbance px in a 4096-px tile, rest ignore);
  negatives are added at pretraining-assembly time.
