# USGS Exotic Annual Grass (EAG) Fractional Cover — invasive-grass percent cover

- **slug**: `usgs_exotic_annual_grass_fractional_cover`
- **task_type**: regression
- **num_samples**: 4976
- **source**: USGS EROS / MRLC — "RCMAP – Weekly Herbaceous and Exotic Annual Grass (EAG)".
  Product page https://www.mrlc.gov/data/type/exotic-annual-grass ; current generation data
  release DOI https://doi.org/10.5066/P13QWBFH (2016–2026). The manifest cites the sibling
  annual all-species release DOI https://doi.org/10.5066/P9GC5JVG (2016–2024) — same product
  line, 30 m Total EAG.
- **license**: public domain (US Government work).

## What the source is

EAG maps the per-pixel **percent cover (0–100)** of combined invasive **exotic annual
grasses** — cheatgrass (*Bromus tectorum*), medusahead (*Taeniatherum caput-medusae*), field
brome, and ~12 other *Bromus*/*Ventenata* species — across the **western US sagebrush biome
/ drylands** at **30 m**, derived from **Harmonized Landsat-Sentinel (HLS)** imagery via a
machine-learning regression trained on **BLM AIM + RCMAP field plots**. Native rasters are
single-band **uint8, EPSG:5070 (CONUS Albers)**; values 0–100 are valid percent cover and
**101 marks masked / non-rangeland / water / out-of-area** pixels.

This is a **regression** dataset (continuous per-pixel percent cover). We regress the
**Total EAG** component — the combined exotic-annual-grass cover — which is the manifest's
primary "exotic annual grass cover" class.

## Access / download (no full-mosaic download)

The full-resolution rasters are distributed via a USGS ScienceBase download that is now
gated behind an **interactive Captcha** (large files migrated to S3, `?f=__disk__` returns
404, `requestDownload` serves a Captcha page), and the MRLC `data-bundles/` filenames for
EAG are not resolvable. The **same Total EAG native (30 m) coverage** is served as an OGC
service from the MRLC GeoServer, so labels are read on demand via **bounded WMS GetMap**
requests:

```
https://dmsdata.cr.usgs.gov/geoserver/mrlc_total-eag-native_westernconus_week_data/wms
  layer = total-eag-native_westernconus_week_data   (Total EAG native, 30 m, EPSG:5070)
  format = image/geotiff  (GeoServer returns the RAW 0–100 values, not a styled RGB)
  TIME   = 2026-06-25T00:00:00.000Z
```

`image/geotiff` GetMap is the OGC analogue of a COG range read: we never download the whole
western-US mosaic. A reusable helper `download.wms_getmap_geotiff(...)` was added to the
shared module. Only one small file is persisted to `raw/`: a decimated (~1.48 km/px)
full-extent overview used to pick candidate windows.

**Time / generation note.** Only the 2016–2026 generation is currently loaded on the
GeoServer (weekly granules; 2026 available so far). The **Total EAG** component is
cumulative "at any point year-to-date", so the **latest available week (2026-06-25)** is the
most-complete annual-representative EAG cover map. 2026 is post-2016 (Sentinel era); each
tile is anchored to a **1-year window on 2026** (`[2026-01-01, 2027-01-01)`). This mirrors
the RCMAP dataset's choice to use the current MRLC generation rather than the manifest's
older release year range.

## Processing

- **Candidate selection (spatially distributed, cover-balanced).** A single full-extent
  overview GetMap (2000×2002 px ≈ 1.48 km/px) is read; valid (0–100) pixels give candidate
  window centers with an approximate cover value. Candidates are **confined to the western
  US drylands** by keeping longitude ≤ **−100°** (the 100th-meridian arid/humid divide): the
  coverage's rectangular EPSG:5070 extent runs east to ~−90° where the margin is 0-cover edge
  fill outside the mapping region. Up to 400k candidates are pooled.
- **Bucket balancing** (EAG cover is heavily zero-inflated). Tiles are balanced across
  **fixed percent-cover buckets `[0,1,5,10,20,30,50,101]`** by the candidate cover value,
  **714 per bucket** — giving an even spread of cover levels (many low/zero-cover tiles **and**
  high-invasion hotspots), not a mostly-0% corpus. (The shared quantile bucketer degenerates
  on zero-inflated data, so fixed buckets are used, as in the RCMAP dataset.)
- **Per-tile read + reprojection.** For each selected window, a native-30 m GeoTIFF is
  fetched from the server over a ~1200 m EPSG:5070 window (covers the 640 m tile with
  margin). It is reprojected/resampled to **local UTM at 10 m, 64×64 (~640 m)** via
  `GeotiffRasterFormat.decode_raster` (WarpedVRT), using **bilinear** resampling for the
  continuous cover field (30 m → 10 m upsample). WarpedVRT respects the source `nodata=101`,
  so the mask is not blended into valid pixels; output pixels that are <0, >100, equal to the
  source mask (101), or non-finite are set to `-99999`.
- **30 m → 10 m resample (documented).** Native resolution is 30 m; pretraining tiles are
  10 m. Each tile is resampled by a factor of 3 with bilinear interpolation. No new
  information is created — the label remains the 30 m EAG cover field, expressed on the 10 m
  grid.
- **Time range**: 1-year window on 2026 (seasonal/annual label per spec §5). No change
  labels (`change_time = null`).
- **All-nodata tiles dropped**: 24 selected windows fell entirely on the source mask
  (all-nodata) and were not written; the script skips writing them (idempotent).

## Output

- `datasets/usgs_exotic_annual_grass_fractional_cover/locations/{000000..004999}.tif` —
  single-band **float32**, local UTM, 10 m, 64×64, nodata **-99999**, values = Total EAG
  percent cover 0–100. (Sample ids are a running index; 24 all-nodata ids are absent.)
- matching `.json` sidecars (crs, pixel_bounds, ≤1-year `time_range`, `change_time=null`,
  `source_id=total_eag_2026-06-25`).
- `metadata.json` — regression block (`name: exotic_annual_grass_cover`, unit `percent
  cover`, dtype float32, value_range, nodata -99999, `source_mask_value: 101`, buckets).

## Stats

- **num_samples**: 4976 (all 2026). Overview-value bucket counts: 714 in each of the 7
  buckets.
- Observed **per-pixel cover range across tiles: [0, 92] %**. Pixel-level (19.1 M valid
  pixels): mean ≈ 16.5%, median 10%, p90 43%, p99 65%; ~18% of valid pixels are 0-cover and
  ~82% non-zero (bucket balancing pulls the sample toward invaded areas while retaining a
  full low/zero tail).
- Spatial spread: tile centroids span **UTM zones 10N–14N** (Pacific coast → western Great
  Plains), lon −124.3° to −100.0°, lat 28.3° to 49.5° — all in the western US drylands.

## Verification (spec §9)

- All 4976 tiles: single-band **float32, 64×64, 10 m, UTM (EPSG:326xx), nodata -99999**,
  values in [0, 100] — 0 failures.
- Every `.tif` has a matching `.json`; **max `time_range` = 365 days** (≤ 1 year); all
  `change_time = null`.
- `metadata.json` regression `value_range` [0, 92] covers all tile values.
- **Spatial sanity**: all 4976 centroids fall inside a western-US bbox (−125…−100°,
  25…50°). Georeferencing validated via exact EPSG:5070→UTM reprojection and coordinate
  placement; a full Sentinel-2 overlay was not performed (standard USGS Albers→UTM path, as
  for the RCMAP fractional-cover dataset).
- **Idempotent**: re-running skips existing tiles (and reads them back so stats stay
  correct); a second run left the count unchanged at 4976.

## Caveats / judgment calls

- **Product generation / year**: the manifest references the annual 2016–2024 EAG release,
  but its full-resolution rasters are only obtainable through a Captcha-gated ScienceBase
  download. The GeoServer serves the current 2016–2026 generation of the *same* Total EAG
  product; the latest cumulative week (2026-06-25) was used. Both are 30 m HLS-based EAG
  fractional cover. 2026 is post-2016.
- **Weekly cumulative vs annual**: "Total EAG" at the latest week is the year-to-date
  maximum EAG cover, a close analogue of the annual product's peak cover.
- **Single component**: only the combined Total EAG cover is regressed; species-resolved
  components (cheatgrass, medusahead, field brome) are separate single-band products and are
  not included here.
- **Western-US restriction**: candidates east of −100° were dropped to exclude rectangular
  edge fill; genuine (sparser) EAG mapping in the far eastern Great Plains is therefore not
  sampled.

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.usgs_exotic_annual_grass_fractional_cover --workers 64
```

(Fetches the decimated overview on first run if absent, then reads per-tile windows from the
MRLC GeoServer via bounded WMS GetMap; idempotent.)
