# MMFlood

- **Slug**: `mmflood`
- **Status**: completed
- **Task type**: classification (dense per-pixel, binary)
- **Num samples**: 1000 label patches (64×64 GeoTIFFs)
- **Classes**: `0 = not-flooded`, `1 = flooded`; `255 = nodata/ignore`
- **Family / label_type**: flood / dense_raster
- **License**: CC-BY-4.0

## Source

MMFlood — "MMFlood: A Multimodal Dataset for Flood Delineation from Satellite
Imagery" (Montello, Arnaudo, Rossi; *IEEE Access* 2022, DOI
10.1109/ACCESS.2022.3205419). Zenodo record
[6534637](https://zenodo.org/records/6534637), a single `mmflood.zip` (11.26 GB).
The dataset covers 95 Copernicus Emergency Management Service (EMS) flood
activations (EMSR codes) across 42 countries. Each activation is split into one or
more AOIs stored as `mmflood/EMSR{code}-{aoi}/` with four co-registered
modalities: `s1_raw` (Sentinel-1 SAR), `DEM`, `hydro` (permanent hydrography),
and `mask` (the manually-delineated flood extent). `activations.json` gives each
activation's title, country, event `start`/`end` datetimes, lon/lat centroid, and
train/val/test subset.

## Access method

The archive is compressed with **Deflate64** (ZIP method 9), which Python's stdlib
`zipfile` cannot decode — the `zipfile-deflate64` shim is required (installed at
runtime).

Zenodo throttles aggregate bandwidth per IP (~6–13 MB/s) and **429-rate-limits by
request count**, so per-file HTTP range extraction of the ~1748 mask tifs (each
needing several range reads) reliably tripped `429 TOO MANY REQUESTS` even when
serial. We therefore download the whole zip once over a single connection using a
few large (64 MB) sequential range requests (`download_raw` → `_download_zip`,
resumable, ~14 min at ~13 MB/s), then extract **only** the 1748 `mask/*.tif`
rasters + `activations.json` locally (crash-safe: each member is written to a
`.part` and renamed, and re-extracted if its on-disk size doesn't match the
archive size). The 13 GB of Sentinel-1 SAR and the DEM/hydro layers are **not**
extracted — pretraining supplies its own imagery, so only the labels are needed.

## Label semantics & class mapping

Mask rasters are single-band **float32** in **EPSG:4326** (geographic) at ~10–14 m,
valued `0.0 = not-flooded`, `1.0 = flooded` (no nodata). We keep the manifest's
binary scheme:

| id | name | meaning |
|----|------|---------|
| 0 | not-flooded | mapped AOI area not delineated as flood inundation (dry land + pre-existing water) |
| 1 | flooded | Copernicus EMS flood-inundation delineation (manual photointerpretation) |
| 255 | nodata/ignore | pixels outside the AOI footprint after reprojection |

We deliberately did **not** split permanent water out of `not-flooded` using the
`hydro` layer (unlike `worldfloods_v2`/`sen1floods11`, whose manifests define a
water/permanent split) — the MMFlood manifest defines only flooded/not-flooded.

## Processing (dense_raster, reprojected)

Each geographic mask is reprojected to its local UTM zone at **10 m/pixel**
(nearest resampling — categorical) onto a grid snapped to the global 10 m S2 grid,
with pixels outside the source footprint set to 255 (a reprojected validity mask
distinguishes true `not-flooded` from uncovered corners). The reprojected mask is
tiled into **64×64** patches. Tiles >50% nodata are dropped; a tile counts toward
a class only with ≥32 px of it. From 348,110 candidate tiles, selection is
**tiles-per-class balanced** (`sampling.select_tiles_per_class`, ≤1000 tiles/class,
25k cap) with the rare `flooded` class filled first. Because every flood tile also
contains `not-flooded` background, the 1000 selected flood-bearing tiles already
satisfy the not-flooded target, so the final set is **1000 tiles, each containing
both classes** (no pure-background tiles were added). All three source subsets
(train/val/test) are used.

## Time range & change handling

Flood is a **dated event** → treated as a **change label** (spec §5). Copernicus
EMS activation `start` dates the flood onset to within days (median activation
`start`→`end` span is 3 days), well inside the ~1–2 month timing-precision
requirement. Each sample sets `change_time` = activation start and `time_range` =
a **360-day window centered** on it (≤1 year cap). Pretraining then uses a sample
only when the sampled input window spans the flood.

## Date filtering

9 of the 95 activations have `start` before 2016 (2014–2015 events: EMSR107, 117,
118, 120, 122, 141, 147, 149, 150) and fall outside the Sentinel era — they were
**dropped** (spec §8), removing 192 mask tifs. The remaining **86 activations
(1556 mask tifs)** were processed. The 1000 selected tiles span **75 distinct
flood events**.

## Class / pixel balance

- Tiles containing each class: `not-flooded` 1000, `flooded` 1000 (every tile has
  both).
- Pixel fractions across the 1000 patches: **not-flooded 77.7%, flooded 21.2%,
  nodata 1.1%** (flood fraction is far above the ~1–4% in raw masks because
  flood-bearing tiles were prioritized).

## Verification (spec §9)

- Opened sample patches: single-band, **uint8**, UTM CRS (e.g. EPSG:32629) at
  10 m, 64×64, nodata 255; dataset-wide pixel values are exactly {0, 1, 255},
  matching the class map. Max tile dimension 64.
- Every `.tif` has a matching `.json`; all 1000 `time_range`s are ≤360 days and all
  carry `change_time`.
- Spatial sanity: tile-center lon/lat fall within their activation's country/region
  (Mexico, Croatia, Madagascar, France, … tiles 27–98 km from the single AOI
  centroid — consistent with AOI extent), confirming correct georeferencing.
- Re-running the script is idempotent (still exactly 1000 tif + 1000 json; extract
  and write phases skip existing/size-matching files).

## Caveats

- Binary scheme only; permanent water is folded into `not-flooded` (no `hydro`
  split).
- The nearest-resampled reprojection from ~10–14 m geographic to 10 m UTM is a mild
  regrid; flood pixel counts are preserved to within ~0.1%.
- `activations.json` provides one centroid per activation; the per-AOI/per-tile
  georeferencing comes from the mask GeoTIFFs themselves (verified above).

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.mmflood --workers 48
```

Requires the `zipfile-deflate64` package (`pip install zipfile-deflate64`). The
script downloads the Zenodo zip to
`raw/mmflood/mmflood.zip` (resumable), extracts only the mask rasters, and writes
`datasets/mmflood/{metadata.json, locations/*.tif, locations/*.json}` under the
open-set-segmentation output root on weka.
