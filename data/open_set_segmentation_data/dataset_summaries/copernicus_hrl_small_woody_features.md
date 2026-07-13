# Copernicus HRL Small Woody Features

- **Slug:** `copernicus_hrl_small_woody_features`
- **Status:** completed
- **Task type:** classification (dense, 2-class)
- **Samples:** 1000 label tiles (64x64, 10 m, local UTM)

## Source

Copernicus Land Monitoring Service (CLMS) High Resolution Layer **Small Woody Features
(SWF)**, EEA / Copernicus Land. Pan-European (EEA38 + UK) 5 m raster produced by photo-
interpretation of 2.5-5 m VHR imagery, marking hedgerows, tree rows, and small
woods/patches too small or narrow to be captured by the standard forest layers.
Product page: https://land.copernicus.eu/en/products/high-resolution-layer-small-woody-features

**Reference year 2021** (the most recent vintage; 2015 and 2018 also exist). 2021 sits at
the end of the manifest range (2016-2021) and inside the Sentinel era.

## Access

The CLMS download portal is login-gated, but the products are published open-access as
**public ArcGIS ImageServer layers on the EEA DiscoMap server (no credential required)**:

    https://image.discomap.eea.europa.eu/arcgis/rest/services/GioLandPublic/
        HRL_SmallWoodyFeatures_2021_005m/ImageServer/exportImage

We pull raw 5 m pixel blocks (`f=image`, `format=tiff`, EPSG:3035 / LAEA Europe, U8
thematic, nearest interpolation) over a spread of representative regions.

## Class mapping (judgment call)

The manifest lists two classes -- "linear woody features (hedgerows)" and "patchy woody
features". **That linear/patchy split lives only in the SWF *vector* product; the public
5 m raster is a single woody-presence mask** (2021 legend: 0 = Non-SWF area, 1 = SWF
area; 2018 legend adds 254 = unclassified, 255 = outside). We therefore produce a 2-class
dense segmentation:

| id | name | meaning |
|----|------|---------|
| 0 | non_woody | Non-SWF background — a REAL observed class (open land, cropland, built-up, water, large forest), **not** a fabricated negative. |
| 1 | small_woody_feature | Hedgerow / tree row / small wood / small woody patch (linear + patchy merged). |

`255` = nodata/ignore (unclassified / outside-product pixels, reprojection edges).

## 10 m observability (VHR-native handling, spec §4)

Native label is 5 m. We reproject each source window from EPSG:3035 5 m to a local UTM
grid at **10 m with MODE resampling** (categorical — never bilinear). A 10 m pixel
aggregates ~4 native 5 m pixels; mode keeps woody only where it is the **majority** of
the 10 m pixel. This deliberately **retains resolvable woody cover** (multi-row
hedgerows, small woods, dense field-boundary networks) and **coarsens away sub-pixel
single-row hedgerows** (a 5 m-wide hedge occupying 1 of 4 sub-pixels is dropped). This is
the intended behaviour per the VHR-native guidance: fine sub-pixel features are lost at
10 m rather than fabricated. Selected tiles retain ~10-20% woody cover, so the resolvable
signal is substantial.

## Sampling (bounded-tile, spec §5)

Large European product -> bounded-tile sampling; no full-continent download. We download
one 12.8 km (2560x2560 @ 5 m) block per region over 12 representative hedgerow / small-
woody landscapes, scan for 64x64 (@10 m = 640 m; 128x128 native) windows containing
>= 2% woody cover and <= 20% nodata, then apply **tiles-per-class balanced** selection
(`select_tiles_per_class`, per_class=1000). Every window contains both classes, so the
result is 1000 tiles (woody and background co-present in all).

Per-region selected counts: brittany_fr 117, ireland_midlands 113, galicia_es 113,
normandy_fr 110, vendee_fr 108, denmark 101, po_valley_it 89, lower_saxony_de 81,
netherlands 65, austria 52, poland 51. **devon_uk 0** — the downloaded UK block was
entirely Non-SWF (value 0); **the UK is not covered in the 2021 SWF vintage** (post-
Brexit), so it yielded no woody windows. Ireland covers the British-Isles hedgerow
landscape. (To include the UK one would use the 2018 vintage; not done here to keep a
single reference year.)

## Time range

Static annual product -> 1-year window anchored on the reference year, `[2021-01-01,
2022-01-01)`. No change labels.

## Verification

- 1000 `.tif` + 1000 `.json`; every tif is single-band uint8, 64x64, local UTM at 10 m.
- Pixel values across all tiles are exactly {0, 1, 255}; all 1000 tiles contain the woody
  class; nodata (255) appears only at block/reprojection edges.
- metadata.json class ids {0,1} cover all non-nodata values; sample JSONs carry a 1-year
  2021 range and `classes_present`.
- Georeferencing sanity: sample UTM zones match region longitudes (e.g. Galicia -> UTM
  29N / EPSG:32629). A full Sentinel-2 overlay was not run (derived-product raster,
  georeferencing is exact via rslearn `encode_raster`).

## Caveats

- Binary woody-presence only; the manifest's linear-vs-patchy distinction is unavailable
  in the public raster.
- Sub-pixel single-row hedgerows are coarsened away by the 5 m->10 m mode resampling.
- UK absent from the 2021 vintage (0 samples from the UK block).

## Reproduce

    python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.copernicus_hrl_small_woody_features

Idempotent: re-download skips existing blocks; re-write skips existing `{id}.tif`;
selection is seeded/deterministic.
