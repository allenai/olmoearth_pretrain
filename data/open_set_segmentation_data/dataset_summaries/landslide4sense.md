# Landslide4Sense — REJECTED (no recoverable georeferencing)

- **Slug**: `landslide4sense`
- **Name**: Landslide4Sense
- **Source**: IARAI Landslide4Sense 2022 competition (https://www.iarai.ac.at/landslide4sense/);
  code/data description https://github.com/iarai/Landslide4Sense-2022; public mirrors: Zenodo record 10463239
  (https://zenodo.org/records/10463239), HuggingFace
  `ibm-nasa-geospatial/Landslide4sense`.
- **Family / region**: landslide / Japan, India, Nepal, Taiwan (country level only)
- **Label type (manifest)**: dense_raster, binary (landslide / non-landslide),
  expert photointerpretation, time_range 2016-2019
- **License**: open (research)
- **Status**: **rejected**
- **Rejection reason**: patches carry no real-world coordinates and cannot be placed on
  the Sentinel-2 grid.

## What Landslide4Sense is

A pixel-level landslide-segmentation benchmark released for the IARAI 2022 competition
(Ghorbanzadeh et al.). It provides 3,799 train / 245 validation / 800 test patches of
128 x 128 px, each fusing 14 bands — Sentinel-2 multispectral B1-B12, ALOS PALSAR slope
(B13) and DEM (B14) — resampled to ~10 m, with an expert-annotated binary mask
(landslide vs non-landslide). Data are distributed as HDF5: `img/image_X.h5`
(128x128x14 float array) and `mask/mask_X.h5` (128x128 uint8 mask).

## Why it is rejected

The open-set-segmentation pipeline pairs each label patch with Sentinel-2 / Sentinel-1 /
Landsat imagery by **geography and time**, so every patch must carry real-world
coordinates to reproject to a local-UTM 10 m grid. Landslide4Sense provides none:

1. **Release format is coordinate-free HDF5.** Each `.h5` is a bare numeric array
   (image `(128,128,14)`, mask `(128,128)`) with an opaque running-index filename.
   There is no CRS, geotransform, lon/lat, or bounding box in the archives
   (`TrainData.zip` / `ValidData.zip` / `TestData.zip`), and no sidecar coordinate table
   in the GitHub, Zenodo, or HuggingFace distributions.
2. **The organizers intentionally stripped geolocation.** The IARAI Landslide4Sense-2022
   data description states verbatim: "The detailed geographic information and acquisition time will not be released at the current phase in case participants may directly look for the corresponding high-resolution images to check."
3. **Region is country-level only.** The manifest region is "Japan, India, Nepal, Taiwan"
   — far too coarse to place a ~640 m patch on the S2 grid even approximately.

Because per-patch geocoordinates are unrecoverable, the patches cannot be located on the
S2 grid — the spec's stated rejection condition ("No recoverable geocoordinates"; the
guidance explicitly names coordinate-free HDF5 arrays, "reject like LoveDA").

## Access method (for the record)

Publicly available without credentials via Zenodo record 10463239 and the
HuggingFace mirror `ibm-nasa-geospatial/Landslide4sense`, so this is **not** an
`iarai` credential rejection — the sole blocker is the missing georeferencing. Nothing
was written under weka `datasets/` (rejection path).

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.landslide4sense
```

This re-verifies the rejection and re-writes this summary; it makes no dataset outputs.
