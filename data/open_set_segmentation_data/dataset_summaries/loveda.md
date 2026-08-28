# LoveDA — REJECTED (no recoverable georeferencing)

- **Slug**: `loveda`
- **Name**: LoveDA
- **Source**: Zenodo record 5706578 (https://zenodo.org/records/5706578) / NeurIPS 2021 D&B
- **Family / region**: land_cover / China (Nanjing, Changzhou, Wuhan)
- **Label type (manifest)**: dense_raster, VHR ~0.3 m, manual photointerpretation
- **Classes (manifest)**: background, building, road, water, barren, forest, agriculture
- **License**: CC-BY-NC-SA-4.0
- **Status**: **rejected**
- **Rejection reason**: tiles cannot be georeferenced to real coordinates (cannot be
  placed on the Sentinel-2 grid).

## What LoveDA is

LoveDA is a 0.3 m very-high-resolution semantic-segmentation dataset (5987 1024x1024
image tiles, 166768 annotated objects) sourced from Google Earth over three Chinese
cities, split Train/Val/Test and Urban/Rural, with 7 land-cover classes. It is designed
for VHR semantic segmentation and domain adaptation, not for pairing with satellite
time series.

## Why it is rejected

The open-set-segmentation pipeline pairs each label patch with Sentinel-2 / Sentinel-1 /
Landsat imagery by **geography and time**, so every label must carry real-world
coordinates to reproject to a local-UTM 10 m grid. LoveDA provides none:

1. **Release format is coordinate-free PNG.** The Zenodo record ships only
   `Train.zip`, `Val.zip`, `Test.zip` (plus `Datasheet.pdf`). Their contents are
   exclusively `.png` files under `<Split>/<Urban|Rural>/{images_png,masks_png}/`
   with opaque numeric filenames (e.g. `2522.png`). Verified by reading each zip's
   central directory: Val = 3338 `.png` entries, Train = 5044 `.png` entries, **zero**
   GeoTIFFs / world files / CSVs / coordinate indices. PNG carries no CRS and no
   geotransform.
2. **The authors confirm coordinates were stripped.** The official `Datasheet.pdf`
   states verbatim: "All data was obtained from Google Earth platform, and do not contain any coordinate location information." (data are Google Earth screenshots with no
   embedded geolocation).
3. **No per-tile lookup exists.** The manifest note that tiles are "georeferenced only
   loosely (patches over Nanjing/Changzhou/Wuhan)" describes city-level provenance only;
   there is no published table mapping the numeric tile IDs to lat/lon, and the GitHub
   distribution mirrors the same Zenodo PNGs.

Because per-tile geocoordinates are unrecoverable, the tiles cannot be resampled to 10 m
and located on the S2 grid — this is the spec's stated rejection condition
(§8.2: "Phenomenon cannot be georeferenced to real coordinates").

The secondary VHR concern (individual buildings and narrow roads at ~0.3 m are likely
unresolvable at Sentinel-2 10 m and would need to be coarsened/dropped) is moot: the
dataset fails the prior georeferencing gate.

## Access method (for the record)

Public, no credentials needed. `download.download_zenodo("5706578", raw_dir)`
fetches the zips. Nothing was written under weka `datasets/` (rejection path).

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.loveda
```

This re-lists the Zenodo record and re-writes this summary; it makes no dataset outputs.
