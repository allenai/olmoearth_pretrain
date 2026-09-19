# S2Looking — REJECTED (no recoverable georeferencing; VHR change unsuited to 10 m)

- **Slug**: `s2looking`
- **Name**: S2Looking
- **Source**: GitHub / arXiv (https://github.com/S2Looking/Dataset), paper
  arXiv:2107.09244 / Remote Sensing 13(24):5094 (2021)
- **Family / region**: change_detection / rural areas worldwide
- **Label type (manifest)**: dense_raster + instance (building-change), VHR 0.5–0.8 m,
  manual annotation
- **Classes (manifest)**: building appeared, building disappeared, no-change
- **License**: CC-BY-NC-SA-4.0
- **Status**: **rejected**
- **Rejection reason**: released tiles carry no real-world coordinates (coordinate-free
  8-bit PNG pairs with opaque numeric filenames; the authors explicitly removed the
  coordinate information), so labels cannot be placed on the Sentinel-2 grid. A secondary
  blocker (VHR building change too fine for 10 m S2, and no per-pair dates for a valid
  change_time) is moot given the georeferencing failure.

## What S2Looking is

S2Looking is a satellite side-looking building-change-detection dataset: 5,000 registered
bitemporal image pairs (1024×1024, 0.5–0.8 m/pixel) over rural areas worldwide, with
65,920+ annotated building-change instances. Images come from the GaoFen, SuperView and
BeiJing-2 satellites, 2017–2020, with a 1–3 year span between the two acquisitions. Each
sample has two images (Image1/Image2) and label maps separating newly-built (label1) and
demolished (label2) buildings. It is designed for VHR change-detection benchmarking, not
for pairing with a satellite time series.

## Why it is rejected

The open-set-segmentation pipeline pairs each label patch with Sentinel-2 / Sentinel-1 /
Landsat imagery by **geography and time**, so every label must carry real-world
coordinates to reproject to a local-UTM 10 m grid. S2Looking provides none:

1. **Release format is coordinate-free 8-bit PNG.** The paper states verbatim: "The image
   pairs in the dataset are converted from the original TIFF format with 16 bit to PNG
   format with 8 bit." PNG carries no CRS and no geotransform. The single distributed
   archive (`S2Looking.zip`, ~10.2 GB) lays tiles out under `train|val|test/{Image1,
   Image2,label,label1,label2}/` with opaque numeric filenames (e.g. `1.png`); there is no
   world file, GeoTIFF, CSV, or coordinate index.
2. **The authors explicitly removed the coordinates.** The dataset was built by cropping
   large scenes using "rough coordinate information," but that "geographic coordinate
   information has been removed from the data" (removed for the associated challenge and
   in the public release). So per-tile lon/lat is not recoverable from the release.
3. **No per-tile coordinate lookup exists.** Region provenance is only "rural areas
   throughout the world"; there is no published table mapping the numeric tile IDs to
   lat/lon, and the GitHub/Google-Drive/Baidu distributions all mirror the same
   coordinate-free PNG archive.

Because per-tile geocoordinates are unrecoverable, the tiles cannot be resampled to 10 m
and located on the S2 grid — the spec's stated rejection condition (§8.2: no recoverable
geocoordinates).

### Secondary blockers (moot, but recorded)

- **Too fine for 10 m.** The labels mark individual rural buildings at 0.5–0.8 m. A single
  building footprint is well under one 10 m Sentinel-2 pixel, so the "building appeared /
  disappeared" change signal is not resolvable from S2/S1/Landsat; the class set could not
  be salvaged by coarsening.
- **No usable change_time.** The pipeline needs a `change_time` to center a ≤1-year window
  (§5). S2Looking gives only a dataset-wide 2017–2020 range with a 1–3 year gap between the
  two images and **no per-pair acquisition dates**, so a 1-year change window would be
  ill-posed even if the tiles were georeferenced.

## Access method (for the record)

Public, no credentials strictly required for the data itself (Google Drive + Baidu links,
Baidu password `25Ao`; a ~10.2 GB HyperAI mirror also exists). Nothing was downloaded in
bulk — triage rejects cheaply on the documented coordinate-free PNG format. Nothing was
written under weka `datasets/` beyond the rejection `registry_entry.json`.

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.s2looking
```

This re-states the georeferencing check and re-writes this summary; it makes no dataset
outputs.
