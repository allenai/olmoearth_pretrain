"""Triage Landslide4Sense for open-set-segmentation -> REJECTED (no georeferencing).

Landslide4Sense (IARAI 2022 competition; Ghorbanzadeh et al.) is a pixel-level
landslide-segmentation benchmark of 128x128 patches over four landslide-prone regions
(Japan, India, Nepal, Taiwan). Each patch fuses 14 bands — 12 Sentinel-2 multispectral
bands (B1-B12) plus ALOS PALSAR slope (B13) and DEM (B14) — all resampled to ~10 m, with
an expert photo-interpreted binary mask (landslide / non-landslide).

The open-set-segmentation pipeline pairs every label patch with Sentinel-2 / Sentinel-1 /
Landsat imagery by GEOGRAPHY and TIME, so each patch must carry real-world coordinates to
reproject onto a local-UTM 10 m grid. Landslide4Sense cannot:

  * The data ship as coordinate-free HDF5 arrays. Each ``image_X.h5`` holds a raw
    (128, 128, 14) float array and each ``mask_X.h5`` a (128, 128) uint8 mask, with
    opaque running indices as filenames — no CRS, no geotransform, no lon/lat, no
    bounding box anywhere in the archives (TrainData.zip / ValidData.zip / TestData.zip).
  * The organizers deliberately stripped geolocation. The IARAI Landslide4Sense-2022
    data description states verbatim: "The detailed geographic information and acquisition
    time will not be released at the current phase in case participants may directly look
    for the corresponding high-resolution images to check."

So per-patch real-world coordinates are unrecoverable -> patches cannot be located on the
S2 grid. This is the spec's explicit rejection condition, identical in kind to LoveDA
(coordinate-free ML-ready tensors). Note the region is known only at country level
(Japan/India/Nepal/Taiwan), which is far too coarse to place a 640 m patch.

Access note: the data is publicly mirrored (Zenodo record 10463239; HuggingFace
``ibm-nasa-geospatial/Landslide4sense``), so it is NOT credential-gated — the blocker is
the missing georeferencing, not access.

Running this module re-verifies the rejection and (re)writes the rejection summary. It
writes nothing under weka ``datasets/`` other than the per-dataset registry_entry.json,
and never touches the central ``registry.json``.
"""

from pathlib import Path

from olmoearth_pretrain.open_set_segmentation_data import manifest

SLUG = "landslide4sense"
NAME = "Landslide4Sense"
ZENODO_RECORD = "10463239"
URL = "https://www.iarai.ac.at/landslide4sense/"
GITHUB = "https://github.com/iarai/Landslide4Sense-2022"

SUMMARY_PATH = Path(
    "data/open_set_segmentation_data/"
    "dataset_summaries/landslide4sense.md"
)

IARAI_QUOTE = (
    "The detailed geographic information and acquisition time will not be released at "
    "the current phase in case participants may directly look for the corresponding "
    "high-resolution images to check."
)

REJECT_NOTE = (
    "no-georeferencing: coordinate-free HDF5 patches (128x128x14 arrays); IARAI "
    "deliberately withheld geographic info/coordinates, so patches cannot be placed on "
    "the S2 grid (like LoveDA). Publicly mirrored (Zenodo 10463239 / HF "
    "ibm-nasa-geospatial), so not credential-gated."
)

SUMMARY = f"""# Landslide4Sense — REJECTED (no recoverable georeferencing)

- **Slug**: `{SLUG}`
- **Name**: {NAME}
- **Source**: IARAI Landslide4Sense 2022 competition ({URL});
  code/data description {GITHUB}; public mirrors: Zenodo record {ZENODO_RECORD}
  (https://zenodo.org/records/{ZENODO_RECORD}), HuggingFace
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
   data description states verbatim: "{IARAI_QUOTE}"
3. **Region is country-level only.** The manifest region is "Japan, India, Nepal, Taiwan"
   — far too coarse to place a ~640 m patch on the S2 grid even approximately.

Because per-patch geocoordinates are unrecoverable, the patches cannot be located on the
S2 grid — the spec's stated rejection condition ("No recoverable geocoordinates"; the
guidance explicitly names coordinate-free HDF5 arrays, "reject like LoveDA").

## Access method (for the record)

Publicly available without credentials via Zenodo record {ZENODO_RECORD} and the
HuggingFace mirror `ibm-nasa-geospatial/Landslide4sense`, so this is **not** an
`iarai` credential rejection — the sole blocker is the missing georeferencing. Nothing
was written under weka `datasets/` (rejection path).

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.landslide4sense
```

This re-verifies the rejection and re-writes this summary; it makes no dataset outputs.
"""


def main() -> None:
    print(f"{NAME}: data description at {GITHUB}")
    print(f'IARAI withheld geographic information (verbatim): "{IARAI_QUOTE}"')
    print(
        "HDF5 patches are coordinate-free raw arrays (128x128x14 image, 128x128 mask); "
        "no CRS/geotransform/lon-lat anywhere in the Zenodo/HF/GitHub distributions."
    )

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(SUMMARY)
    print(f"Wrote rejection summary -> {SUMMARY_PATH}")

    manifest.write_registry_entry(SLUG, "rejected", notes=REJECT_NOTE)
    print("Wrote registry_entry.json (status=rejected).")
    print(
        "STATUS: rejected — reason: no recoverable georeferencing "
        "(coordinate-free HDF5 patches; IARAI deliberately withheld coordinates)."
    )


if __name__ == "__main__":
    main()
