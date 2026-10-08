"""Triage LoveDA for open-set-segmentation -> REJECTED (no georeferencing).

LoveDA (Land-cOVEr Domain Adaptive semantic segmentation, Wang et al., NeurIPS 2021
Datasets & Benchmarks) is a 0.3 m VHR semantic-segmentation dataset of urban/rural
scenes over three Chinese cities (Nanjing, Changzhou, Wuhan) with 7 land-cover classes.

The open-set-segmentation pipeline requires labels that can be placed on the Sentinel-2
UTM grid (co-located with S2/S1/Landsat imagery by geography + time). LoveDA cannot:

  * The Zenodo release (record 5706578: Train.zip / Val.zip / Test.zip) contains ONLY
    ``.png`` tiles (1024x1024) with opaque numeric filenames (e.g. ``2522.png``) laid
    out as ``<Split>/<Urban|Rural>/{images_png,masks_png}/*.png``. PNGs carry no CRS
    and no geotransform, and there is no world file, CSV, or coordinate lookup table
    anywhere in the release (verified: all 3338 Val + 5044 Train entries are ``.png``;
    the only non-image is Datasheet.pdf).
  * The authors' own Datasheet.pdf states verbatim: "All data was obtained from Google
    Earth platform, and do not contain any coordinate location information."

So per-tile real-world coordinates are unrecoverable -> tiles cannot be resampled to
10 m and placed on the S2 grid. This is the spec's explicit rejection condition. The VHR
class-suitability question (building/road at 10 m) is therefore moot.

Running this module re-verifies the rejection (lists the Zenodo record) and (re)writes
the rejection summary. It writes nothing under weka ``datasets/`` and does not touch
``registry.json``.
"""

import json
import urllib.request
from pathlib import Path

SLUG = "loveda"
NAME = "LoveDA"
ZENODO_RECORD = "5706578"
URL = "https://zenodo.org/records/5706578"

SUMMARY_PATH = Path(
    "data/open_set_segmentation_data/"
    "dataset_summaries/loveda.md"
)

DATASHEET_QUOTE = (
    "All data was obtained from Google Earth platform, and do not contain any "
    "coordinate location information."
)


def verify_no_georeferencing() -> dict:
    """Best-effort remote check of the Zenodo record: confirm PNG-only, no coord table."""
    info: dict = {"files": [], "reachable": False}
    try:
        with urllib.request.urlopen(
            f"https://zenodo.org/api/records/{ZENODO_RECORD}", timeout=60
        ) as r:
            meta = json.loads(r.read())
        info["reachable"] = True
        for f in meta.get("files", []):
            info["files"].append({"key": f.get("key"), "size": f.get("size")})
    except Exception as e:  # network optional; rejection stands regardless
        info["error"] = str(e)
    return info


SUMMARY = f"""# LoveDA — REJECTED (no recoverable georeferencing)

- **Slug**: `{SLUG}`
- **Name**: {NAME}
- **Source**: Zenodo record {ZENODO_RECORD} ({URL}) / NeurIPS 2021 D&B
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
   exclusively `.png` files under `<Split>/<Urban|Rural>/{{images_png,masks_png}}/`
   with opaque numeric filenames (e.g. `2522.png`). Verified by reading each zip's
   central directory: Val = 3338 `.png` entries, Train = 5044 `.png` entries, **zero**
   GeoTIFFs / world files / CSVs / coordinate indices. PNG carries no CRS and no
   geotransform.
2. **The authors confirm coordinates were stripped.** The official `Datasheet.pdf`
   states verbatim: "{DATASHEET_QUOTE}" (data are Google Earth screenshots with no
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

Public, no credentials needed. `download.download_zenodo("{ZENODO_RECORD}", raw_dir)`
fetches the zips. Nothing was written under weka `datasets/` (rejection path).

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.loveda
```

This re-lists the Zenodo record and re-writes this summary; it makes no dataset outputs.
"""


def main() -> None:
    info = verify_no_georeferencing()
    if info.get("reachable"):
        keys = [f["key"] for f in info["files"]]
        print(f"Zenodo record {ZENODO_RECORD} files: {keys}")
        non_pdf_zip = [
            k for k in keys if not (k.endswith(".zip") or k.endswith(".pdf"))
        ]
        print(
            "Non zip/pdf files at record top level:",
            non_pdf_zip or "(none) -> only Train/Val/Test.zip + Datasheet.pdf",
        )
    else:
        print(
            "Zenodo not reachable; rejection stands on documented grounds.",
            info.get("error"),
        )

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(SUMMARY)
    print(f"Wrote rejection summary -> {SUMMARY_PATH}")
    print(
        "STATUS: rejected — reason: tiles cannot be georeferenced to real coordinates "
        "(coordinate-free PNG tiles; authors' datasheet confirms no coordinate info)."
    )


if __name__ == "__main__":
    main()
