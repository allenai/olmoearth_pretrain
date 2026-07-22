"""Triage TimeSen2Crop for open-set segmentation -> REJECTED (no geocoordinates).

Source: TimeSen2Crop (Weikmann, Paris & Bruzzone, IEEE JSTARS 2021), a pixel-based
dataset of >1.1M Sentinel-2 daily-interpolated time series covering Austria for the
2017-2018 agronomic year, with per-pixel crop-type labels. The manifest points at the
MONSTER reformat on Hugging Face (monster-monash/TimeSen2Crop).

Why this dataset is rejected
----------------------------
Placing a label on the S2 grid requires each sample's lon/lat (or an S2 tile + pixel
row/col from which lon/lat can be recovered). TimeSen2Crop provides neither:

* HF MONSTER version files: ``TimeSen2Crop_X.npy`` (N x 9 x 365 spectral time series),
  ``TimeSen2Crop_y.npy`` (crop class id), ``TimeSen2Crop_meta.npy`` (a single int64 per
  sample in 0..14 = which of the 15 Sentinel-2 MGRS tiles the pixel came from). The tile
  id localizes a pixel only to a ~110 x 110 km tile; there is no within-tile pixel index.
* Original Zenodo release (record 4715631): the dataset is organized hierarchically as
  ``<tile>/<crop_class>/<n>.csv`` where each CSV is one pixel's multispectral temporal
  signature (rows = acquisition dates, columns = bands + a clear/cloud/shadow/snow flag).
  Samples are anonymized running-index CSVs; no coordinate, no pixel row/col is stored.

So the pixel time series cannot be georeferenced to a specific 10 m pixel. Per the task
contract (points / pixel time series without recoverable geocoordinates cannot be placed
on the S2 grid) this dataset is REJECTED.

This script downloads the small metadata artifacts to raw/ for provenance and re-verifies
the absence of coordinates. It writes nothing to datasets/ and does not touch the
registry. Idempotent.
"""

import numpy as np

from olmoearth_pretrain.open_set_segmentation_data import download, io

SLUG = "timesen2crop"
HF_REPO = "monster-monash/TimeSen2Crop"
ZENODO_RECORD = "4715631"
REJECT_REASON = (
    "no-geocoordinates: pixel time series carry no lon/lat and no within-tile pixel "
    "index (HF meta gives only the S2 tile id 0-14; original Zenodo stores anonymized "
    "numbered CSVs per tile/class). Cannot place samples on the S2 grid."
)


def main() -> None:
    io.check_disk()

    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            f"Hugging Face: https://huggingface.co/datasets/{HF_REPO}\n"
            f"Original: https://zenodo.org/records/{ZENODO_RECORD}\n"
        )

    # Pull the small metadata artifacts (not the multi-GB X.npy) to verify.
    download.hf_download(HF_REPO, "README.md", raw)
    download.hf_download(HF_REPO, "TimeSen2Crop.py", raw)
    meta_path = download.hf_download(HF_REPO, "TimeSen2Crop_meta.npy", raw)
    y_path = download.hf_download(HF_REPO, "TimeSen2Crop_y.npy", raw)

    meta = np.load(str(meta_path), allow_pickle=True)
    y = np.load(str(y_path))

    print(
        f"meta: shape={meta.shape} dtype={meta.dtype} unique={sorted(set(meta.tolist()))}"
    )
    print(f"y: shape={y.shape} dtype={y.dtype} num_classes={len(set(y.tolist()))}")

    # Coordinate check: meta is one int64 (tile id) per sample; there is no lon/lat and no
    # per-pixel index anywhere in the release.
    has_coordinates = meta.ndim > 1 or meta.dtype.names is not None
    assert not has_coordinates, "unexpected: meta appears to carry structured coords"

    print(f"REJECTED {SLUG}: {REJECT_REASON}")
    print(
        "Nothing written to datasets/; registry left untouched (parent updates status)."
    )


if __name__ == "__main__":
    main()
