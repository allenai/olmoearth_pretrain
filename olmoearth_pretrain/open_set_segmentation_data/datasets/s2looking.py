"""Triage S2Looking for open-set-segmentation -> REJECTED (no georeferencing).

S2Looking (Shen et al., Remote Sensing 2021; arXiv:2107.09244) is a satellite
side-looking building-change-detection dataset: 5,000 registered bitemporal VHR
(0.5-0.8 m) image pairs over rural areas worldwide with 65,920+ annotated building-change
instances (newly built / demolished), collected 2017-2020.

The open-set-segmentation pipeline requires labels that can be placed on the Sentinel-2
UTM grid (co-located with S2/S1/Landsat imagery by geography + time). S2Looking cannot:

  * The public release is a single ``S2Looking.zip`` (~10.2 GB) laid out as
    ``train|val|test/{Image1,Image2,label,label1,label2}/*.png`` with opaque numeric
    filenames (e.g. ``1.png``). The paper states verbatim: "The image pairs in the dataset
    are converted from the original TIFF format with 16 bit to PNG format with 8 bit."
    8-bit PNG carries no CRS and no geotransform, and no world file / GeoTIFF / CSV /
    coordinate index ships with the archive.
  * The authors built the tiles by cropping scenes using "rough coordinate information,"
    but that "geographic coordinate information has been removed from the data." So
    per-tile lon/lat is unrecoverable from the release; region provenance is only "rural
    areas throughout the world."

So per-tile real-world coordinates are unrecoverable -> tiles cannot be resampled to 10 m
and placed on the S2 grid. This is the spec's explicit rejection condition (§8.2).

Two secondary blockers are moot given the above but recorded in the summary: (1) building
footprints at 0.5-0.8 m are sub-pixel at Sentinel-2 10 m (change signal unresolvable), and
(2) no per-pair acquisition dates exist (only a dataset-wide 2017-2020 range with a 1-3
year gap), so no valid ``change_time`` / <=1-year window could be assigned.

Running this module re-states the rejection and (re)writes the rejection summary. It writes
nothing under weka ``datasets/`` except this dataset's ``registry_entry.json`` (status
``rejected``); it never touches the central ``registry.json``.
"""

from pathlib import Path

from olmoearth_pretrain.open_set_segmentation_data import manifest

SLUG = "s2looking"
NAME = "S2Looking"
URL = "https://github.com/S2Looking/Dataset"

SUMMARY_PATH = Path(
    "data/open_set_segmentation_data/"
    "dataset_summaries/s2looking.md"
)

REJECT_NOTES = (
    "no-georef: released as coordinate-free 8-bit PNG pairs (converted from 16-bit TIFF) "
    "with opaque numeric filenames; authors removed the coordinate information and no "
    "per-tile lon/lat table ships with the archive, so tiles cannot be placed on the S2 "
    "grid. Secondary (moot): VHR 0.5-0.8 m building change is sub-pixel at 10 m and no "
    "per-pair dates exist for a valid change_time."
)


def main() -> None:
    manifest.write_registry_entry(SLUG, "in_progress")

    print(f"S2Looking triage ({URL})")
    print("Georeferencing check (cheap, before bulk download):")
    print(
        "  - Public release: single S2Looking.zip (~10.2 GB), layout "
        "train|val|test/{Image1,Image2,label,label1,label2}/*.png (numeric names)."
    )
    print(
        "  - Format: 8-bit PNG converted from original 16-bit TIFF -> no CRS/geotransform."
    )
    print(
        "  - Authors: 'geographic coordinate information has been removed from the data'."
    )
    print("  - No world file / GeoTIFF / CSV / coordinate index in the archive.")
    print("=> per-tile real-world coordinates are UNRECOVERABLE.")

    if not SUMMARY_PATH.exists():
        raise SystemExit(f"expected rejection summary at {SUMMARY_PATH}")
    print(f"Rejection summary present -> {SUMMARY_PATH}")

    manifest.write_registry_entry(SLUG, "rejected", notes=REJECT_NOTES)
    print(
        "STATUS: rejected — reason: no recoverable georeferencing (coordinate-free PNG "
        "tiles; authors removed coordinate info). VHR-vs-10m and missing change_time are "
        "secondary/moot blockers."
    )


if __name__ == "__main__":
    main()
