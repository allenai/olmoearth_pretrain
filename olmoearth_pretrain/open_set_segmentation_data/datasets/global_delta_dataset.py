"""Process the Global Delta Dataset (Nienhuis et al. 2020, Nature) into open-set labels.

Source: https://github.com/jhnienhuis/GlobalDeltaChange (MIT / CC-BY-4.0). A global
inventory of ~10,848 river deltas. Each delta has a river-mouth lon/lat and three modeled
sediment fluxes -- QWave, QRiver (pristine), QTide -- whose relative magnitude defines the
delta's morphology. The published classification (validation/global_delta_validation.m) is
simply the dominant flux:

    [~, morphology] = max([QWave, QRiver_prist, QTide], [], 2)
    -> ["Wave dominated", "River dominated", "Tide dominated"]

Encoding decisions (see summary):
- The morphology dominance is a single per-delta attribute attached to the river-mouth
  point. The full inventory provides reliable *points* for all deltas (polygons exist only
  for the 100 largest, in land_area_change/GlobalDeltaMax100_poly.kml). We therefore encode
  each delta as one sparse point at its river mouth -> one dataset-wide points.geojson
  (spec 2a), NOT per-sample GeoTIFFs. Deltas are large landforms, so the S2/S1/Landsat
  context around the mouth carries the morphological signal at 10-30 m.
- The manifest lists a generic "delta" class plus the three dominance classes. Every point
  is a delta and is exactly one dominance type, so a per-point label uses only the three
  morphology classes (the "delta" class is the umbrella of all points, not a separate id).
- The land/water change component (GSW/Aquamonitor) is a multi-year change with no precise
  event date, so it is NOT encoded as a change label (change_time=null); we keep the static
  morphology classification with a representative 1-year Sentinel-era window (2016).

Class ids are assigned by descending frequency:
    0 = wave-dominated (8245), 1 = river-dominated (1825), 2 = tide-dominated (778).

Run: python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.global_delta_dataset
"""

import argparse
from collections import Counter

import h5py
import numpy as np

from olmoearth_pretrain.open_set_segmentation_data import download, io, manifest

SLUG = "global_delta_dataset"
MAT_URL = (
    "https://github.com/jhnienhuis/GlobalDeltaChange/raw/master/GlobalDeltaData.mat"
)
PER_CLASS = 1000
YEAR = 2016  # representative Sentinel-era 1-year window (static morphology label)

# id -> (name, description). Order = descending global frequency.
CLASSES = [
    (
        "wave-dominated",
        "River delta whose morphology is dominated by wave-driven sediment flux "
        "(QWave > QRiver and QTide); typically smooth, cuspate/arcuate shorelines.",
    ),
    (
        "river-dominated",
        "River delta whose morphology is dominated by fluvial sediment flux "
        "(QRiver > QWave and QTide); typically protruding, birdsfoot/lobate forms.",
    ),
    (
        "tide-dominated",
        "River delta whose morphology is dominated by tidal energy flux "
        "(QTide > QWave and QRiver); typically funnel-shaped estuaries with tidal bars.",
    ),
]
# Morphology column order in max([QWave, QRiver_prist, QTide]) -> class id.
COL_TO_ID = {0: 0, 1: 1, 2: 2}  # wave, river, tide already match class ids above


def load_records(mat_path: str) -> list[dict]:
    """Read river-mouth coords + dominant flux for every delta."""
    with h5py.File(mat_path, "r") as f:
        lon = np.asarray(f["MouthLon"][0], dtype=np.float64)
        lat = np.asarray(f["MouthLat"][0], dtype=np.float64)
        qwave = np.asarray(f["QWave"][0], dtype=np.float64)
        qriver = np.asarray(f["QRiver_prist"][0], dtype=np.float64)
        qtide = np.asarray(f["QTide"][0], dtype=np.float64)

    # MouthLon is stored in [0, 360); convert to [-180, 180).
    lon = ((lon + 180.0) % 360.0) - 180.0

    q = np.vstack([qwave, qriver, qtide]).T  # cols: wave, river, tide
    dom_col = np.argmax(np.nan_to_num(q, nan=-np.inf), axis=1)

    records = []
    for i in range(lon.shape[0]):
        if not (np.isfinite(lon[i]) and np.isfinite(lat[i])):
            continue
        if not np.isfinite(q[i]).any():
            continue
        records.append(
            {
                "lon": float(lon[i]),
                "lat": float(lat[i]),
                "label": COL_TO_ID[int(dom_col[i])],
                "source_id": f"delta_{i}",
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    mat_path = raw / "GlobalDeltaData.mat"
    download.download_http(MAT_URL, mat_path)

    records = load_records(str(mat_path))
    print(f"loaded {len(records)} deltas")
    print("raw class counts:", Counter(r["label"] for r in records))

    from olmoearth_pretrain.open_set_segmentation_data.sampling import balance_by_class

    selected = balance_by_class(records, "label", per_class=PER_CLASS)
    counts = Counter(r["label"] for r in selected)
    print(f"selected {len(selected)} (<= {PER_CLASS}/class):", dict(counts))

    points = [
        {
            "id": f"{i:06d}",
            "lon": r["lon"],
            "lat": r["lat"],
            "label": r["label"],
            "time_range": io.year_range(YEAR),
            "change_time": None,
            "source_id": r["source_id"],
        }
        for i, r in enumerate(selected)
    ]
    io.write_points_table(SLUG, "classification", points)

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": "Global Delta Dataset",
            "task_type": "classification",
            "source": "GitHub (jhnienhuis/GlobalDeltaChange); Nienhuis et al. 2020, Nature",
            "license": "MIT / CC-BY-4.0",
            "provenance": {
                "url": "https://github.com/jhnienhuis/GlobalDeltaChange",
                "have_locally": False,
                "annotation_method": "automated model (WBMSED river, WaveWatch wave, TOPEX tide fluxes); dominant flux = morphology",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {"id": i, "name": name, "description": desc}
                for i, (name, desc) in enumerate(CLASSES)
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(selected),
            "class_counts": {
                CLASSES[i][0]: counts.get(i, 0) for i in range(len(CLASSES))
            },
            "notes": (
                "Sparse points at delta river mouths; label = dominant sediment flux "
                "(argmax of QWave/QRiver_prist/QTide) per Nienhuis et al. 2020. Static "
                "morphology -> representative 1-year window (2016), change_time=null. "
                "Multi-year land/water change component intentionally not encoded as a "
                "change label. Balanced to <=1000/class."
            ),
        },
    )
    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=len(selected)
    )
    print("done")


if __name__ == "__main__":
    main()
