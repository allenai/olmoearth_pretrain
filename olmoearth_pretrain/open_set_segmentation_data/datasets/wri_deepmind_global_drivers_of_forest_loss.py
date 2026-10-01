"""Process the WRI/DeepMind Global Drivers of Forest Loss point label sets.

Source: Zenodo record 15366671 ("Global drivers of forest loss at 1 km resolution",
Sims et al. / WRI + Google DeepMind, ERL). The record ships a 1 km derived-product
raster AND the photointerpreted interpreter train/validation label *points* used to
train/validate the CNN. Per the manifest note and spec §1 (prefer reference over derived
maps), we use the POINTS, not the 1 km raster.

Each point carries the dominant driver of tree-cover loss at that location (a 7-class
scheme). The point is a single 10 m location with a class id -> pure sparse-point
segmentation, so we write one dataset-wide GeoJSON point table (spec §2a), balanced to
<=1000 per class.

Time handling (spec §5): the points are NOT dated to a loss year in their properties (the
product spans 2001-2024). The driver is treated as a persistent land-use STATE, so each
point gets a static 1-year Sentinel-era window (2020) with change_time=null. See the
dataset summary for the caveat about event-like classes (wildfire / other natural
disturbances).
"""

import argparse
import json
import multiprocessing
from collections import Counter
from typing import Any

from olmoearth_pretrain.open_set_segmentation_data import io, manifest
from olmoearth_pretrain.open_set_segmentation_data.sampling import balance_by_class

SLUG = "wri_deepmind_global_drivers_of_forest_loss"
PER_CLASS = 1000
STATIC_YEAR = 2020  # representative Sentinel-era window for a static land-use state.

SOURCE_FILES = [
    ("training", "training_2001_2024_v1_2.geojson"),
    ("validation", "validation_2001_2022.geojson"),
]

# Source Driver_primary_code (1-7) -> (class id, name, description). The manifest's 7
# classes are exactly codes 1-7 in order. Code 8 ("Noise/non-forest") is an annotation
# -quality flag (non-forest / mislabeled; training-only), not a semantic driver -> dropped.
CODE_TO_CLASS = {
    1: (
        0,
        "Permanent agriculture",
        "Tree-cover loss converted to long-term agricultural land use (cropland, pasture, "
        "tree crops/orchards) that does not revert to forest.",
    ),
    2: (
        1,
        "Hard commodities",
        "Loss driven by mining and energy infrastructure (mineral extraction, oil and gas).",
    ),
    3: (
        2,
        "Shifting cultivation",
        "Small-to-medium clearing for temporary/swidden agriculture within a forest matrix, "
        "typically rotational and followed by regrowth.",
    ),
    4: (
        3,
        "Logging",
        "Forestry / timber harvesting within managed forests (clear-cut or selective) where "
        "forest is expected to regrow.",
    ),
    5: (
        4,
        "Wildfire",
        "Tree-cover loss due to fire not associated with land-use conversion.",
    ),
    6: (
        5,
        "Settlements & infrastructure",
        "Loss from expansion of built-up areas, roads, and other infrastructure.",
    ),
    7: (
        6,
        "Other natural disturbances",
        "Loss from other, non-fire natural causes (storms, windthrow, flooding, pests, "
        "landslides).",
    ),
}
CLASSES = [CODE_TO_CLASS[c][1:] for c in sorted(CODE_TO_CLASS)]  # ordered by id 0..6


def scan_records() -> list[dict[str, Any]]:
    """Read both point files into flat records (codes 1-7 only)."""
    raw = io.raw_dir(SLUG)
    recs: list[dict[str, Any]] = []
    for split, fn in SOURCE_FILES:
        with (raw / fn).open() as f:
            fc = json.load(f)
        for feat in fc["features"]:
            p = feat["properties"]
            code = p.get("Driver_primary_code")
            if code not in CODE_TO_CLASS:
                continue  # drop code 8 (Noise/non-forest) and anything else
            lon, lat = feat["geometry"]["coordinates"]
            cid, name, _desc = CODE_TO_CLASS[code]
            recs.append(
                {
                    "lon": float(lon),
                    "lat": float(lat),
                    "label": cid,
                    "driver_name": name,
                    "confidence": p.get("Confidence_primary"),
                    "region": p.get("Region"),
                    "source_id": f"{split}/{p.get('ID')}",
                }
            )
    return recs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    recs = scan_records()
    print(f"scanned {len(recs)} labeled driver points (codes 1-7)")
    total_counts = Counter(r["label"] for r in recs)
    print("available per class:", dict(sorted(total_counts.items())))

    selected = balance_by_class(recs, "label", per_class=PER_CLASS)
    print(f"selected {len(selected)} (<= {PER_CLASS}/class)")

    time_range = io.year_range(STATIC_YEAR)
    points = []
    for i, r in enumerate(selected):
        points.append(
            {
                "id": f"{i:06d}",
                "lon": r["lon"],
                "lat": r["lat"],
                "label": r["label"],
                "time_range": time_range,
                "change_time": None,
                "source_id": r["source_id"],
                "driver_name": r["driver_name"],
                "confidence": r["confidence"],
                "region": r["region"],
            }
        )
    io.write_points_table(SLUG, "classification", points)

    counts = Counter(r["label"] for r in selected)
    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": "WRI/DeepMind Global Drivers of Forest Loss",
            "task_type": "classification",
            "source": "Zenodo / ERL",
            "license": "CC-BY-4.0",
            "provenance": {
                "url": "https://zenodo.org/records/15366671",
                "have_locally": False,
                "annotation_method": "photointerpretation + CNN (interpreter train/validation points)",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {"id": i, "name": name, "description": desc}
                for i, (name, desc) in enumerate(CLASSES)
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(selected),
            "class_counts": {
                CODE_TO_CLASS[c][1]: counts.get(CODE_TO_CLASS[c][0], 0)
                for c in sorted(CODE_TO_CLASS)
            },
            "notes": (
                "Sparse-point classification (points.geojson, 1x1). Uses the photointerpreted "
                "interpreter train+validation points (both splits), preferred over the 1 km "
                "raster per the manifest. Code 8 (Noise/non-forest) dropped. Driver treated as "
                "a persistent land-use state: static 1-year Sentinel-era window (2020), "
                "change_time=null (points are not dated to a loss year). Per-point confidence/"
                "region kept as auxiliary properties."
            ),
        },
    )
    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=len(selected)
    )
    print("done")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
