"""Process OlmoEarth Kenya Nandi crop type into open-set-segmentation label points.

Source: local rslearn eval at crop/kenya_nandi/20250625. Smallholder crop/land-cover
type reference for Nandi County, Kenya, collected by manual field survey (crop types)
plus homogeneous ESA-WorldCover-derived Water/Built-up context points. Each window is a
32x32 patch built centered on one reference point; the labeled pixel is the center pixel
(verified 400/400), so the point is the center of the window bounds. Window
``metadata.json`` carries the ``category`` (class) and a UTM projection/bounds; some
windows use UTM 36N (EPSG:32636) and some UTM 36S (EPSG:32736), so we reproject each
window's own CRS to WGS84 (the window-name lon/lat string is unreliable -- do not use it).

Sparse point segmentation => one dataset-wide GeoJSON point table (points.geojson, spec
2a), balanced to <=1000 per class. All labels observed in the 2023 growing season, so
each point gets a 1-year (2023) time range.
"""

import argparse
import json
import multiprocessing
import os
from collections import Counter
from typing import Any

from pyproj import Transformer

from olmoearth_pretrain.open_set_segmentation_data import io, manifest
from olmoearth_pretrain.open_set_segmentation_data.sampling import balance_by_class

SLUG = "olmoearth_kenya_nandi_crop_type"
SOURCE = "/weka/dfive-default/rslearn-eai/datasets/crop/kenya_nandi/20250625"
PER_CLASS = 1000
LABEL_YEAR = 2023  # all reference points observed in the 2023 growing season

# Unified class scheme: the 6 manifest crop types first (ids 0-5), then two additional
# crop types present in the source (Legumes, Vegetables), then two ESA-WorldCover-derived
# land-cover context classes (Water, Built-up). Descriptions are short definitions.
CLASSES = [
    ("Coffee", "Perennial Coffea shrub plots (smallholder coffee)."),
    (
        "Trees",
        "Tree cover / woody perennials and agroforestry trees (non-crop woodland).",
    ),
    ("Grassland", "Grass-dominated herbaceous cover / pasture / rangeland."),
    ("Maize", "Maize (corn) cropland, the dominant annual staple crop."),
    ("Sugarcane", "Sugarcane plantations."),
    ("Tea", "Perennial tea (Camellia sinensis) plantations."),
    ("Legumes", "Legume crops (beans, pulses)."),
    ("Vegetables", "Mixed vegetable / horticultural cropland."),
    ("Water", "Open water (ESA WorldCover-derived homogeneous samples)."),
    (
        "Built-up",
        "Built-up / impervious surfaces (ESA WorldCover-derived homogeneous samples).",
    ),
]
NAME_TO_ID = {name: i for i, (name, _d) in enumerate(CLASSES)}

_TRANSFORMERS: dict[str, Transformer] = {}


def _transformer(crs: str) -> Transformer:
    if crs not in _TRANSFORMERS:
        _TRANSFORMERS[crs] = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    return _TRANSFORMERS[crs]


def _read_one(path: str) -> dict[str, Any] | None:
    """Read one window metadata.json -> flat record (lon/lat/label/source_id)."""
    try:
        with open(os.path.join(path, "metadata.json")) as f:
            md = json.load(f)
    except FileNotFoundError:
        return None
    cat = md.get("options", {}).get("category")
    if cat not in NAME_TO_ID:
        return None
    b = md["bounds"]
    crs = md["projection"]["crs"]
    # Labeled pixel is the center (16,16) of the 32-px window; center-pixel UTM coord.
    ux = (b[0] + 16.5) * 10.0
    uy = -(b[1] + 16.5) * 10.0
    lon, lat = _transformer(crs).transform(ux, uy)
    return {
        "lon": lon,
        "lat": lat,
        "label": cat,
        "source_id": f"{os.path.basename(os.path.dirname(path))}/{os.path.basename(path)}",
    }


def scan_records() -> list[dict[str, Any]]:
    jobs = []
    windows_root = os.path.join(SOURCE, "windows")
    for group in os.listdir(windows_root):
        gd = os.path.join(windows_root, group)
        if os.path.isdir(gd):
            for name in os.listdir(gd):
                jobs.append(os.path.join(gd, name))
    with multiprocessing.Pool(64) as p:
        recs = [r for r in p.map(_read_one, jobs, chunksize=64) if r]
    return recs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    parser.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(f"local rslearn dataset: {SOURCE}\n")

    recs = scan_records()
    print(f"scanned {len(recs)} labeled points")
    selected = balance_by_class(recs, "label", per_class=PER_CLASS)
    print(f"selected {len(selected)} (<= {PER_CLASS}/class)")

    points = []
    for i, r in enumerate(selected):
        points.append(
            {
                "id": f"{i:06d}",
                "lon": r["lon"],
                "lat": r["lat"],
                "label": NAME_TO_ID[r["label"]],
                "time_range": io.year_range(LABEL_YEAR),
                "source_id": r["source_id"],
            }
        )
    io.write_points_table(SLUG, "classification", points)

    counts = Counter(r["label"] for r in selected)
    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": "OlmoEarth Kenya Nandi crop type",
            "task_type": "classification",
            "source": "olmoearth",
            "license": "internal",
            "provenance": {
                "url": SOURCE,
                "have_locally": True,
                "annotation_method": (
                    "manual field survey (crop types); ESA WorldCover-derived homogeneous "
                    "samples (Water, Built-up)"
                ),
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {"id": i, "name": name, "description": desc}
                for i, (name, desc) in enumerate(CLASSES)
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(selected),
            "class_counts": {name: counts.get(name, 0) for name, _ in CLASSES},
            "notes": (
                "1x1 point-segmentation (points.geojson, spec 2a). Nandi County, Kenya. "
                "Coordinates derived from each window's own UTM CRS (32636/32736) + bounds "
                "center pixel; window-name lon/lat is unreliable. All labels observed in the "
                "2023 growing season -> 1-year (2023) time range (manifest listed 2024-2025 "
                "but on-disk data is 2023). Source has 8 crop categories (manifest listed 6) "
                "plus WorldCover Water/Built-up; all combined into one unified class scheme. "
                "All source splits used."
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
