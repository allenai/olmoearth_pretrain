"""Process EuroSAT into open-set-segmentation label patches (scene-level, spec sec 4).

Source (have_locally): the internal rslearn dataset at
``/weka/dfive-default/rslearn-eai/datasets/eurosat/rslearn_dataset`` (aka "small_eurosat"),
built from the *georeferenced* EuroSAT MS release (Helber et al. 2019/2018, Zenodo record
7711810). EuroSAT is 27,000 Sentinel-2 patches, each 64x64 @ 10 m (a coherent 640 m
footprint), hand-labeled into one of 10 land-use / land-cover categories. Crucially each
patch retains its real Sentinel-2 UTM CRS + bounds (verified below), so it can be placed on
the S2 grid.

Triage: ACCEPT. Per spec sec 4 (scene-level, e.g. EuroSAT): EuroSAT patches are genuinely
coherent land-cover patches (that is how the benchmark was constructed -- each 640 m tile is
a single homogeneous LULC class), so we emit ONE uniform-class 64x64 tile per patch rather
than rejecting it as mere patch classification. The tile is filled with the patch's single
class id and written at the patch's own UTM projection + pixel bounds (already UTM @ ~10 m in
the source window metadata -- we reuse it exactly, spec sec 2 "reuse the source window's CRS
if already UTM at 10 m"). Labels are 2018 (post-2016); georeferencing is present. Sparse-point
rules do not apply (label footprint is 640 m, >> 1 px).

Each source window's metadata.json carries: projection (crs + x/y_resolution ~= 10 m),
64x64 pixel bounds, a 1-year time_range (2018-01-01..2019-01-01, how the source materialized
its S2 imagery; the manifest lists 2017/2018 for acquisition -- either is a valid <=1 yr
window, we keep the source's 2018 anchor), and options.category (the LULC class). We do NOT
read the imagery or the label vector layer -- the class is in options.category and matches the
label feature.

Balancing (spec sec 5): tiles-per-class balanced to <= 1000/class over 10 classes = 10,000
tiles, well under the 25k cap. Every class has >= 2000 source windows so all reach 1000. All
source splits (train+val) are used.

Run:  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.eurosat
"""

import argparse
import multiprocessing
import os
from collections import Counter
from typing import Any

import numpy as np
from rasterio.crs import CRS
from rslearn.utils.geometry import Projection
from rslearn.utils.mp import star_imap_unordered

from olmoearth_pretrain.open_set_segmentation_data import io, manifest
from olmoearth_pretrain.open_set_segmentation_data.sampling import balance_by_class

SLUG = "eurosat"
NAME = "EuroSAT"
SOURCE = "/weka/dfive-default/rslearn-eai/datasets/eurosat/rslearn_dataset"
GROUP = "default"
PER_CLASS = 1000
TILE = 64  # native EuroSAT patch size (64x64 @ 10 m == 640 m), == MAX_TILE cap

# 10 LULC classes in manifest order -> id, with EuroSAT (Helber et al. 2019) definitions.
# The tuple is (source_category_folder_name, manifest_display_name, description).
CLASSES = [
    (
        "AnnualCrop",
        "Annual Crop",
        "Agricultural areas of annual (seasonal) crops -- arable cropland harvested and "
        "replanted within the year (e.g. cereals, root crops).",
    ),
    (
        "Forest",
        "Forest",
        "Land covered by trees / forest canopy (broadleaved, coniferous, or mixed woodland).",
    ),
    (
        "HerbaceousVegetation",
        "Herbaceous Vegetation",
        "Natural herbaceous vegetation -- grasslands and other non-cultivated low green "
        "vegetation not managed as pasture or crop.",
    ),
    (
        "Highway",
        "Highway",
        "Highways / major roads and their immediate transport corridor surroundings.",
    ),
    (
        "Industrial",
        "Industrial",
        "Industrial or commercial units -- factories, warehouses, and associated built-up "
        "impervious areas.",
    ),
    (
        "Pasture",
        "Pasture",
        "Pastures -- grassland managed for grazing / fodder production.",
    ),
    (
        "PermanentCrop",
        "Permanent Crop",
        "Permanent crops -- perennial plantations such as orchards, vineyards, and olive "
        "groves that are not replanted annually.",
    ),
    (
        "Residential",
        "Residential",
        "Residential built-up areas -- housing and associated urban fabric.",
    ),
    ("River", "River", "Rivers and other flowing inland watercourses."),
    ("SeaLake", "Sea/Lake", "Open water bodies -- seas, coastal water, and lakes."),
]
CAT_TO_ID = {cat: i for i, (cat, _name, _desc) in enumerate(CLASSES)}


def _read_one(path: str) -> dict[str, Any] | None:
    """Read one window's metadata.json into a flat record (fast, direct file read)."""
    import json

    try:
        with open(os.path.join(path, "metadata.json")) as f:
            md = json.load(f)
    except FileNotFoundError:
        return None
    opt = md.get("options", {})
    cat = opt.get("category")
    if cat not in CAT_TO_ID:
        return None
    return {
        "cls": CAT_TO_ID[cat],
        "crs": md["projection"]["crs"],
        "x_res": md["projection"]["x_resolution"],
        "y_res": md["projection"]["y_resolution"],
        "bounds": tuple(int(v) for v in md["bounds"]),
        "time_range": md["time_range"],
        "source_id": f"{GROUP}/{os.path.basename(path)}",
    }


def scan_records(workers: int = 64) -> list[dict[str, Any]]:
    """Parallel-scan all window metadata into records (weka small-file I/O -> use a Pool)."""
    windows_root = os.path.join(SOURCE, "windows", GROUP)
    jobs = [
        os.path.join(windows_root, name) for name in sorted(os.listdir(windows_root))
    ]
    with multiprocessing.Pool(workers) as p:
        recs = [r for r in p.map(_read_one, jobs, chunksize=64) if r]
    return recs


def _projection_and_bounds(rec: dict[str, Any]):
    """Reuse the source window's exact UTM projection + 64x64 pixel bounds (spec sec 2)."""
    proj = Projection(CRS.from_string(rec["crs"]), rec["x_res"], rec["y_res"])
    return proj, rec["bounds"]


def _write_one(rec: dict[str, Any]) -> None:
    from datetime import datetime

    sample_id = rec["sample_id"]
    tif = io.locations_dir(SLUG) / f"{sample_id}.tif"
    js = io.locations_dir(SLUG) / f"{sample_id}.json"
    if tif.exists() and js.exists():
        return
    proj, bounds = _projection_and_bounds(rec)
    arr = np.full(
        (bounds[3] - bounds[1], bounds[2] - bounds[0]), rec["cls"], dtype=np.uint8
    )
    io.write_label_geotiff(SLUG, sample_id, arr, proj, bounds)
    tr = [datetime.fromisoformat(t) for t in rec["time_range"]]
    io.write_sample_json(
        SLUG,
        sample_id,
        proj,
        bounds,
        (tr[0], tr[1]),
        change_time=None,
        source_id=rec["source_id"],
        classes_present=[rec["cls"]],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            "EuroSAT (Helber et al. 2018/2019), georeferenced MS release, Zenodo record "
            "7711810 (https://github.com/phelber/EuroSAT). License: MIT.\n"
            f"have_locally=true: source is the internal rslearn dataset (aka small_eurosat) "
            f"at {SOURCE}\n"
            "Raw NOT copied (spec sec 1). Each window's metadata.json provides the patch's "
            "real UTM projection + 64x64 bounds + 1-year time_range + options.category "
            "(LULC class). We emit one uniform-class 64x64 uint8 tile per patch.\n"
        )

    print("Scanning source windows...")
    recs = scan_records(args.workers)
    print(f"  {len(recs)} labeled patches")
    print("  source distribution:", dict(Counter(r["cls"] for r in recs)))
    io.check_disk()

    selected = balance_by_class(recs, "cls", per_class=PER_CLASS)
    selected.sort(key=lambda r: r["source_id"])
    for i, r in enumerate(selected):
        r["sample_id"] = f"{i:06d}"
    counts = Counter(r["cls"] for r in selected)
    print(
        f"  selected {len(selected)} tiles (<= {PER_CLASS}/class over {len(CLASSES)} classes)"
    )
    for cat, name, _ in CLASSES:
        cid = CAT_TO_ID[cat]
        print(f"    {cid:2d} {name:22s} {counts.get(cid, 0)}")

    print("Writing label tiles...")
    with multiprocessing.Pool(args.workers) as p:
        for _ in star_imap_unordered(p, _write_one, [{"rec": r} for r in selected]):
            pass

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "EuroSAT (GitHub / Zenodo)",
            "license": "MIT",
            "provenance": {
                "url": "https://github.com/phelber/EuroSAT",
                "have_locally": True,
                "annotation_method": "manual/photointerpretation",
                "internal_source": SOURCE,
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {"id": i, "name": name, "description": desc}
                for i, (_cat, name, desc) in enumerate(CLASSES)
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(selected),
            "class_counts": {
                name: counts.get(i, 0) for i, (_c, name, _d) in enumerate(CLASSES)
            },
            "notes": (
                "Scene-level EuroSAT: each 64x64 @ 10 m patch is a coherent 640 m LULC patch, "
                "emitted as one uniform-class uint8 tile at the patch's own UTM CRS/bounds. "
                "1-year time_range (2018-01-01..2019-01-01) from the source window; manifest "
                "lists 2017/2018 acquisition years -- either is a valid <=1 yr window. All "
                "source splits (train+val) used; balanced to <=1000/class."
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
