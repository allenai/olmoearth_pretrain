"""Process OlmoEarth US tree genus into open-set-segmentation label patches.

Source: local rslearn eval at olmoearth_evals/us_trees. Each window is one tree-inventory
point labeled with a plant genus (39 genera across the United States), with the genus name,
lon/lat, and a ~1-year time range stored in window metadata.json ``options``. Sparse point
segmentation, so we write one dataset-wide point table (points.geojson, spec 2a), balanced
to <=1000 per class (subject to the 25k per-dataset cap -> ~641/class for 39 classes).

Class ids are assigned by descending frequency (spec 5 top-by-frequency rule); the source
has only 39 genera, well under the 254-class uint8 cap, so no genus is dropped.
"""

import argparse
import json
import multiprocessing
import os
from collections import Counter
from typing import Any

from olmoearth_pretrain.open_set_segmentation_data import io, manifest

SLUG = "olmoearth_us_tree_genus"
NAME = "OlmoEarth US tree genus"
SOURCE = "/weka/dfive-default/rslearn-eai/datasets/olmoearth_evals/us_trees"
PER_CLASS = 1000

# Genus -> short description (common name). The source stores only the latin genus; these
# common-name glosses are added for readability in metadata.json.
GENUS_DESC = {
    "abies": "Fir (genus Abies).",
    "acer": "Maple (genus Acer).",
    "aesculus": "Buckeye / horse chestnut (genus Aesculus).",
    "ailanthus": "Tree-of-heaven (genus Ailanthus).",
    "alnus": "Alder (genus Alnus).",
    "amelanchier": "Serviceberry / shadbush (genus Amelanchier).",
    "asimina": "Pawpaw (genus Asimina).",
    "betula": "Birch (genus Betula).",
    "carya": "Hickory / pecan (genus Carya).",
    "cercis": "Redbud (genus Cercis).",
    "cornus": "Dogwood (genus Cornus).",
    "diospyros": "Persimmon (genus Diospyros).",
    "elaeagnus": "Silverberry / oleaster (genus Elaeagnus).",
    "fagus": "Beech (genus Fagus).",
    "gleditsia": "Honey locust (genus Gleditsia).",
    "ilex": "Holly (genus Ilex).",
    "juglans": "Walnut (genus Juglans).",
    "juniperus": "Juniper (genus Juniperus).",
    "liquidambar": "Sweetgum (genus Liquidambar).",
    "liriodendron": "Tulip tree / yellow-poplar (genus Liriodendron).",
    "maclura": "Osage orange (genus Maclura).",
    "magnolia": "Magnolia (genus Magnolia).",
    "morus": "Mulberry (genus Morus).",
    "picea": "Spruce (genus Picea).",
    "pinus": "Pine (genus Pinus).",
    "populus": "Poplar / cottonwood / aspen (genus Populus).",
    "prosopis": "Mesquite (genus Prosopis).",
    "prunus": "Cherry / plum (genus Prunus).",
    "pseudotsuga": "Douglas-fir (genus Pseudotsuga).",
    "quercus": "Oak (genus Quercus).",
    "sabal": "Palmetto (genus Sabal).",
    "salix": "Willow (genus Salix).",
    "sassafras": "Sassafras (genus Sassafras).",
    "taxodium": "Bald cypress (genus Taxodium).",
    "thuja": "Arborvitae / white cedar (genus Thuja).",
    "triadica": "Chinese tallow (genus Triadica).",
    "tsuga": "Hemlock (genus Tsuga).",
    "ulmus": "Elm (genus Ulmus).",
    "yucca": "Yucca (genus Yucca).",
}


def scan_records() -> list[dict[str, Any]]:
    """Parallel-read window metadata into flat records."""
    jobs = []
    windows_root = os.path.join(SOURCE, "windows")
    for group in os.listdir(windows_root):
        gd = os.path.join(windows_root, group)
        if os.path.isdir(gd):
            for name in os.listdir(gd):
                jobs.append(os.path.join(gd, name))
    with multiprocessing.Pool(64) as p:
        recs = [r for r in p.map(_read_one, jobs, chunksize=128) if r]
    return recs


def _read_one(path: str) -> dict[str, Any] | None:
    try:
        with open(os.path.join(path, "metadata.json")) as f:
            md = json.load(f)
    except FileNotFoundError:
        return None
    opt = md.get("options", {})
    tr = md.get("time_range")
    if opt.get("lon") is None or not opt.get("label"):
        return None
    year = int(tr[0][:4]) if tr else None
    if year is None or year < 2016:  # anchor to Sentinel era
        return None
    return {
        "lon": opt["lon"],
        "lat": opt["lat"],
        "label": opt["label"],
        "year": year,
        "source_id": f"{os.path.basename(os.path.dirname(path))}/{os.path.basename(path)}",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(f"local rslearn dataset: {SOURCE}\n")

    recs = scan_records()
    print(f"scanned {len(recs)} labeled points")

    # Assign class ids by descending frequency (spec 5 top-by-frequency rule). Only 39
    # genera here (< 254 uint8 cap), so all are kept; ties broken by name for determinism.
    freq = Counter(r["label"] for r in recs)
    ordered = sorted(freq, key=lambda g: (-freq[g], g))
    name_to_id = {g: i for i, g in enumerate(ordered)}
    print(f"{len(ordered)} genera (uint8 cap 254 not exceeded; none dropped)")

    from olmoearth_pretrain.open_set_segmentation_data.sampling import balance_by_class

    selected = balance_by_class(recs, "label", per_class=PER_CLASS)
    print(f"selected {len(selected)} points (<= {PER_CLASS}/class, 25k total cap)")

    points = []
    for i, r in enumerate(selected):
        points.append(
            {
                "id": f"{i:06d}",
                "lon": r["lon"],
                "lat": r["lat"],
                "label": name_to_id[r["label"]],
                "time_range": io.year_range(r["year"]),
                "source_id": r["source_id"],
            }
        )
    io.write_points_table(SLUG, "classification", points)

    sel_counts = Counter(r["label"] for r in selected)
    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "olmoearth",
            "license": "internal",
            "provenance": {
                "url": SOURCE,
                "have_locally": True,
                "annotation_method": "derived (tree inventory)",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {"id": i, "name": g, "description": GENUS_DESC.get(g)}
                for i, g in enumerate(ordered)
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(selected),
            "class_counts": {g: sel_counts.get(g, 0) for g in ordered},
            "notes": (
                "1x1 point-segmentation labels; genus per point. All source splits "
                "(train+test) used; ~1-year time range anchored on the labeled year "
                "(2017-2022, all post-2016). Class ids assigned by descending genus "
                "frequency; 39 genera, none dropped (uint8 254-class cap not reached)."
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
